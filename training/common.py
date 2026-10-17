"""Common utilities and resource governance for Grace micromodel fleet fine-tunes (PLAN.md §3, §8, Task 12).

Provides:
- ResourceGovernor: checks battery, power state, and user idle status (Task 15 scheduler hooks)
- CheckpointManager: periodic atomic checkpoint saving and resumption
- ModelReplacementGate: deterministic evaluation on held-out test data ensuring a new model replaces the old only if it strictly scores better
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import psutil

# Add src and bench to sys.path
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bench.harness import append_result, create_benchmark_record
from grace.priority import set_background_task_priority

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Standardized training configuration for fleet models."""
    model_name: str
    output_dir: Path
    base_model_path: Optional[Path] = None
    batch_size: int = 4
    learning_rate: float = 2e-4
    epochs: int = 3
    checkpoint_interval_seconds: int = 180  # Save checkpoint every 3 minutes
    smoke_run: bool = False  # If True, runs 2 steps for smoke test
    max_steps: Optional[int] = None
    device: str = "cuda"  # Auto-falls back to cpu if cuda unavailable
    seed: int = 42
    allow_on_battery: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        d["base_model_path"] = str(self.base_model_path) if self.base_model_path else None
        return d


def get_windows_idle_seconds() -> float:
    """Get system idle time in seconds on Windows."""
    if sys.platform != "win32":
        return 0.0
    try:
        import ctypes
        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
            return max(0.0, millis / 1000.0)
    except Exception:
        pass
    return 0.0


def get_windows_thermal_status() -> Tuple[float, bool]:
    """Get CPU/GPU temperature and throttling status.
    Returns (temperature_celsius, is_throttling).
    """
    temp = 45.0
    throttling = False
    try:
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                all_current = [
                    entry.current for entries in temps.values()
                    for entry in entries if hasattr(entry, "current") and entry.current is not None
                ]
                if all_current:
                    temp = max(all_current)
        freq = psutil.cpu_freq()
        if freq and freq.max > 0:
            if freq.current < (freq.max * 0.50):
                throttling = True
    except Exception:
        pass
    return temp, throttling


class ResourceGovernor:
    """Monitors system power, battery, and thermals to safely pause opportunistic training."""

    def __init__(
        self,
        allow_on_battery: bool = False,
        max_cpu_percent: float = 85.0,
        idle_threshold_seconds: Optional[float] = None,
        max_temp_celsius: float = 80.0,
        power_provider: Optional[Callable[[], Tuple[bool, int]]] = None,
        idle_provider: Optional[Callable[[], float]] = None,
        thermal_provider: Optional[Callable[[], Tuple[float, bool]]] = None,
        turn_active_provider: Optional[Callable[[], bool]] = None,
        clock_provider: Optional[Callable[[], float]] = None,
        cpu_percent_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        self.allow_on_battery = allow_on_battery
        self.max_cpu_percent = max_cpu_percent
        self.idle_threshold_seconds = idle_threshold_seconds
        self.max_temp_celsius = max_temp_celsius
        self._user_active_hook: Optional[Callable[[], bool]] = None
        self.power_provider = power_provider
        self.idle_provider = idle_provider
        self.thermal_provider = thermal_provider
        self.turn_active_provider = turn_active_provider
        self.clock_provider = clock_provider or time.time
        self.cpu_percent_provider = cpu_percent_provider

    def register_user_active_hook(self, hook: Callable[[], bool]) -> None:
        """Register hook for Task 15 scheduler to detect user keyboard/mouse activity."""
        self._user_active_hook = hook

    def should_pause(self) -> Tuple[bool, str]:
        """Check whether training should immediately pause.

        Returns:
            Tuple of (should_pause: bool, reason: str)
        """
        # 1. Check battery status
        if self.power_provider:
            plugged, pct = self.power_provider()
            if not plugged and not self.allow_on_battery:
                return True, f"Machine running on battery ({pct}% remaining)"
        else:
            try:
                battery = psutil.sensors_battery()
                if battery is not None and not battery.power_plugged and not self.allow_on_battery:
                    return True, f"Machine running on battery ({battery.percent}% remaining)"
            except Exception:
                pass

        # 2. Check Grace active turn
        if self.turn_active_provider and self.turn_active_provider():
            return True, "Grace turn active"

        # 3. Check user activity hook
        if self._user_active_hook and self._user_active_hook():
            return True, "User activity detected"

        # 4. Check user idle time if configured
        if self.idle_threshold_seconds is not None:
            if self.idle_provider:
                idle_secs = self.idle_provider()
                if idle_secs < self.idle_threshold_seconds:
                    return True, f"User active recently ({idle_secs:.1f}s ago < {self.idle_threshold_seconds}s threshold)"
            elif sys.platform == "win32":
                idle_secs = get_windows_idle_seconds()
                if 0.0 <= idle_secs < self.idle_threshold_seconds:
                    return True, f"User active recently ({idle_secs:.1f}s ago < {self.idle_threshold_seconds}s threshold)"

        # 5. Check thermals and throttling
        if self.thermal_provider:
            temp, throttled = self.thermal_provider()
            if temp > self.max_temp_celsius:
                return True, f"System temperature high ({temp:.1f}°C > {self.max_temp_celsius:.1f}°C)"
            if throttled:
                return True, "System thermal throttling detected"
        else:
            temp, throttled = get_windows_thermal_status()
            if temp > self.max_temp_celsius:
                return True, f"System temperature high ({temp:.1f}°C > {self.max_temp_celsius:.1f}°C)"
            if throttled:
                return True, "System thermal throttling detected"

        # 6. Check system CPU saturation
        try:
            if self.cpu_percent_provider is not None:
                cpu_pct = self.cpu_percent_provider()
            else:
                cpu_pct = psutil.cpu_percent(interval=None)
            if cpu_pct > self.max_cpu_percent:
                return True, f"System CPU utilization high ({cpu_pct:.1f}%)"
        except Exception:
            pass

        return False, "ok"

    def apply_training_priority(self) -> None:
        """Apply BELOW_NORMAL priority class with Windows EcoQoS to training process."""
        set_background_task_priority()


class CheckpointManager:
    """Handles atomic saving and resumption of training state."""

    def __init__(self, checkpoint_dir: Path, interval_seconds: int = 180) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_seconds
        self.last_save_time = time.time()

    def should_save(self) -> bool:
        """Check if checkpoint interval has elapsed."""
        return (time.time() - self.last_save_time) >= self.interval_seconds

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        state_dict: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Atomically persist checkpoint state."""
        meta = {
            "step": step,
            "epoch": epoch,
            "timestamp": time.time(),
            "metrics": metrics or {},
        }
        ckpt_folder = self.checkpoint_dir / f"checkpoint_step_{step}"
        ckpt_folder.mkdir(parents=True, exist_ok=True)

        meta_file = ckpt_folder / "meta.json"
        tmp_meta = meta_file.with_suffix(".tmp")
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        tmp_meta.replace(meta_file)

        # Record state contents or serialized representation
        state_file = ckpt_folder / "state.json"
        tmp_state = state_file.with_suffix(".tmp")
        try:
            with open(tmp_state, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2)
        except (TypeError, ValueError):
            with open(tmp_state, "w", encoding="utf-8") as f:
                json.dump({k: str(v) for k, v in state_dict.items()}, f, indent=2)
        tmp_state.replace(state_file)

        self.last_save_time = time.time()
        logger.info("Saved checkpoint at step %d to %s", step, ckpt_folder)
        return ckpt_folder

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest valid checkpoint folder."""
        if not self.checkpoint_dir.is_dir():
            return None
        ckpts = [
            d for d in self.checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint_step_") and (d / "meta.json").is_file()
        ]
        if not ckpts:
            return None
        return max(ckpts, key=lambda p: (p / "meta.json").stat().st_mtime)

    def load_latest_checkpoint_meta(self) -> Optional[Dict[str, Any]]:
        latest = self.find_latest_checkpoint()
        if not latest:
            return None
        meta_file = latest / "meta.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_latest_checkpoint_state(self) -> Optional[Dict[str, Any]]:
        latest = self.find_latest_checkpoint()
        if not latest:
            return None
        state_file = latest / "state.json"
        if not state_file.is_file():
            return None
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)


class ModelReplacementGate:
    """Enforces strict rule: A newly trained model replaces the active model ONLY if it beats baseline."""

    def __init__(self, slot_name: str, baseline_metric_key: str = "accuracy") -> None:
        self.slot_name = slot_name
        self.baseline_metric_key = baseline_metric_key

    def should_replace(
        self,
        new_metric: float,
        current_baseline_metric: float,
        higher_is_better: bool = True,
    ) -> Tuple[bool, str]:
        if higher_is_better:
            improved = new_metric > current_baseline_metric
            margin = new_metric - current_baseline_metric
        else:
            improved = new_metric < current_baseline_metric
            margin = current_baseline_metric - new_metric

        if improved:
            msg = f"New model improved {self.slot_name} {self.baseline_metric_key} from {current_baseline_metric:.4f} to {new_metric:.4f} (+{margin:.4f}). APPROVED for replacement."
            return True, msg
        else:
            msg = f"New model did not beat baseline ({new_metric:.4f} vs {current_baseline_metric:.4f}). REJECTED from replacement."
            return False, msg
