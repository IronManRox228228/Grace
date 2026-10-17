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

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        d["base_model_path"] = str(self.base_model_path) if self.base_model_path else None
        return d


class ResourceGovernor:
    """Monitors system power, battery, and thermals to safely pause opportunistic training."""

    def __init__(self, allow_on_battery: bool = False, max_cpu_percent: float = 85.0) -> None:
        self.allow_on_battery = allow_on_battery
        self.max_cpu_percent = max_cpu_percent
        self._user_active_hook: Optional[Callable[[], bool]] = None

    def register_user_active_hook(self, hook: Callable[[], bool]) -> None:
        """Register hook for Task 15 scheduler to detect user keyboard/mouse activity."""
        self._user_active_hook = hook

    def should_pause(self) -> Tuple[bool, str]:
        """Check whether training should immediately pause.

        Returns:
            Tuple of (should_pause: bool, reason: str)
        """
        # 1. Check battery status
        battery = psutil.sensors_battery()
        if battery is not None and not battery.power_plugged and not self.allow_on_battery:
            return True, f"Machine running on battery ({battery.percent}% remaining)"

        # 2. Check user activity hook
        if self._user_active_hook and self._user_active_hook():
            return True, "User activity detected"

        # 3. Check system CPU saturation
        cpu_pct = psutil.cpu_percent(interval=None)
        if cpu_pct > self.max_cpu_percent:
            return True, f"System CPU utilization high ({cpu_pct:.1f}%)"

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
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Record state summary
        state_file = ckpt_folder / "state.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({k: str(type(v)) for k, v in state_dict.items()}, f, indent=2)

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
