"""Generic benchmark harness for Grace (PLAN.md §8, §9, Task 09).

Captures full machine profiles, runtime environment versions, model hashes,
thread settings, and system tier diagnostics. Appends structured records to
bench/results/*.jsonl for cross-model and cross-machine comparability.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

import psutil

# Add src to sys.path if not present so grace modules are importable
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from grace.thread_budget import calculate_thread_budget
from grace.tier import detect_hardware_tier

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def get_git_commit() -> str:
    """Return the current HEAD git commit hash or 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> Optional[str]:
    """Compute SHA-256 hash of a model or binary file."""
    p = Path(file_path)
    if not p.is_file():
        return None

    hasher = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.warning("Failed to compute SHA-256 for %s: %s", p, e)
        return None


def get_memory_channels() -> str:
    """Detect memory channels / populated slots on Windows."""
    if sys.platform == "win32":
        try:
            cmd = ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_PhysicalMemory).Count"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
            if res.returncode == 0 and res.stdout.strip().isdigit():
                return f"{res.stdout.strip()} populated slots"
        except Exception:
            pass
    return "unknown"


def get_machine_profile(background_load: str = "idle") -> Dict[str, Any]:
    """Gather machine hardware profile, memory, accelerators, and selected tier."""
    tier_diag = detect_hardware_tier()
    vm = psutil.virtual_memory()

    # Detect GPU visibility / status
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_disabled = cuda_env == "" or cuda_env == "-1"

    gpu_available = False
    try:
        import ctranslate2
        gpu_available = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor() or "Unknown CPU",
        "cpu_count_physical": psutil.cpu_count(logical=False) or 0,
        "cpu_count_logical": psutil.cpu_count(logical=True) or 0,
        "ram_total_gb": round(vm.total / (1024.0**3), 2),
        "ram_available_gb": round(vm.available / (1024.0**3), 2),
        "memory_channels": get_memory_channels(),
        "gpu_available": gpu_available,
        "gpu_disabled": gpu_disabled,
        "tier_chosen": tier_diag.tier.value,
        "tier_rationale": tier_diag.rationale,
        "background_load": background_load,
    }


def get_runtime_info() -> Dict[str, str]:
    """Return versions of active runtime libraries."""
    info = {
        "python": platform.python_version(),
    }

    for mod_name in ["ctranslate2", "faster_whisper", "sherpa_onnx", "onnxruntime", "numpy", "scipy"]:
        try:
            mod = __import__(mod_name)
            info[mod_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            info[mod_name] = "not installed"

    return info


def get_thread_settings(tier_str: str) -> Dict[str, Any]:
    """Return explicit thread budget parameters for the active tier."""
    from grace.tier import HardwareTier

    tier = HardwareTier(tier_str) if tier_str in [t.value for t in HardwareTier] else HardwareTier.FLOOR_CPU
    budget = calculate_thread_budget(tier=tier)
    return {
        "total_cores": budget.total_cores,
        "active_worker_threads": budget.active_worker_threads,
        "model_threads": budget.llama_threads,
        "intra_op_threads": budget.ort_intra_op_threads,
        "inter_op_threads": budget.ort_inter_op_threads,
        "allow_tts_lookahead": budget.allow_tts_lookahead,
        "rationale": budget.rationale,
    }


@dataclass
class BenchmarkRecord:
    """Standardized record for any benchmark evaluation run."""
    benchmark_name: str
    model_id: str
    file_hash: Optional[str]
    runtime: Dict[str, str]
    machine_profile: Dict[str, Any]
    thread_settings: Dict[str, Any]
    timestamp: str
    git_commit: str
    metrics: Dict[str, Any]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_benchmark_record(
    benchmark_name: str,
    model_id: str,
    metrics: Dict[str, Any],
    file_path: Optional[Union[str, Path]] = None,
    background_load: str = "idle",
    details: Optional[Dict[str, Any]] = None,
) -> BenchmarkRecord:
    """Factory creating a populated BenchmarkRecord."""
    machine = get_machine_profile(background_load=background_load)
    runtime = get_runtime_info()
    threads = get_thread_settings(machine["tier_chosen"])
    file_hash = compute_file_sha256(file_path) if file_path else None
    git_commit = get_git_commit()
    timestamp = datetime.now(timezone.utc).isoformat()

    return BenchmarkRecord(
        benchmark_name=benchmark_name,
        model_id=model_id,
        file_hash=file_hash,
        runtime=runtime,
        machine_profile=machine,
        thread_settings=threads,
        timestamp=timestamp,
        git_commit=git_commit,
        metrics=metrics,
        details=details or {},
    )


def append_result(
    benchmark_name: str,
    record: Union[BenchmarkRecord, Dict[str, Any]],
    results_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Append a benchmark record to bench/results/<benchmark_name>.jsonl.

    Returns the Path to the written JSONL file.
    """
    target_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / f"{benchmark_name}.jsonl"
    record_dict = record.to_dict() if isinstance(record, BenchmarkRecord) else record

    line = json.dumps(record_dict, ensure_ascii=False)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    logger.info("Appended benchmark result for %s to %s", record_dict.get("model_id"), file_path)
    return file_path


def load_results(
    benchmark_name: str,
    results_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Load all records from bench/results/<benchmark_name>.jsonl."""
    target_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    file_path = target_dir / f"{benchmark_name}.jsonl"
    if not file_path.is_file():
        return []

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
