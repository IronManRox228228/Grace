"""Hardware tier detection for Grace (PLAN.md §2).

Detects available hardware at startup and selects the runtime tier:
- BEST_NPU: Core Ultra, Ryzen AI, Snapdragon X (OpenVINO / ONNX Runtime NPU EP).
- MIDDLE_IGPU: Integrated GPU (DirectML / OpenVINO GPU).
- FLOOR_CPU: CPU floor (int8 ONNX, llama.cpp with quantized models).

Rule: Choose the tier from AVAILABLE RAM at startup, not total RAM.
A system with an iGPU but < 3.5 GB of free RAM will thrash if models are loaded into
GPU shared memory alongside active desktop applications; it must drop to the CPU floor.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import os
import subprocess
import sys
from enum import Enum
from typing import Any, NamedTuple, Optional, Tuple

logger = logging.getLogger("grace.tier")


class HardwareTier(str, Enum):
    BEST_NPU = "npu"
    BEST_DGPU = "dgpu"
    MIDDLE_IGPU = "igpu"
    FLOOR_CPU = "cpu"


class TierDiagnostics(NamedTuple):
    tier: HardwareTier
    available_ram_gb: float
    total_ram_gb: float
    has_npu: bool
    has_dgpu: bool
    has_igpu: bool
    override: Optional[str]
    rationale: str


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def get_memory_info() -> Tuple[float, float]:
    """Return (total_ram_gb, available_ram_gb) using Win32 API or psutil fallback."""
    if sys.platform == "win32":
        try:
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_gb = stat.ullTotalPhys / (1024.0**3)
                avail_gb = stat.ullAvailPhys / (1024.0**3)
                return total_gb, avail_gb
        except Exception as e:
            logger.debug(f"Win32 GlobalMemoryStatusEx failed: {e}")

    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.total / (1024.0**3), vm.available / (1024.0**3)
    except Exception:
        # Conservative default: assume 8 GB total with 4 GB available
        return 8.0, 4.0


def check_npu_available() -> bool:
    """Check if an NPU execution provider or accelerator device is available."""
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        npu_providers = {"OpenVINOExecutionProvider", "QNNExecutionProvider", "VitisAIExecutionProvider"}
        if npu_providers & set(providers):
            return True
    except Exception:
        pass

    try:
        from openvino.runtime import Core

        core = Core()
        if "NPU" in core.available_devices:
            return True
    except Exception:
        pass

    return False


def check_igpu_available() -> bool:
    """Check if an integrated GPU or DirectML provider is available."""
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            return True
    except Exception:
        pass

    try:
        from openvino.runtime import Core

        core = Core()
        if "GPU" in core.available_devices:
            return True
    except Exception:
        pass

    return False


def check_dgpu_available() -> bool:
    """Detect a discrete NVIDIA GPU.

    Prefers NVML (pynvml) because it queries the driver directly without
    spawning a process; falls back to parsing `nvidia-smi -L` when pynvml is
    not installed. Returns True if at least one NVIDIA dGPU is present.
    """
    # 1. NVML via pynvml - no subprocess, reads the driver directly.
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return pynvml.nvmlDeviceGetCount() > 0
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass

    # 2. nvidia-smi fallback.
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0 and "GPU" in out.stdout:
            return True
    except Exception:
        pass

    return False


def detect_hardware_tier(config: Optional[Any] = None) -> TierDiagnostics:
    """Detect and select the hardware tier based on available RAM and accelerators.

    Supports manual override via config.hardware_tier or HARDWARE_TIER env var.
    """
    total_gb, avail_gb = get_memory_info()
    override = None

    if config is not None:
        override = getattr(config, "hardware_tier", None)
    if not override:
        override = os.getenv("HARDWARE_TIER")

    if override:
        override_clean = override.strip().lower()
        if override_clean in ("npu", "best", "best_npu"):
            tier = HardwareTier.BEST_NPU
        elif override_clean in ("igpu", "middle", "middle_igpu", "gpu"):
            tier = HardwareTier.MIDDLE_IGPU
        else:
            tier = HardwareTier.FLOOR_CPU
        diag = TierDiagnostics(
            tier=tier,
            available_ram_gb=avail_gb,
            total_ram_gb=total_gb,
            has_npu=check_npu_available(),
            has_dgpu=check_dgpu_available(),
            has_igpu=check_igpu_available(),
            override=override,
            rationale=f"Manual configuration override: {override}",
        )
        logger.info(f"Hardware tier configured via override: {tier.value} ({diag.rationale})")
        return diag

    has_npu = check_npu_available()
    has_dgpu = check_dgpu_available()
    has_igpu = check_igpu_available()

    # Rule (PLAN §2): Available RAM bounds the tier.
    # Below 3.5 GB of free RAM, always use the CPU floor to prevent memory thrashing.
    if avail_gb < 3.5:
        tier = HardwareTier.FLOOR_CPU
        rationale = f"Low available RAM ({avail_gb:.1f} GB < 3.5 GB threshold); forced to CPU floor"
    elif has_npu:
        tier = HardwareTier.BEST_NPU
        rationale = f"NPU accelerator detected with {avail_gb:.1f} GB available RAM"
    elif has_dgpu:
        # Discrete NVIDIA GPU: CUDA compute is available, so llama can run on the dGPU
        # even when there is no iGPU. Report an honest "dgpu" tier rather than "cpu".
        tier = HardwareTier.BEST_DGPU
        rationale = f"Discrete NVIDIA GPU detected with {avail_gb:.1f} GB available RAM; CUDA compute available"
    elif has_igpu:
        tier = HardwareTier.MIDDLE_IGPU
        rationale = f"iGPU / DirectML accelerator detected with {avail_gb:.1f} GB available RAM"
    else:
        tier = HardwareTier.FLOOR_CPU
        rationale = f"No dedicated NPU/dGPU/iGPU acceleration available; {avail_gb:.1f} GB available RAM"

    diag = TierDiagnostics(
        tier=tier,
        available_ram_gb=avail_gb,
        total_ram_gb=total_gb,
        has_npu=has_npu,
        has_dgpu=has_dgpu,
        has_igpu=has_igpu,
        override=None,
        rationale=rationale,
    )
    logger.info(f"Hardware tier detected: {tier.value} [{diag.rationale}] (Total: {total_gb:.1f} GB, Avail: {avail_gb:.1f} GB)")
    return diag
