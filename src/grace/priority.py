"""Process priority and EcoQoS control for Grace (PLAN.md §4).

Adjusts process execution priority dynamically without requiring administrative privileges:
- Above-normal priority during an active turn: ensures real-time speech processing,
  wake-word responsiveness, and low-latency desktop actuation.
- Normal or below-normal priority with Windows EcoQoS when idle: minimizes background
  CPU power draw and thermal throttling while awaiting activation.
- Below-normal with EcoQoS for background indexing or opportunistic model training.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import sys
from typing import Optional

logger = logging.getLogger("grace.priority")

# Win32 Priority Classes
IDLE_PRIORITY_CLASS = 0x00000040
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
NORMAL_PRIORITY_CLASS = 0x00000020
ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
HIGH_PRIORITY_CLASS = 0x00000080

# Win32 ProcessPowerThrottling
PROCESS_POWER_THROTTLING = 4
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1


class _PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
    _fields_ = [
        ("Version", wintypes.DWORD),
        ("ControlMask", wintypes.DWORD),
        ("StateMask", wintypes.DWORD),
    ]


_kernel32 = None
if sys.platform == "win32":
    try:
        _kernel32 = ctypes.windll.kernel32
        _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        _kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        _kernel32.SetPriorityClass.restype = wintypes.BOOL
        _kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        _kernel32.GetPriorityClass.restype = wintypes.DWORD
        if hasattr(_kernel32, "SetProcessInformation"):
            _kernel32.SetProcessInformation.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
            ]
            _kernel32.SetProcessInformation.restype = wintypes.BOOL
    except Exception as e:
        logger.debug(f"Failed to bind Win32 priority APIs: {e}")
        _kernel32 = None


def set_priority_class(priority_class: int) -> bool:
    """Set the current process priority class."""
    if _kernel32 is None:
        return False
    try:
        proc = _kernel32.GetCurrentProcess()
        return bool(_kernel32.SetPriorityClass(proc, priority_class))
    except Exception as e:
        logger.debug(f"SetPriorityClass({hex(priority_class)}) failed: {e}")
        return False


def set_eco_qos(enable: bool) -> bool:
    """Enable or disable Windows EcoQoS (execution speed power throttling)."""
    if _kernel32 is None or not hasattr(_kernel32, "SetProcessInformation"):
        return False
    try:
        proc = _kernel32.GetCurrentProcess()
        state = _PROCESS_POWER_THROTTLING_STATE()
        state.Version = 1
        state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        state.StateMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED if enable else 0
        return bool(
            _kernel32.SetProcessInformation(
                proc,
                PROCESS_POWER_THROTTLING,
                ctypes.byref(state),
                ctypes.sizeof(state),
            )
        )
    except Exception as e:
        logger.debug(f"SetProcessInformation(EcoQoS={enable}) failed: {e}")
        return False


def set_turn_priority() -> None:
    """Elevate priority to above-normal during active turn; disable EcoQoS."""
    set_priority_class(ABOVE_NORMAL_PRIORITY_CLASS)
    set_eco_qos(enable=False)
    logger.debug("Process priority elevated: ABOVE_NORMAL, EcoQoS=OFF")


def set_idle_priority() -> None:
    """Restore normal priority and enable EcoQoS when waiting for wake word."""
    set_priority_class(NORMAL_PRIORITY_CLASS)
    set_eco_qos(enable=True)
    logger.debug("Process priority restored: NORMAL, EcoQoS=ON")


def set_background_task_priority() -> None:
    """Lower priority to below-normal with EcoQoS for opportunistic background work."""
    set_priority_class(BELOW_NORMAL_PRIORITY_CLASS)
    set_eco_qos(enable=True)
    logger.debug("Process priority throttled: BELOW_NORMAL, EcoQoS=ON")


class turn_priority:
    """Context manager for elevating priority during an active turn."""

    def __enter__(self):
        set_turn_priority()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_idle_priority()
