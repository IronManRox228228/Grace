"""Google Project Gameface coexistence guard (PLAN.md §7, Task 10).

Detects if Google Project Gameface (open-source hands-free facial gesture / head
tracking mouse on Windows) is active, and ensures Grace never injects automated
inputs while Gameface is mid-gesture.

Rule: No Project Gameface code is vendored. Coexistence is achieved entirely
through process observation, OS cursor stability checks, and cooperative quiet windows.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional, Tuple

import psutil

logger = logging.getLogger("grace.automation.gameface")

_GAMEFACE_PROCESS_NAMES = {
    "gameface.exe",
    "project-gameface.exe",
    "gameface-desktop.exe",
    "gameface_service.exe",
    "project_gameface.exe",
}


def _get_cursor_pos() -> Optional[Tuple[int, int]]:
    """Return current physical cursor position via Win32 API."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        pt = wintypes.POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return pt.x, pt.y
    except Exception as e:
        logger.debug("Failed to query cursor position: %e", e)
    return None


class GamefaceCoexistence:
    """Detects Project Gameface and guards automated input injection."""

    def __init__(self, cache_ttl_seconds: float = 5.0) -> None:
        self.cache_ttl_seconds = cache_ttl_seconds
        self._last_check_time: float = 0.0
        self._is_running_cached: bool = False

    def is_running(self, force_refresh: bool = False) -> bool:
        """Check if Project Gameface process or window is currently active."""
        now = time.perf_counter()
        if not force_refresh and (now - self._last_check_time) < self.cache_ttl_seconds:
            return self._is_running_cached

        self._last_check_time = now
        self._is_running_cached = self._detect_process()
        return self._is_running_cached

    def _detect_process(self) -> bool:
        """Scan running processes for Project Gameface binaries."""
        try:
            for proc in psutil.process_iter(["name"]):
                name = proc.info.get("name")
                if name and name.lower() in _GAMEFACE_PROCESS_NAMES:
                    logger.debug("Detected running Project Gameface process: %s", name)
                    return True
        except Exception as e:
            logger.debug("Process iteration failed during Gameface detection: %s", e)
        return False

    def wait_for_quiet_window(
        self,
        timeout_seconds: float = 0.5,
        quiet_window_ms: float = 120.0,
    ) -> bool:
        """Wait for head/gesture motion to settle before automated input injection.

        Returns True if the system is quiet (safe to click/type), False if timed out.
        """
        if not self.is_running():
            return True

        quiet_window_sec = quiet_window_ms / 1000.0
        deadline = time.perf_counter() + timeout_seconds

        last_pos = _get_cursor_pos()
        last_move_time = time.perf_counter()

        while time.perf_counter() < deadline:
            time.sleep(0.02)
            cur_pos = _get_cursor_pos()
            now = time.perf_counter()

            if cur_pos != last_pos:
                last_pos = cur_pos
                last_move_time = now
            else:
                if (now - last_move_time) >= quiet_window_sec:
                    return True

        logger.warning(
            "Gameface motion did not settle within %.1fs timeout; proceeding with caution",
            timeout_seconds,
        )
        return False
