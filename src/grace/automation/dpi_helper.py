"""DPI Scaling Helper for Windows Automation in Grace.

Sets per-monitor DPI awareness and provides coordinate/resolution helpers so
that UIA logical coordinates, screenshot physical pixels, and SendInput virtual
screen coordinates all map 1:1 under Windows display scaling (125/150/200%).
"""

import ctypes
import logging
from typing import Optional, Tuple

logger = logging.getLogger("grace.automation.dpi_helper")

_dpi_aware_set = False

# Win32 metric / DPI constants
SM_CXSCREEN = 0
SM_CYSCREEN = 1
LOGPIXELSX = 88
USER_DEFAULT_SCREEN_DPI = 96


class DPIHelper:
    """Helper to set per-monitor DPI awareness for accurate Win32/PyAutoGUI coordinates."""

    @classmethod
    def ensure_dpi_aware(cls) -> bool:
        global _dpi_aware_set
        if _dpi_aware_set:
            return True

        try:
            # PROCESS_PER_MONITOR_DPI_AWARE_V2 = 2
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            _dpi_aware_set = True
            logger.info("DPI awareness set to Process_Per_Monitor_DPI_Aware_V2")
            return True
        except Exception:
            try:
                # Fallback to system DPI awareness
                ctypes.windll.user32.SetProcessDPIAware()
                _dpi_aware_set = True
                logger.info("DPI awareness set to System_DPI_Aware")
                return True
            except Exception as e:
                logger.warning(f"Failed to set DPI awareness: {e}")
                return False

    @classmethod
    def get_screen_resolution(cls) -> Tuple[int, int]:
        """Return primary screen resolution in physical pixels as (width, height).

        Returns physical (unscaled) dimensions because DPI awareness is requested
        at the top of every public method that needs it.
        """
        cls.ensure_dpi_aware()
        try:
            user32 = ctypes.windll.user32
            w = user32.GetSystemMetrics(SM_CXSCREEN)
            h = user32.GetSystemMetrics(SM_CYSCREEN)
            if w > 0 and h > 0:
                return w, h
        except Exception as e:
            logger.debug(f"get_screen_resolution failed: {e}")
        # Conservative fallback
        return 1920, 1080

    @classmethod
    def scale_coords(
        cls,
        x: int,
        y: int,
        src_w: int,
        src_h: int,
        dst_w: int,
        dst_h: int,
    ) -> Tuple[int, int]:
        """Scale a point from one coordinate space to another.

        Args:
            x, y: Source coordinates.
            src_w, src_h: Source space dimensions.
            dst_w, dst_h: Destination space dimensions.

        Returns:
            (x', y') scaled to the destination space. If a source dimension is
            non-positive the coordinate is returned unchanged to avoid div-by-zero.
        """
        if src_w <= 0 or src_h <= 0:
            return x, y
        try:
            return int(x * dst_w / src_w), int(y * dst_h / src_h)
        except Exception as e:
            logger.debug(f"scale_coords failed: {e}")
            return x, y

    @classmethod
    def get_dpi_scale(cls, hwnd: Optional[int] = None) -> float:
        """Return the per-monitor DPI scale factor (1.0, 1.25, 1.5, 2.0, ...).

        Args:
            hwnd: Optional window handle to query per-monitor DPI. If omitted,
                  the foreground window (or primary monitor) is used.

        Returns:
            Scale factor such that physical_px = logical_px * scale.
        """
        cls.ensure_dpi_aware()
        try:
            user32 = ctypes.windll.user32
            if hwnd is None:
                try:
                    hwnd = user32.GetForegroundWindow()
                except Exception:
                    hwnd = None

            # GetDpiForWindow is available on Windows 10 1607+
            if hwnd:
                try:
                    dpi = user32.GetDpiForWindow(int(hwnd))
                    if dpi and dpi > 0:
                        return round(dpi / USER_DEFAULT_SCREEN_DPI, 2)
                except Exception:
                    pass

            # Fallback: system DPI via GDI GetDeviceCaps(LOGPIXELSX)
            try:
                hdc = user32.GetDC(0)
                dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
                user32.ReleaseDC(0, hdc)
                if dpi and dpi > 0:
                    return round(dpi / USER_DEFAULT_SCREEN_DPI, 2)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"get_dpi_scale failed: {e}")

        return 1.0
