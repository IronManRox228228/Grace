"""Native Win32 Background Event Driver for Grace Computer Use.

Dispatches low-level mouse and keyboard events directly into application window
message queues and SendInput API with virtual screen metrics and DPI awareness.
"""

import ctypes
import logging
import time
from ctypes import wintypes
from typing import Optional

from grace.automation.dpi_helper import DPIHelper

logger = logging.getLogger("grace.automation.win32_driver")

# Win32 Constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

VK_MENU = 0x12  # ALT
KEYEVENTF_KEYUP = 0x0002

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# Delays tuned for browser focus acquisition (Bug #1: first-click sink)
FOCUS_SETTLE_MS = 150
HOVER_SETTLE_MS = 60


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", INPUT_UNION)]


class Win32Driver:
    """Dispatches background input events to Windows application message queues."""

    @classmethod
    def _ensure_foreground(cls, target_hwnd: int) -> bool:
        """Bring target window to the foreground and verify focus.

        Win32 blocks SetForegroundWindow for background processes unless the
        caller is already in the foreground. We unlock that restriction with a
        quick ALT keypress (a documented trick), then verify with
        GetForegroundWindow and retry once.

        Returns True if target_hwnd is the foreground window on exit.
        """
        try:
            import win32gui
            import win32con

            def _is_active() -> bool:
                try:
                    return win32gui.GetForegroundWindow() == target_hwnd
                except Exception:
                    return False

            if _is_active():
                return True

            try:
                win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)
            except Exception:
                pass

            for attempt in range(2):
                try:
                    # ALT-tap unlocks SetForegroundWindow for background activation
                    cls._send_alt_tap()
                    win32gui.SetForegroundWindow(target_hwnd)
                    time.sleep(FOCUS_SETTLE_MS / 1000.0)
                except Exception as e:
                    logger.debug(f"SetForegroundWindow attempt {attempt + 1} failed: {e}")

                if _is_active():
                    return True

            logger.warning(f"Win32Driver: could not verify foreground focus for hwnd={target_hwnd}")
            return _is_active()
        except Exception as e:
            logger.debug(f"Win32Driver _ensure_foreground skipped: {e}")
            return False

    @classmethod
    def _send_alt_tap(cls) -> None:
        """Send a non-destructive ALT keypress to unlock SetForegroundWindow."""
        try:
            user32 = ctypes.windll.user32
            inputs = (INPUT * 2)()
            inputs[0].type = INPUT_KEYBOARD
            inputs[0].u.ki = KEYBDINPUT(VK_MENU, 0, 0, 0, None)
            inputs[1].type = INPUT_KEYBOARD
            inputs[1].u.ki = KEYBDINPUT(VK_MENU, 0, KEYEVENTF_KEYUP, 0, None)
            user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))
        except Exception as e:
            logger.debug(f"ALT-tap unlock failed: {e}")

    @classmethod
    def click_at(cls, x: int, y: int, hwnd: Optional[int] = None) -> bool:
        """Dispatch a mouse click to physical coordinates (x, y).

        Fix Bug #1 (first-click sink): activate & verify foreground focus first.
        Fix Bug #3 (missing hover): move the hardware cursor (SetCursorPos) and
        send a non-destructive mouse-move event so :hover / onmouseenter fires
        before the down/up. Browsers consume a click that lacks a prior hover.
        """
        DPIHelper.ensure_dpi_aware()

        target_hwnd = None
        try:
            import win32gui

            target_hwnd = hwnd or win32gui.WindowFromPoint((int(x), int(y)))
            if target_hwnd and win32gui.IsWindowVisible(target_hwnd):
                cls._ensure_foreground(target_hwnd)
        except Exception as e:
            # pywin32 optional; focus activation is best-effort
            logger.debug(f"Win32Driver focus step skipped: {e}")

        # Position the hardware cursor first so the OS tracks a real hover state
        try:
            ctypes.windll.user32.SetCursorPos(int(x), int(y))
            time.sleep(HOVER_SETTLE_MS / 1000.0)
        except Exception as e:
            logger.debug(f"SetCursorPos failed: {e}")

        cls.send_input_click(x, y)
        return True

    @classmethod
    def send_input_click(cls, x: int, y: int):
        """Send absolute mouse click event via native Windows SendInput API with virtual screen metrics.

        A non-destructive MOVE event is dispatched before DOWN/UP so that web
        elements relying on hover/mouseenter receive the full interaction cycle.
        """
        DPIHelper.ensure_dpi_aware()
        try:
            user32 = ctypes.windll.user32

            # FIX BUG #11: Query Virtual Screen metrics for multi-monitor / DPI support
            vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
            vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
            vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
            vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

            if vw <= 0 or vh <= 0:
                vx, vy, vw, vh = 0, 0, 1920, 1080

            # Normalize coordinates to 0..65535 absolute range across virtual desk
            norm_x = int((x - vx) * 65535 / vw)
            norm_y = int((y - vy) * 65535 / vh)

            flags_move = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK | MOUSEEVENTF_MOVE
            flags_down = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK | MOUSEEVENTF_LEFTDOWN
            flags_up = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK | MOUSEEVENTF_LEFTUP

            # FIX BUG #12: Use actual SendInput API call
            # MOVE first (hover), then DOWN, then UP.
            inputs = (INPUT * 3)()

            inputs[0].type = INPUT_MOUSE
            inputs[0].u.mi = MOUSEINPUT(norm_x, norm_y, 0, flags_move, 0, None)

            inputs[1].type = INPUT_MOUSE
            inputs[1].u.mi = MOUSEINPUT(norm_x, norm_y, 0, flags_down, 0, None)

            inputs[2].type = INPUT_MOUSE
            inputs[2].u.mi = MOUSEINPUT(norm_x, norm_y, 0, flags_up, 0, None)

            sent = user32.SendInput(3, ctypes.byref(inputs), ctypes.sizeof(INPUT))
            if sent != 3:
                # Fallback to mouse_event if SendInput failed
                user32.mouse_event(flags_move, norm_x, norm_y, 0, 0)
                user32.mouse_event(flags_down, norm_x, norm_y, 0, 0)
                user32.mouse_event(flags_up, norm_x, norm_y, 0, 0)

            logger.info(f"Win32Driver: SendInput mouse click executed at absolute ({x}, {y})")

        except Exception as e:
            logger.debug(f"SendInput click error: {e}")
