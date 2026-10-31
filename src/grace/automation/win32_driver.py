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

# Delays tuned for browser focus acquisition (Bug #1: first-click sink).
# HOVER_SETTLE_FAST_MS applies when the target window was already foreground,
# which is the common case for a multi-step agent acting inside one window.
FOCUS_SETTLE_MS = 150
HOVER_SETTLE_MS = 60
HOVER_SETTLE_FAST_MS = 10

# How long the left button is held before the first move and again before the
# release. Paint (and other win32 apps) can drop a drag that starts moving the
# instant the button goes down, so the cursor rests on the start point first.
DRAG_HOLD_PAUSE_MS = 50

# A drag segment is cut into steps no wider than this, so a canvas stroke is
# continuous rather than one straight jump. 10 px at a time.
DRAG_STEP_PX = 10

# About 5-10 ms per step, as close to human input as the loop can get.
DRAG_STEP_DELAY_S = 0.006


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
    def _is_foreground(cls, target_hwnd: int) -> bool:
        """True if target_hwnd (or its root ancestor) is already the foreground window."""
        try:
            import win32gui

            fg = win32gui.GetForegroundWindow()
            if fg == target_hwnd:
                return True
            try:
                root = win32gui.GetAncestor(target_hwnd, 2)  # GA_ROOT
                if root and (fg == root or win32gui.GetAncestor(fg, 2) == root):
                    return True
            except Exception:
                pass
            return False
        except Exception:
            return False

    @classmethod
    def _ensure_foreground(cls, target_hwnd: int) -> bool:
        """Bring target window to the foreground and verify focus.

        Win32 blocks SetForegroundWindow for background processes unless the
        caller is already in the foreground. Uses documented safe techniques:
        - Resolving root ancestor if target_hwnd is a child control (e.g. Chrome/Edge render widget)
        - AttachThreadInput to the foreground thread to share input queue rights
        - AllowSetForegroundWindow(-1)
        - Synthetic ALT-tap to unlock foreground lock
        - ShowWindow(SW_RESTORE) and BringWindowToTop

        Returns True if target_hwnd is the foreground window on exit.
        """
        try:
            import win32gui
            import win32con

            # Resolve root ancestor if target is a child window
            try:
                root = win32gui.GetAncestor(target_hwnd, 2)  # GA_ROOT
                if isinstance(root, int) and root > 0:
                    target_hwnd = root
            except Exception:
                pass

            def _is_active(fg=None) -> bool:
                try:
                    if fg is None:
                        fg = win32gui.GetForegroundWindow()
                    if fg == target_hwnd:
                        return True
                    root_fg = win32gui.GetAncestor(fg, 2)
                    if isinstance(root_fg, int) and root_fg > 0 and root_fg == target_hwnd:
                        return True
                    return False
                except Exception:
                    return False

            fg_hwnd = win32gui.GetForegroundWindow()
            if _is_active(fg_hwnd):
                return True

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            cur_thread = kernel32.GetCurrentThreadId() if hasattr(kernel32, "GetCurrentThreadId") else 0
            fg_thread = user32.GetWindowThreadProcessId(fg_hwnd, None) if (fg_hwnd and hasattr(user32, "GetWindowThreadProcessId")) else 0

            attached = False
            if fg_thread and cur_thread and fg_thread != cur_thread:
                try:
                    attached = bool(user32.AttachThreadInput(cur_thread, fg_thread, True))
                except Exception as e:
                    logger.debug(f"AttachThreadInput attach failed: {e}")

            try:
                try:
                    user32.AllowSetForegroundWindow(-1)  # ASFW_ANY
                except Exception:
                    pass

                try:
                    win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)
                except Exception:
                    pass

                try:
                    win32gui.BringWindowToTop(target_hwnd)
                except Exception:
                    pass

                for attempt in range(2):
                    try:
                        cls._send_alt_tap()
                        win32gui.SetForegroundWindow(target_hwnd)
                        time.sleep(FOCUS_SETTLE_MS / 1000.0)
                    except Exception as e:
                        logger.debug(f"SetForegroundWindow attempt {attempt + 1} failed: {e}")

                    if _is_active():
                        return True
            finally:
                if attached:
                    try:
                        user32.AttachThreadInput(cur_thread, fg_thread, False)
                    except Exception as e:
                        logger.debug(f"AttachThreadInput detach failed: {e}")

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
    def send_virtual_key(cls, vk: int) -> bool:
        """Send a virtual key press and release via SendInput."""
        try:
            user32 = ctypes.windll.user32
            inputs = (INPUT * 2)()
            inputs[0].type = INPUT_KEYBOARD
            inputs[0].u.ki = KEYBDINPUT(vk, 0, 0, 0, None)
            inputs[1].type = INPUT_KEYBOARD
            inputs[1].u.ki = KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP, 0, None)
            return user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT)) == 2
        except Exception as e:
            logger.debug(f"SendInput virtual key {hex(vk)} failed: {e}")
            return False

    @classmethod
    def click_at(cls, x: int, y: int, hwnd: Optional[int] = None, clicks: int = 1) -> bool:
        """Dispatch a mouse click to physical coordinates (x, y).

        `clicks` > 1 sends repeated down/up pairs inside the double-click time,
        so double-click requests are honoured here rather than only on the
        pyautogui fallback path.

        Fix Bug #1 (first-click sink): activate & verify foreground focus first.
        Fix Bug #3 (missing hover): move the hardware cursor (SetCursorPos) and
        send a non-destructive mouse-move event so :hover / onmouseenter fires
        before the down/up. Browsers consume a click that lacks a prior hover.
        """
        DPIHelper.ensure_dpi_aware()

        target_hwnd = None
        already_foreground = False
        try:
            import win32gui

            target_hwnd = hwnd or win32gui.WindowFromPoint((int(x), int(y)))
            if target_hwnd and win32gui.IsWindowVisible(target_hwnd):
                # _ensure_foreground returns immediately (and sleeps nothing)
                # when the window is already active; remember that so the hover
                # settle below can be shortened too.
                already_foreground = cls._is_foreground(target_hwnd)
                if not already_foreground:
                    if not cls._ensure_foreground(target_hwnd):
                        logger.warning(f"Win32Driver: failed to focus target window {target_hwnd}")
                        return False
        except Exception as e:
            # pywin32 optional; focus activation is best-effort
            logger.debug(f"Win32Driver focus step skipped: {e}")

        # Position the hardware cursor first so the OS tracks a real hover state.
        # Check Gameface coexistence: do not click while user is mid-gesture.
        try:
            from .gameface import GamefaceCoexistence
            GamefaceCoexistence().wait_for_quiet_window(timeout_seconds=0.3)
        except Exception:
            pass

        try:
            ctypes.windll.user32.SetCursorPos(int(x), int(y))

            settle_ms = HOVER_SETTLE_FAST_MS if already_foreground else HOVER_SETTLE_MS
            time.sleep(settle_ms / 1000.0)
        except Exception as e:
            logger.debug(f"SetCursorPos failed: {e}")

        for i in range(max(1, int(clicks or 1))):
            if i:
                # Comfortably inside the default 500ms double-click interval.
                time.sleep(0.05)
            if not cls.send_input_click(x, y):
                return False
        return True

    @classmethod
    def send_input_click(cls, x: int, y: int) -> bool:
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
            norm_x = int((x - vx) * 65535 / max(vw - 1, 1))
            norm_y = int((y - vy) * 65535 / max(vh - 1, 1))

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
            return True

        except Exception as e:
            logger.warning(f"SendInput click error: {e}")
            return False

    @classmethod
    def _send_mouse(cls, x: int, y: int, button_flag: int) -> bool:
        """One absolute mouse event (move, left down or left up) via SendInput."""
        DPIHelper.ensure_dpi_aware()
        try:
            user32 = ctypes.windll.user32
            vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
            vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
            vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
            vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
            if vw <= 0 or vh <= 0:
                vx, vy, vw, vh = 0, 0, 1920, 1080
            norm_x = int((x - vx) * 65535 / max(vw - 1, 1))
            norm_y = int((y - vy) * 65535 / max(vh - 1, 1))
            flags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK | button_flag
            inputs = (INPUT * 1)()
            inputs[0].type = INPUT_MOUSE
            inputs[0].u.mi = MOUSEINPUT(norm_x, norm_y, 0, flags, 0, None)
            return user32.SendInput(1, ctypes.byref(inputs), ctypes.sizeof(INPUT)) == 1
        except Exception as e:
            logger.warning(f"SendInput mouse event error: {e}")
            return False

    @classmethod
    def mouse_move(cls, x: int, y: int) -> bool:
        return cls._send_mouse(int(x), int(y), MOUSEEVENTF_MOVE)

    @classmethod
    def mouse_down(cls, x: int, y: int) -> bool:
        return cls._send_mouse(int(x), int(y), MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTDOWN)

    @classmethod
    def mouse_up(cls, x: int, y: int) -> bool:
        return cls._send_mouse(int(x), int(y), MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTUP)

    @classmethod
    def drag_path(cls, points: list[tuple[int, int]], step_px: int = DRAG_STEP_PX, step_delay: float = DRAG_STEP_DELAY_S) -> bool:
        """Press at the first point, move through the rest in small steps, release at the last.

        A real human-style drag, all via SendInput absolute moves:

        - move the cursor to the start point
        - left button down
        - pause (~50 ms) so the app registers the button-down on the start point
        - move through the path in steps of at most `step_px` pixels
          (about `step_delay` each) with the button held
        - pause (~50 ms)
        - left button up
        """
        if len(points) < 2:
            return False
        first = (int(points[0][0]), int(points[0][1]))
        try:
            import win32gui

            target_hwnd = win32gui.WindowFromPoint(first)
            if target_hwnd and win32gui.IsWindowVisible(target_hwnd) and not cls._is_foreground(target_hwnd):
                if not cls._ensure_foreground(target_hwnd):
                    logger.warning(f"Win32Driver: failed to focus target window {target_hwnd}")
                    return False
        except Exception as e:
            logger.debug(f"Win32Driver focus step skipped: {e}")

        try:
            ctypes.windll.user32.SetCursorPos(*first)
            time.sleep(HOVER_SETTLE_FAST_MS / 1000.0)
        except Exception as e:
            logger.debug(f"SetCursorPos failed: {e}")

        if not cls.mouse_down(*first):
            return False
        time.sleep(DRAG_HOLD_PAUSE_MS / 1000.0)
        last = first
        try:
            cls.mouse_move(*first)
            for px, py in points[1:]:
                px, py = int(px), int(py)
                dist = max(abs(px - last[0]), abs(py - last[1]))
                steps = max(1, -(-dist // max(1, step_px)))
                for i in range(1, steps + 1):
                    cls.mouse_move(
                        last[0] + (px - last[0]) * i // steps,
                        last[1] + (py - last[1]) * i // steps,
                    )
                    time.sleep(step_delay)
                last = (px, py)
            time.sleep(DRAG_HOLD_PAUSE_MS / 1000.0)
        finally:
            cls.mouse_up(*last)
        return True
