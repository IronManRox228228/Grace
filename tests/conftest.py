"""Make the test suite import the project's `grace`, not a lookalike.

An unrelated PyPI distribution named `grace` (a build helper) can be present in
site-packages. Most test modules do their own `sys.path.insert(0, ../src)`, but
a few do not, and whichever module is collected first decides which `grace`
lands in sys.modules for the whole session. Pinning the path here removes that
ordering dependency.
"""

import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)


# --- Hard block on real desktop input --------------------------------------
#
# No test may drive the real desktop. Some tests reach ComputerUse and
# Win32Driver without mocking every call, and under pytest-xdist up to 20
# workers share one desktop: one worker's Alt tap plus another's F4 or Ctrl+W
# became a real Alt+F4 / Ctrl+W on whatever window had focus, closing the
# user's own windows. Every input-injecting or focus-stealing call is stubbed
# here, at import time, so it applies in every worker before any test runs.
# A test that patches one of these itself still wins (its patch sits on top).

import subprocess as _subprocess

BLOCKED_CALLS: list = []


def _blocked(name, result):
    def stub(*args, **kwargs):
        BLOCKED_CALLS.append(name)
        return result
    stub.__name__ = f"blocked_{name}"
    return stub


if sys.platform == "win32":
    import ctypes as _ctypes

    _user32 = _ctypes.windll.user32
    for _name, _result in {
        "SendInput": 1, "keybd_event": None, "mouse_event": None,
        "SetCursorPos": 1, "SetForegroundWindow": 1, "BringWindowToTop": 1,
        "ShowWindow": 1, "SetFocus": 0, "SetActiveWindow": 0,
        "PostMessageW": 1, "PostMessageA": 1,
        "SendMessageW": 0, "SendMessageA": 0,
        "SendMessageTimeoutW": 0, "SendMessageTimeoutA": 0,
    }.items():
        setattr(_user32, _name, _blocked(f"user32.{_name}", _result))

    try:
        import win32gui as _win32gui
        for _name, _result in {
            "SetForegroundWindow": None, "BringWindowToTop": None,
            "ShowWindow": 1, "SetFocus": 0, "SetActiveWindow": 0,
            "PostMessage": None, "SendMessage": 0, "SendMessageTimeout": (0, 0),
        }.items():
            setattr(_win32gui, _name, _blocked(f"win32gui.{_name}", _result))
    except ImportError:
        pass

    try:
        import win32api as _win32api
        for _name in ("keybd_event", "mouse_event", "SetCursorPos",
                      "PostMessage", "SendMessage", "TerminateProcess"):
            setattr(_win32api, _name, _blocked(f"win32api.{_name}", None))
    except ImportError:
        pass

try:
    import pyautogui as _pyautogui
    for _name in ("press", "hotkey", "keyDown", "keyUp", "typewrite", "write",
                  "click", "doubleClick", "tripleClick", "rightClick",
                  "middleClick", "mouseDown", "mouseUp", "moveTo", "moveRel",
                  "move", "dragTo", "dragRel", "drag", "scroll", "hscroll",
                  "vscroll"):
        setattr(_pyautogui, _name, _blocked(f"pyautogui.{_name}", None))
except Exception:
    pass

# taskkill must never run for real from a test: a stale or recycled PID with
# /T kills an unrelated process tree.
_real_run, _real_popen = _subprocess.run, _subprocess.Popen


def _is_taskkill(args):
    first = args[0] if isinstance(args, (list, tuple)) and args else args
    return isinstance(first, str) and os.path.basename(first).lower().split(" ")[0] in ("taskkill", "taskkill.exe")


def _guarded_run(args, *a, **kw):
    if _is_taskkill(args):
        BLOCKED_CALLS.append("taskkill")
        return _subprocess.CompletedProcess(args, 128, "", "blocked in tests")
    return _real_run(args, *a, **kw)


class _GuardedPopen(_real_popen):
    def __init__(self, args, *a, **kw):
        if _is_taskkill(args):
            BLOCKED_CALLS.append("taskkill")
            raise OSError("taskkill is blocked in tests")
        super().__init__(args, *a, **kw)


_subprocess.run = _guarded_run
_subprocess.Popen = _GuardedPopen

# os.startfile opens real apps, files and URLs on the user's desktop.
if hasattr(os, "startfile"):
    os.startfile = _blocked("os.startfile", None)
