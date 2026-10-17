"""Tests proving the real desktop guard in tests/conftest.py works as intended.

Verifies:
1. An unpatched ctypes.windll.user32.SendInput call fails loudly with AssertionError.
2. A taskkill call through subprocess.run is blocked without spawning a process.
"""

import sys
import subprocess
import pytest


def test_guard_blocks_real_input_and_taskkill():
    if sys.platform == "win32":
        import ctypes
        with pytest.raises(AssertionError, match="real desktop call user32.SendInput blocked in tests"):
            ctypes.windll.user32.SendInput(0, None, 0)

    res = subprocess.run(["taskkill", "/F", "/PID", "99999"], capture_output=True, text=True)
    assert res.returncode == 128
    assert "blocked in tests" in res.stderr
