"""Test local ComputerUse backend."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.automation import computer_use as computer_use_module
from grace.automation.computer_use import ComputerUse


class TestLaunchHasNoShellFallback:
    """`_launch` used to fall back to `Popen(f'start "" "{target}"', shell=True)`.

    `target` can be model-generated text, so that fallback let a click on the
    wrong element run an arbitrary second shell command. Now a failed
    `os.startfile` is just reported as a failure.
    """

    def test_a_failed_launch_never_touches_a_shell(self, monkeypatch):
        popen_calls = []
        monkeypatch.setattr(
            computer_use_module.subprocess, "Popen",
            lambda *a, **k: popen_calls.append((a, k)),
        )
        monkeypatch.setattr(
            computer_use_module.os, "startfile",
            lambda *a, **k: (_ for _ in ()).throw(OSError("not found")),
        )

        cu = ComputerUse()
        res = cu._launch({"app": "nonexistent_app_xyz"})

        assert res["ok"] is False
        assert "not found" in res["message"]
        assert popen_calls == [], "a failed launch must never fall back to a shell"


class TestComputerUse:
    """Test suite for ComputerUse."""

    def test_init(self):
        cu = ComputerUse()
        assert not cu.is_ready

    def test_start_stop(self):
        cu = ComputerUse()
        cu.start()
        assert cu.is_ready
        cu.stop()
        assert not cu.is_ready

    def test_perform_screenshot(self):
        cu = ComputerUse()
        cu.start()
        result = cu.perform("screenshot", {})
        assert result.get("ok") is True
        assert "png_b64" in result
        assert result.get("width", 0) > 0
        cu.stop()
