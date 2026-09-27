"""Test local ComputerUse backend."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.automation import computer_use as computer_use_module
from grace.automation.computer_use import ComputerUse


class _StubAppIndexer:
    """Stands in for `AppIndexer`: no real filesystem scan, one canned result."""

    def __init__(self, result=None):
        self.result = result or {"status": "error", "error": "not found"}
        self.calls: list[str] = []

    def launch(self, name):
        self.calls.append(name)
        return self.result


class TestLaunchHasNoShellFallback:
    """`_launch` used to fall back to `Popen(f'start "" "{target}"', shell=True)`.

    `target` can be model-generated text, so that fallback let a click on the
    wrong element run an arbitrary second shell command. Now a failed launch
    (via the app indexer's own validated resolution) is just reported as a
    failure.
    """

    def test_a_failed_launch_never_touches_a_shell(self, monkeypatch):
        popen_calls = []
        monkeypatch.setattr(
            computer_use_module.subprocess, "Popen",
            lambda *a, **k: popen_calls.append((a, k)),
        )

        cu = ComputerUse()
        cu.set_app_indexer(_StubAppIndexer({"status": "error", "error": "not found"}))
        res = cu._launch({"app": "nonexistent_app_xyz"})

        assert res["ok"] is False
        assert "not found" in res["message"]
        assert popen_calls == [], "a failed launch must never fall back to a shell"


class TestLaunchValidatesTheAppName:
    """R3: `cua_launch` used to hand raw model text straight to `os.startfile`
    with no path/protocol/character restriction at all - a UNC path leaked
    the Windows user's NTLM handshake, and anything else just ran.
    """

    def test_refuses_a_unc_path(self):
        cu = ComputerUse()
        cu.set_app_indexer(_StubAppIndexer())
        res = cu._launch({"app": r"\\attacker\share\x"})
        assert res["ok"] is False
        assert cu._app_indexer.calls == [], "a UNC path must never reach the app indexer"

    def test_refuses_a_forward_slash_unc_path(self):
        cu = ComputerUse()
        cu.set_app_indexer(_StubAppIndexer())
        res = cu._launch({"app": "//attacker/share/x"})
        assert res["ok"] is False
        assert cu._app_indexer.calls == []

    def test_refuses_a_path_with_separators(self):
        cu = ComputerUse()
        cu.set_app_indexer(_StubAppIndexer())
        res = cu._launch({"app": r"C:\Windows\System32\cmd.exe"})
        assert res["ok"] is False
        assert cu._app_indexer.calls == []

    def test_refuses_a_non_http_uri_scheme(self):
        cu = ComputerUse()
        cu.set_app_indexer(_StubAppIndexer())
        res = cu._launch({"app": "file:///C:/Windows/System32/cmd.exe"})
        assert res["ok"] is False
        assert cu._app_indexer.calls == []

    def test_refuses_a_custom_registered_scheme(self):
        cu = ComputerUse()
        cu.set_app_indexer(_StubAppIndexer())
        res = cu._launch({"app": "myapp://do-something-bad"})
        assert res["ok"] is False
        assert cu._app_indexer.calls == []

    def test_an_ordinary_app_name_reaches_the_indexer(self):
        stub = _StubAppIndexer({"status": "ok", "text": "I've opened Notepad."})
        cu = ComputerUse()
        cu.set_app_indexer(stub)
        res = cu._launch({"app": "notepad"})
        assert res["ok"] is True
        assert stub.calls == ["notepad"]

    def test_an_http_url_is_allowed_through(self):
        stub = _StubAppIndexer({"status": "ok", "text": "opened"})
        cu = ComputerUse()
        cu.set_app_indexer(stub)
        res = cu._launch({"app": "https://example.com"})
        assert res["ok"] is True
        assert stub.calls == ["https://example.com"]


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
