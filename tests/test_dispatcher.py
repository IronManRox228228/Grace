"""Test tool dispatcher."""

import sys
import os
import asyncio

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.intent.parser import Intent
from grace.tools import dispatcher as dispatcher_module
from grace.tools.dispatcher import Dispatcher


class TestDispatcher:
    """Test suite for Dispatcher."""

    def setup_method(self):
        self.dispatcher = Dispatcher(computer_use=None)

    def test_execute_unknown_tool(self):
        """Execute with an unknown tool returns error."""
        intent = Intent(tool="nonexistent_tool", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "error"
        assert "nonexistent_tool" in result["error"]

    def test_execute_converse(self):
        """converse has a handler and speaks the response back.

        This asserted `"error" in result or "status" in result`, which every
        possible result satisfies - both branches of the dispatcher always
        return one or the other - so it passed whatever converse did. Its
        comment claimed converse "is not in handler_map" and "falls through to
        unknown tool handler", contradicting the handler map, which has listed
        it for as long as the file has existed.
        """
        intent = Intent(tool="converse", params={"response": "Hi!"})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "ok"
        assert result["text"] == "Hi!"

    def test_cua_tool_without_bridge(self):
        """CUA tool without bridge returns error."""
        intent = Intent(tool="cua_click", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "error"
        assert "not available" in result["error"].lower()


@pytest.fixture
def safe_home(tmp_path, monkeypatch):
    """A fake USERPROFILE with empty Documents/Desktop/Downloads directories."""
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for name in ("Documents", "Desktop", "Downloads"):
        (tmp_path / name).mkdir()
    return tmp_path


class TestOpenFileSafety:
    """R2: `open_file` used to fall back to `os.startfile(name)` on a raw,
    unvalidated model-supplied string, with no location or extension check at
    all before the call. `os.startfile` is mocked in every test here - nothing
    may launch anything for real.
    """

    def test_refuses_a_unc_path(self, safe_home, monkeypatch):
        calls = []
        monkeypatch.setattr(dispatcher_module.os, "startfile", lambda p: calls.append(p))
        dispatcher = Dispatcher(computer_use=None)
        intent = Intent(tool="open_file", params={"name": r"\\attacker\share\evil.txt"})
        result = asyncio.run(dispatcher.execute(intent))
        assert result["status"] == "error"
        assert calls == []

    def test_refuses_a_path_instead_of_a_bare_name(self, safe_home, monkeypatch):
        calls = []
        monkeypatch.setattr(dispatcher_module.os, "startfile", lambda p: calls.append(p))
        dispatcher = Dispatcher(computer_use=None)
        # A harmless extension, so this isolates the path-separator check
        # from the (separately tested) dangerous-extension confirmation gate.
        intent = Intent(tool="open_file", params={"name": r"C:\Users\Public\notes.txt"})
        result = asyncio.run(dispatcher.execute(intent))
        assert result["status"] == "error"
        assert calls == []

    def test_refuses_a_name_that_does_not_resolve_anywhere_safe(self, safe_home, monkeypatch):
        calls = []
        monkeypatch.setattr(dispatcher_module.os, "startfile", lambda p: calls.append(p))
        dispatcher = Dispatcher(computer_use=None)
        intent = Intent(tool="open_file", params={"name": "does-not-exist.txt"})
        result = asyncio.run(dispatcher.execute(intent))
        assert result["status"] == "error"
        assert calls == [], "an unresolved name must never reach os.startfile"

    def test_an_ordinary_file_in_documents_opens(self, safe_home, monkeypatch):
        (safe_home / "Documents" / "budget.pdf").write_text("x")
        calls = []
        monkeypatch.setattr(dispatcher_module.os, "startfile", lambda p: calls.append(p))
        dispatcher = Dispatcher(computer_use=None)
        intent = Intent(tool="open_file", params={"name": "budget.pdf"})
        result = asyncio.run(dispatcher.execute(intent))
        assert result["status"] == "ok"
        assert calls == [str(safe_home / "Documents" / "budget.pdf")]

    def test_an_exact_match_wins_over_a_substring_match(self, safe_home, monkeypatch):
        docs = safe_home / "Documents"
        (docs / "budget.pdf").write_text("x")
        (docs / "old budget draft.pdf").write_text("x")
        calls = []
        monkeypatch.setattr(dispatcher_module.os, "startfile", lambda p: calls.append(p))
        dispatcher = Dispatcher(computer_use=None)
        intent = Intent(tool="open_file", params={"name": "budget.pdf"})
        result = asyncio.run(dispatcher.execute(intent))
        assert result["status"] == "ok"
        assert calls == [str(docs / "budget.pdf")]

    def test_glob_metacharacters_in_the_name_are_escaped(self, safe_home, monkeypatch):
        docs = safe_home / "Documents"
        (docs / "report[final].pdf").write_text("x")
        calls = []
        monkeypatch.setattr(dispatcher_module.os, "startfile", lambda p: calls.append(p))
        dispatcher = Dispatcher(computer_use=None)
        intent = Intent(tool="open_file", params={"name": "report[final].pdf"})
        result = asyncio.run(dispatcher.execute(intent))
        assert result["status"] == "ok"
        assert calls == [str(docs / "report[final].pdf")]

    def test_an_executable_extension_requires_confirmation(self, safe_home, monkeypatch):
        (safe_home / "Downloads" / "installer.exe").write_text("x")
        calls = []
        monkeypatch.setattr(dispatcher_module.os, "startfile", lambda p: calls.append(p))
        dispatcher = Dispatcher(computer_use=None)
        intent = Intent(tool="open_file", params={"name": "installer.exe"})

        result = asyncio.run(dispatcher.execute(intent))
        assert result["status"] == "confirmation_required"
        assert calls == []

        confirmed = asyncio.run(dispatcher.execute(intent, confirmed=True))
        assert confirmed["status"] == "ok"
        assert calls == [str(safe_home / "Downloads" / "installer.exe")]

    def test_a_lnk_shortcut_also_requires_confirmation(self, safe_home, monkeypatch):
        (safe_home / "Desktop" / "shortcut.lnk").write_text("x")
        monkeypatch.setattr(dispatcher_module.os, "startfile", lambda p: None)
        dispatcher = Dispatcher(computer_use=None)
        intent = Intent(tool="open_file", params={"name": "shortcut.lnk"})
        result = asyncio.run(dispatcher.execute(intent))
        assert result["status"] == "confirmation_required"


class TestOpenAppUrlSafety:
    """R15's Python analogue: `open_app`'s explicit `url` param used to reach
    `webbrowser.open` with no scheme validation at all. `webbrowser.open` is
    mocked in every test here - nothing may launch a browser for real.
    """

    def test_an_http_url_opens(self, monkeypatch):
        import webbrowser
        calls = []
        monkeypatch.setattr(webbrowser, "open", lambda u: calls.append(u))
        dispatcher = Dispatcher(computer_use=None)
        intent = Intent(tool="open_app", params={"url": "https://example.com"})
        result = asyncio.run(dispatcher.execute(intent))
        assert result["status"] == "ok"
        assert calls == ["https://example.com"]

    def test_refuses_a_file_url(self, monkeypatch):
        import webbrowser
        calls = []
        monkeypatch.setattr(webbrowser, "open", lambda u: calls.append(u))
        dispatcher = Dispatcher(computer_use=None)
        intent = Intent(tool="open_app", params={"url": "file:///C:/Windows/System32/cmd.exe"})
        result = asyncio.run(dispatcher.execute(intent))
        assert result["status"] == "error"
        assert calls == []

    def test_refuses_a_unc_path_as_url(self, monkeypatch):
        import webbrowser
        calls = []
        monkeypatch.setattr(webbrowser, "open", lambda u: calls.append(u))
        dispatcher = Dispatcher(computer_use=None)
        intent = Intent(tool="open_app", params={"url": r"\\attacker\share\x"})
        result = asyncio.run(dispatcher.execute(intent))
        assert result["status"] == "error"
        assert calls == []

    def test_refuses_a_custom_uri_scheme_as_url(self, monkeypatch):
        import webbrowser
        calls = []
        monkeypatch.setattr(webbrowser, "open", lambda u: calls.append(u))
        dispatcher = Dispatcher(computer_use=None)
        intent = Intent(tool="open_app", params={"url": "myapp://do-something-bad"})
        result = asyncio.run(dispatcher.execute(intent))
        assert result["status"] == "error"
        assert calls == []
