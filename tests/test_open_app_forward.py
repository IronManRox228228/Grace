"""Cockpit 4 item 3b: open_app prefers the installed app and brings the window forward."""

import asyncio
import types

import pytest

from grace.automation import computer_use as cu_mod
from grace.automation.computer_use import ComputerUse
from grace.intent.parser import Intent
from grace.tools.dispatcher import Dispatcher


class FakeCua:
    """A desktop that lists windows and records activations. Nothing real."""

    is_ready = True

    def __init__(self, before, after, activate_ok=True):
        self.snapshots = [before, after]
        self.calls = []
        self.activate_ok = activate_ok

    def perform(self, action, params, confirmed=False):
        self.calls.append((action, params))
        if action == "list_windows":
            windows = self.snapshots[0] if len(self.calls) == 1 else self.snapshots[1]
            return {"ok": True, "windows": windows}
        if action == "activate":
            if self.activate_ok:
                return {"ok": True, "action": "activate"}
            return {"ok": False, "error": "Activation failed"}
        return {"ok": True}


class FakeIndexer:
    def __init__(self, installed=True):
        self.installed = installed
        self.launched = []

    def is_installed(self, name):
        return self.installed

    def launch(self, name):
        self.launched.append(name)
        return {"status": "ok", "text": f"I've opened {name}."}


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    real_sleep = asyncio.sleep

    async def fast(delay, *a, **k):
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", fast)


def run(dispatcher, params, goal=""):
    dispatcher.set_goal(goal)
    return asyncio.run(dispatcher.execute(Intent(tool="open_app", params=params)))


def test_installed_app_with_url_launches_the_app_and_ignores_the_url(monkeypatch):
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda u: opened.append(u))
    indexer = FakeIndexer(installed=True)
    d = Dispatcher(computer_use=FakeCua([], [{"id": 5, "title": "WhatsApp"}]))
    d.set_app_indexer(indexer)
    res = run(d, {"name": "WhatsApp", "url": "https://web.whatsapp.com"}, goal="open whatsapp and message sam")
    assert indexer.launched == ["WhatsApp"] and opened == []
    assert res["status"] == "ok"


def test_url_stands_when_the_goal_says_browser_web_or_online(monkeypatch):
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda u: opened.append(u))
    indexer = FakeIndexer(installed=True)
    d = Dispatcher(computer_use=None)
    d.set_app_indexer(indexer)
    run(d, {"name": "WhatsApp", "url": "https://web.whatsapp.com"}, goal="open whatsapp web")
    assert opened == ["https://web.whatsapp.com"] and indexer.launched == []


def test_url_stands_when_no_installed_app_matches(monkeypatch):
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda u: opened.append(u))
    d = Dispatcher(computer_use=None)
    d.set_app_indexer(FakeIndexer(installed=False))
    run(d, {"name": "Some Site", "url": "https://some.site"})
    assert opened == ["https://some.site"]


def test_launch_activates_the_new_window_and_reports_it():
    cua = FakeCua([{"id": 1, "title": "Editor"}], [{"id": 1, "title": "Editor"}, {"id": 9, "title": "Documents"}])
    d = Dispatcher(computer_use=cua)
    d.set_app_indexer(FakeIndexer())
    res = run(d, {"name": "File Explorer"})
    assert ("activate", {"id": 9, "window_title": "Documents"}) in cua.calls
    assert res["text"] == "Opened File Explorer; it is now in front."


def test_failed_activation_is_reported_as_a_fact():
    cua = FakeCua([], [{"id": 9, "title": "Documents"}], activate_ok=False)
    d = Dispatcher(computer_use=cua)
    d.set_app_indexer(FakeIndexer())
    res = run(d, {"name": "File Explorer"})
    assert res["text"] == "Opened File Explorer; it could not be brought to the front."


def test_no_window_within_the_wait_is_reported_not_guessed():
    cua = FakeCua([{"id": 1, "title": "Editor"}], [{"id": 1, "title": "Editor"}])
    d = Dispatcher(computer_use=cua)
    d.set_app_indexer(FakeIndexer())
    res = run(d, {"name": "File Explorer"})
    assert "could not be brought to the front" in res["text"]
    assert not [c for c in cua.calls if c[0] == "activate"]


class TestMinimisedWindowIsRestoredBeforeForeground:
    def test_show_window_restore_precedes_the_foreground_techniques(self, monkeypatch):
        import win32con
        import win32gui

        order = []
        monkeypatch.setattr(win32gui, "IsIconic", lambda h: True)
        monkeypatch.setattr(win32gui, "ShowWindow", lambda h, cmd: order.append(("show", h, cmd)))
        monkeypatch.setattr(
            ComputerUse, "_apply_activation_technique",
            classmethod(lambda c, h, t: order.append(("technique", t))),
        )
        monkeypatch.setattr(
            ComputerUse, "_verify_foreground",
            classmethod(lambda c, h, timeout_ms=500: (True, h, "WhatsApp")),
        )
        res = ComputerUse()._activate_hwnd(4242, "WhatsApp")
        assert res["ok"] is True
        assert order[0] == ("show", 4242, win32con.SW_RESTORE)
        assert order[1][0] == "technique"

    def test_a_window_that_is_not_minimised_is_not_restored(self, monkeypatch):
        import win32gui

        shown = []
        monkeypatch.setattr(win32gui, "IsIconic", lambda h: False)
        monkeypatch.setattr(win32gui, "ShowWindow", lambda h, cmd: shown.append(cmd))
        monkeypatch.setattr(ComputerUse, "_apply_activation_technique", classmethod(lambda c, h, t: None))
        monkeypatch.setattr(
            ComputerUse, "_verify_foreground", classmethod(lambda c, h, timeout_ms=500: (True, h, "x"))
        )
        ComputerUse()._activate_hwnd(1, "x")
        assert shown == []
