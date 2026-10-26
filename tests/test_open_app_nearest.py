"""Cockpit 6 item 3: open_app with no match returns the nearest installed apps to the model."""

import asyncio
import types

from grace.automation.app_indexer import AppIndexer
from grace.tools.dispatcher import Dispatcher
from grace.intent.parser import Intent


def make_indexer(monkeypatch, uwp=None, apps=None):
    monkeypatch.setattr(AppIndexer, "_index_apps", lambda self: None)
    monkeypatch.setattr(AppIndexer, "_scan_uwp_apps", lambda self: None)
    idx = AppIndexer(use_cache=False)
    for name, appid in (uwp or {}).items():
        idx._uwp_apps[name.lower()] = appid
        idx._display[name.lower()] = name
    for name, path in (apps or {}).items():
        idx._apps[name.lower()] = path
        idx._display[name.lower()] = name
    return idx


def test_nearest_apps_ranks_word_overlap_both_directions(monkeypatch):
    idx = make_indexer(
        monkeypatch,
        uwp={"PC Manager": "Microsoft.MicrosoftPCManager_8wekyb3d8bbwe!App", "Microsoft Edge": "e!App",
             "Calculator": "c!App", "Manager Pro": "m!App"},
    )
    near = idx.nearest_apps("Microsoft PC Manager")
    assert near[0] == "PC Manager"
    assert "Calculator" not in near
    assert len(near) <= 5
    # reverse direction: app name contains the query's words
    assert idx.nearest_apps("edge")[0] == "Microsoft Edge"


class Launcher:
    def __init__(self, idx):
        self.idx = idx
        self.launched = []

    def is_installed(self, name):
        return self.idx.is_installed(name)

    def nearest_apps(self, name):
        return self.idx.nearest_apps(name)

    def launch(self, name):
        self.launched.append(name)
        return {"status": "ok", "text": f"I've opened {name}."}


def _open(indexer, name):
    d = Dispatcher(computer_use=None)
    d.set_app_indexer(indexer)
    d.set_goal("")
    return asyncio.run(d.execute(Intent(tool="open_app", params={"name": name})))


def test_no_match_returns_error_with_closest_names_and_does_not_launch(monkeypatch):
    idx = make_indexer(monkeypatch, uwp={"PC Manager": "x!App", "Microsoft Edge": "e!App"})
    ln = Launcher(idx)
    res = _open(ln, "Microsoft PC Manager")
    assert res["status"] == "error"
    assert res["error"].startswith("No installed app is named 'Microsoft PC Manager'. Closest installed apps: 'PC Manager'")
    assert ln.launched == []


def test_exact_name_still_launches(monkeypatch):
    idx = make_indexer(monkeypatch, uwp={"PC Manager": "x!App"})
    ln = Launcher(idx)
    monkeypatch.setattr(Dispatcher, "_bring_forward", lambda *a, **k: _none())
    res = _open(ln, "PC Manager")
    assert res["status"] == "ok" and ln.launched == ["PC Manager"]


async def _none():
    return None


def test_no_match_error_reaches_agent_step_zero_memory():
    from grace.agent.memory import AgentMemory

    msg = "No installed app is named 'X'. Closest installed apps: 'PC Manager'."
    mem = AgentMemory(user_goal="g", max_iterations=5)
    mem.add_step(thought="t", action="open_app", params={"name": "X"},
                 result={"status": "error", "error": msg, "text": msg}, user_update="u", step_number=0)
    assert "Closest installed apps: 'PC Manager'" in mem.format_history_markdown()
