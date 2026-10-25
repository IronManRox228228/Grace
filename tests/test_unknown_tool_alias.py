"""Holo invents tool names ("cua_describe_screen"). The loop must resolve
them against the real registry before the safety guard, never ask the user
about a name that does not exist, and never let an alias dodge a confirmation.
"""

import itertools
import json
import logging
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.agent.loop import AgentLoop
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.intent.tools import resolve_tool_name


def _snap(i):
    return ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title=f"Page [{i}]", class_name="X", rect=(0, 0, 800, 600)),
        ocr_lines=[], width=1920, height=1080, graph=None, ui_elements=None,
    )


def _loop(responses):
    gemma = AsyncMock()
    gemma.generate_text.side_effect = responses
    dispatcher = AsyncMock()
    dispatcher.execute.return_value = {"status": "ok"}
    perception = MagicMock()
    screens = (_snap(i) for i in itertools.count())
    perception.capture_snapshot.side_effect = lambda *a, **k: next(screens)
    perception.capture_snapshot_async.return_value = _snap(0)
    loop = AgentLoop(gemma=gemma, dispatcher=dispatcher, perception=perception)
    return loop, gemma, dispatcher


def _plan(action, params=None, **extra):
    return json.dumps({"thought": "t", "action": action, "params": params or {}, **extra})


DONE = _plan("converse", {}, is_completed=True, final_response="Done.")
KEY = _plan("cua_press_key", {"key": "Return"})


class TestResolveToolName:
    @pytest.mark.parametrize("given,expected", [
        ("cua_describe_screen", "describe_screen"),
        ("describe_screen", "describe_screen"),
        ("click", "cua_click"),
        ("press_key", "cua_press_key"),
        ("screenshot", "cua_screenshot"),
        ("scroll", "cua_scroll"),
        ("type_text", "cua_type_text"),
        ("type", "cua_type_text"),
        ("CUA_Click", "cua_click"),
        ("cua_converse", "converse"),
    ])
    def test_aliases(self, given, expected):
        assert resolve_tool_name(given)[0] == expected

    def test_exact_name_is_not_an_alias(self):
        assert resolve_tool_name("cua_click") == ("cua_click", "exact")

    @pytest.mark.parametrize("given", ["frobnicate", "cua_frobnicate", "", "cua_"])
    def test_unknown_stays_unknown(self, given):
        assert resolve_tool_name(given)[0] is None


@pytest.mark.asyncio
class TestLoopResolvesBeforeSafety:
    async def test_cua_describe_screen_runs_describe_screen(self, caplog):
        loop, _, dispatcher = _loop([_plan("cua_describe_screen"), KEY, DONE])
        with caplog.at_level(logging.INFO, logger="grace.agent.loop"):
            res = await loop.run("look at the screen")
        assert res["status"] != "safety_confirmation_required"
        tools = [c.args[0].tool for c in dispatcher.execute.await_args_list]
        assert tools[0] == "describe_screen"
        assert any("cua_describe_screen" in r.getMessage() and "describe_screen" in r.getMessage()
                   for r in caplog.records)

    async def test_alias_to_confirmation_tool_still_asks(self):
        loop, _, dispatcher = _loop([_plan("cua_delete_file", {"name": "a.txt"})])
        res = await loop.run("clean up")
        assert res["status"] == "safety_confirmation_required"
        assert res["pending_step"]["action"] == "delete_file"
        dispatcher.execute.assert_not_awaited()

    async def test_alias_to_close_app_still_asks(self):
        loop, _, dispatcher = _loop([_plan("cua_close_app", {"name": "edge"})])
        res = await loop.run("close it")
        assert res["status"] == "safety_confirmation_required"
        dispatcher.execute.assert_not_awaited()


@pytest.mark.asyncio
class TestUnknownToolIsPlannerVisible:
    async def test_unknown_tool_never_parks_a_confirmation(self):
        loop, gemma, dispatcher = _loop([_plan("cua_frobnicate"), KEY, DONE])
        res = await loop.run("do a thing")
        assert res["status"] == "ok"
        assert loop._pending is None
        tools = [c.args[0].tool for c in dispatcher.execute.await_args_list]
        assert "cua_frobnicate" not in tools
        second_prompt = gemma.generate_text.await_args_list[1].kwargs["prompt"]
        assert "Unknown tool cua_frobnicate. Valid tools:" in second_prompt
        assert "describe_screen" in second_prompt

    async def test_repeated_unknown_tools_end_the_run(self):
        loop, _, dispatcher = _loop([_plan("cua_frobnicate")] * 10)
        res = await loop.run("do a thing")
        assert res["status"] == "planner_failed"
        dispatcher.execute.assert_not_awaited()

    async def test_unknown_name_with_is_completed_is_a_completion_claim(self):
        loop, _, dispatcher = _loop([KEY, _plan("done", {}, is_completed=True, final_response="Playing.")])
        res = await loop.run("press return")
        assert res["status"] == "ok"
        assert res["final_response"] == "Playing."
        tools = [c.args[0].tool for c in dispatcher.execute.await_args_list]
        assert tools == ["cua_press_key"]
