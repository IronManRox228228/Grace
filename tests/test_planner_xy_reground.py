"""Holo-written x/y are never trusted when the step also names its target.

The planner prompt says there is no x/y. If one arrives anyway it is in an
unknown coordinate space (Holo's habit is [0, 1000]), and clicking it raw lands
nowhere near the control. The grounder converts properly, so it is asked instead.
"""

import itertools
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.agent.grounder import GroundedPoint
from grace.agent.loop import AgentLoop
from grace.agent.perception import ScreenSnapshot, WindowInfo


def _snap(i):
    return ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title=f"Page [{i}]", class_name="X", rect=(0, 0, 800, 600)),
        ocr_lines=[], width=1920, height=1080, graph=None, ui_elements=None,
        png_bytes=b"png", image_width=1920, image_height=1080,
    )


def _loop(responses, point):
    gemma = AsyncMock()
    gemma.generate_text.side_effect = responses
    dispatcher = AsyncMock()
    dispatcher.execute.return_value = {"status": "ok", "result": {"ok": True, "verified": True}}
    perception = MagicMock()
    screens = (_snap(i) for i in itertools.count())
    perception.capture_snapshot.side_effect = lambda *a, **k: next(screens)
    perception.capture_snapshot_async.return_value = _snap(0)
    grounder = MagicMock()
    grounder.locate = AsyncMock(return_value=point)
    loop = AgentLoop(gemma=gemma, dispatcher=dispatcher, perception=perception, grounder=grounder)
    return loop, dispatcher, grounder


DONE = json.dumps({"action": "converse", "params": {}, "is_completed": True, "final_response": "ok"})


@pytest.mark.asyncio
async def test_planner_xy_with_target_name_is_regrounded():
    click = json.dumps({"action": "cua_click", "params": {"target_name": "Play", "x": 500, "y": 300}})
    loop, dispatcher, grounder = _loop([click, DONE], GroundedPoint(x=1140, y=630))
    await loop.run("play it")
    assert grounder.locate.await_count == 1
    sent = dispatcher.execute.await_args_list[0].args[0]
    assert (sent.params["x"], sent.params["y"]) == (1140, 630)


@pytest.mark.asyncio
async def test_xy_without_any_name_is_left_alone():
    click = json.dumps({"action": "cua_click", "params": {"x": 500, "y": 300}})
    loop, dispatcher, grounder = _loop([click, DONE], GroundedPoint(x=1, y=1))
    await loop.run("play it")
    assert grounder.locate.await_count == 0
    sent = dispatcher.execute.await_args_list[0].args[0]
    assert (sent.params["x"], sent.params["y"]) == (500, 300)
