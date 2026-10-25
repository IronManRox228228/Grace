"""Planner pointing coordinates handling in single mode vs fleet mode.

Under Cockpit 1:
- In single mode, x/y in [0-1000] are preserved and scaled to screen coordinates,
  never stripped even if target_name is present. Grounder is not called.
- In fleet mode, planner-written x/y are stripped when target_name is present,
  and the grounder resolves the position.
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
from grace.fleet import build_cpu_fleet_preset, build_gpu_single_preset


def _snap(i):
    return ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title=f"Page [{i}]", class_name="X", rect=(0, 0, 800, 600)),
        ocr_lines=[], width=1920, height=1080, graph=None, ui_elements=None,
        png_bytes=b"png", image_width=1920, image_height=1080,
    )


def _loop(responses, point, fleet_config=None):
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
    loop = AgentLoop(
        gemma=gemma,
        dispatcher=dispatcher,
        perception=perception,
        grounder=grounder,
        fleet_config=fleet_config,
    )
    return loop, dispatcher, grounder


DONE = json.dumps({"action": "converse", "params": {}, "is_completed": True, "final_response": "ok"})


@pytest.mark.asyncio
async def test_single_mode_xy_with_target_name_is_scaled_not_stripped():
    """In single mode, pointing coordinates in [0-1000] scale to screen without stripping or regrounding."""
    click = json.dumps({"action": "cua_click", "params": {"target_name": "Play", "x": 500, "y": 300}})
    loop, dispatcher, grounder = _loop([click, DONE], GroundedPoint(x=1140, y=630), fleet_config=build_gpu_single_preset())
    await loop.run("play it")
    assert grounder.locate.await_count == 0
    sent = dispatcher.execute.await_args_list[0].args[0]
    assert (sent.params["x"], sent.params["y"]) == (960, 324)


@pytest.mark.asyncio
async def test_fleet_mode_xy_with_target_name_is_regrounded():
    """In fleet mode, planner-written coordinates with target_name are stripped and grounder resolves it."""
    click = json.dumps({"action": "cua_click", "params": {"target_name": "Play", "x": 500, "y": 300}})
    loop, dispatcher, grounder = _loop([click, DONE], GroundedPoint(x=1140, y=630), fleet_config=build_cpu_fleet_preset())
    await loop.run("play it")
    assert grounder.locate.await_count == 1
    sent = dispatcher.execute.await_args_list[0].args[0]
    assert (sent.params["x"], sent.params["y"]) == (1140, 630)


@pytest.mark.asyncio
async def test_single_mode_xy_without_any_name_is_scaled():
    """In single mode, coordinates in [0-1000] without target_name are scaled to screen coordinates."""
    click = json.dumps({"action": "cua_click", "params": {"x": 500, "y": 300}})
    loop, dispatcher, grounder = _loop([click, DONE], GroundedPoint(x=1, y=1), fleet_config=build_gpu_single_preset())
    await loop.run("play it")
    assert grounder.locate.await_count == 0
    sent = dispatcher.execute.await_args_list[0].args[0]
    assert (sent.params["x"], sent.params["y"]) == (960, 324)
