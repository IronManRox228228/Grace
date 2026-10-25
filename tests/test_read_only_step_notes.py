"""Read-only steps must not be reported as failed progress, and
describe_screen output must reach the (blind) planner whole.

Run evidence (2026-10-25_21-02-50 step 5): after cua_screenshot the planner was
told "NO VISIBLE EFFECT ... choose a different action", which is meaningless for
a screenshot. And describe_screen builds up to 1200 characters, of which the
note kept 600, usually dropping the page results the planner needed.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.agent.loop import AgentLoop
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.agent.planner import PlannedStep


def _loop():
    return AgentLoop(gemma=AsyncMock(), dispatcher=AsyncMock(), perception=MagicMock())


def _snap():
    return ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title="W", class_name="X", rect=(0, 0, 8, 6)),
        ocr_lines=[], width=1920, height=1080, graph=None, ui_elements=None,
    )


@pytest.mark.parametrize("tool", ["cua_screenshot", "describe_screen", "cua_text",
                                  "cua_list_windows", "cua_list_apps", "cua_get_window"])
def test_read_only_tool_never_gets_no_visible_effect(tool):
    step = PlannedStep(action=tool)
    note = _loop()._expectation_note(step, {"status": "ok"}, _snap(), repeat_count=1)
    assert "The screen has not changed" not in note and "NO VISIBLE EFFECT" not in note
    assert "You have now done this exact action" not in note


def test_click_still_gets_no_visible_effect():
    step = PlannedStep(action="cua_click")
    note = _loop()._expectation_note(step, {"status": "ok"}, _snap(), repeat_count=1)
    assert "The screen has not changed" not in note and "NO VISIBLE EFFECT" not in note
    assert "You have now done this exact action 1 times on this same screen." in note


def test_describe_screen_text_is_not_cut_at_600():
    text = "Active window: W\n" + "\n".join(f"- Item: 'line {i:03d} xxxxxxxxxxxx'" for i in range(40))
    text = text[:1150] + " - Song: 'Signal'"
    step = PlannedStep(action="describe_screen")
    note = _loop()._expectation_note(step, {"status": "ok", "text": text}, _snap())
    assert "Song: 'Signal'" in note
