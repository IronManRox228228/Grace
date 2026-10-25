"""Tests for fleet mode element list and observation modality (cockpit-1 Commit 3).

1. Fleet mode (cpu-fleet) prompt contains the element list (hide_tree is False).
2. Ordinary rich step in fleet mode sends no image_b64 to planner.
3. Blind window in fleet mode still triggers SomOverlay / marked screenshot fallback.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from grace.agent.loop import AgentLoop
from grace.agent.perception import (
    ScreenSnapshot,
    WindowInfo,
    OBSERVABILITY_BLIND,
    OBSERVABILITY_RICH,
    observe_for_planner,
)
from grace.fleet import build_cpu_fleet_preset
from grace.perception.element_graph import ElementGraph, WindowRef
from grace.perception.elements import ElementNode, FRAME_PAGE


def make_test_node(element_id: int, name: str, frame: str = FRAME_PAGE, rect=(100, 100, 200, 150)):
    return ElementNode(
        id=element_id,
        role="button",
        name=name,
        rect=rect,
        center=((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2),
        frame=frame,
    )


def make_snapshot(elements, image_size=(1280, 720), screen_size=(1920, 1080)):
    graph = ElementGraph(
        elements=elements,
        window=WindowRef(hwnd=1001, title="Fleet Test Window", class_name="Chrome_WidgetWin_1"),
    )
    return ScreenSnapshot(
        active_window=WindowInfo(
            hwnd=1001,
            title="Fleet Test Window",
            class_name="Chrome_WidgetWin_1",
            rect=(0, 0, screen_size[0], screen_size[1]),
        ),
        ocr_lines=[],
        width=screen_size[0],
        height=screen_size[1],
        png_bytes=b"fake-png-bytes-fleet-test",
        graph=graph,
        image_width=image_size[0],
        image_height=image_size[1],
    )


@pytest.mark.asyncio
async def test_fleet_mode_prompt_contains_element_list():
    """Fleet mode prompt contains the interactive elements list."""
    gemma = AsyncMock()
    dispatcher = AsyncMock()
    perception = MagicMock()
    grounder = AsyncMock()

    elements = [
        make_test_node(10, "Submit Form"),
        make_test_node(11, "Cancel Operation"),
    ]
    snap = make_snapshot(elements)
    perception.capture_snapshot.return_value = snap

    # Step 1: Model sees elements and clicks
    gemma.generate_text.side_effect = [
        '{"thought": "Click submit", "action": "cua_click", "params": {"element_id": 10}, "is_completed": true, "final_response": "Done"}',
    ]

    dispatcher.execute.return_value = {"status": "ok", "result": {"ok": True}}

    fleet_cfg = build_cpu_fleet_preset()
    loop = AgentLoop(
        gemma=gemma,
        dispatcher=dispatcher,
        perception=perception,
        grounder=grounder,
        fleet_config=fleet_cfg,
    )
    loop._verify_goal_completion = MagicMock(return_value=(True, None))

    result = await loop.run("submit the form")
    assert result["status"] == "ok"

    # Check that prompt contained the interactive elements list
    assert gemma.generate_text.call_count == 1
    call_kwargs = gemma.generate_text.call_args_list[0][1]
    prompt = call_kwargs["prompt"]
    assert "Submit Form" in prompt
    assert "Cancel Operation" in prompt
    assert "10" in prompt


@pytest.mark.asyncio
async def test_fleet_mode_ordinary_step_sends_no_image():
    """An ordinary rich step in fleet mode sends no image_b64."""
    gemma = AsyncMock()
    dispatcher = AsyncMock()
    perception = MagicMock()
    grounder = AsyncMock()

    elements = [make_test_node(1, "Search")]
    snap = make_snapshot(elements)
    perception.capture_snapshot.return_value = snap

    gemma.generate_text.side_effect = [
        '{"thought": "Search", "action": "cua_click", "params": {"element_id": 1}, "is_completed": true, "final_response": "Searched"}',
    ]
    dispatcher.execute.return_value = {"status": "ok", "result": {"ok": True}}

    fleet_cfg = build_cpu_fleet_preset()
    loop = AgentLoop(
        gemma=gemma,
        dispatcher=dispatcher,
        perception=perception,
        grounder=grounder,
        fleet_config=fleet_cfg,
    )
    loop._verify_goal_completion = MagicMock(return_value=(True, None))

    await loop.run("search")

    assert gemma.generate_text.call_count == 1
    call_kwargs = gemma.generate_text.call_args_list[0][1]
    # In fleet mode on rich window, image_b64 must be None
    assert call_kwargs.get("image_b64") is None


def test_observe_for_planner_fleet_mode_blind_window_fallback():
    """observe_for_planner in fleet mode falls back to SomOverlay on blind windows."""
    elements = [make_test_node(1, "Save")]
    snap = make_snapshot(elements)
    # Mark it blind
    snap.active_window.class_name = "ConsoleWindowClass"

    with patch("grace.automation.som_overlay.marks_for") as mock_marks, \
         patch("grace.automation.som_overlay.render_or_none") as mock_render, \
         patch("grace.automation.som_overlay.legend") as mock_legend:
        mock_marks.return_value = ["dummy_mark"]
        mock_render.return_value = b"marked-png-bytes"
        mock_legend.return_value = "1. Save"

        obs = observe_for_planner(snap, is_single=False)
        assert obs.mode == OBSERVABILITY_BLIND
        assert obs.image_b64 is not None
        assert "Save" in obs.markdown
