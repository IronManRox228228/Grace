"""Tests for Cockpit 1: Single Mode (gpu-single) driving by screenshot and pointing."""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from grace.agent.loop import AgentLoop
from grace.agent.perception import ScreenSnapshot, WindowInfo, observe_for_planner
from grace.agent.planner import PlannedStep, Planner
from grace.agent.grounder import GroundedPoint
from grace.fleet import build_gpu_single_preset, build_cpu_fleet_preset
from grace.perception.element_graph import ElementGraph, WindowRef
from grace.perception.elements import ElementNode, FRAME_CHROME, FRAME_PAGE, elements_to_prompt


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
        window=WindowRef(hwnd=1001, title="Test Window", class_name="Chrome_WidgetWin_1"),
    )
    return ScreenSnapshot(
        active_window=WindowInfo(hwnd=1001, title="Test Window", class_name="Chrome_WidgetWin_1", rect=(0, 0, screen_size[0], screen_size[1])),
        ocr_lines=[],
        width=screen_size[0],
        height=screen_size[1],
        png_bytes=b"fake-png-bytes-for-single-mode",
        graph=graph,
        image_width=image_size[0],
        image_height=image_size[1],
    )


def test_page_frame_elements_appear_first():
    """Page-frame elements come first in elements_to_prompt."""
    chrome1 = make_test_node(1, "Minimize", frame=FRAME_CHROME)
    chrome2 = make_test_node(2, "Close", frame=FRAME_CHROME)
    page1 = make_test_node(39, "SignalHome", frame=FRAME_PAGE)
    page2 = make_test_node(40, "OtherSong", frame=FRAME_PAGE)

    prompt = elements_to_prompt([chrome1, chrome2, page1, page2], limit=40)
    # The JSON array should list page elements first
    assert prompt.index('"id": 39') < prompt.index('"id": 1')
    assert prompt.index('"id": 40') < prompt.index('"id": 2')


def test_planner_call_includes_image_b64_on_every_step_in_single_mode():
    """Single mode observation attaches image_b64 to every step."""
    snap = make_snapshot([make_test_node(1, "Button")])
    view_single = observe_for_planner(snap, is_single=True)
    assert view_single.image_b64 is not None
    assert base64.b64decode(view_single.image_b64) == b"fake-png-bytes-for-single-mode"

    # Fleet mode should not have image_b64 on rich steps
    view_fleet = observe_for_planner(snap, is_single=False)
    assert view_fleet.image_b64 is None


@pytest.mark.asyncio
async def test_shown_element_id_plus_target_name_never_invokes_grounder():
    """Step 7: shown element_id=39 plus a target_name never invokes the grounder, and clicks 39."""
    gemma = AsyncMock()
    dispatcher = AsyncMock()
    perception = MagicMock()
    grounder = AsyncMock()

    elements = [make_test_node(39, "SignalHome", frame=FRAME_PAGE, rect=(380, 318, 839, 366))]
    snap = make_snapshot(elements)
    perception.capture_snapshot.return_value = snap

    # Holo returns cua_click with element_id=39 and target_name, then completes
    gemma.generate_text.side_effect = [
        '{"thought": "Click song", "action": "cua_click", "params": {"element_id": 39, "target_name": "Signal song result entry", "click_count": 1}, "is_completed": false, "user_update": "Clicking song"}',
        '{"thought": "Done", "action": "converse", "params": {"response": "Playing"}, "is_completed": true, "final_response": "Playing"}',
    ]

    dispatcher.execute.return_value = {
        "status": "ok",
        "result": {"ok": True, "sent": True, "verified": True, "message": "Clicked at (609, 342)"},
    }

    loop = AgentLoop(
        gemma=gemma,
        dispatcher=dispatcher,
        perception=perception,
        grounder=grounder,
        fleet_config=build_gpu_single_preset(),
    )

    result = await loop.run("play signal")
    assert result["status"] == "ok"

    # Grounder must NEVER be called
    assert grounder.locate.call_count == 0

    # Dispatcher was called with element_id=39
    calls = [c[0][0] for c in dispatcher.execute.call_args_list if c[0][0].tool == "cua_click"]
    assert len(calls) == 1
    assert calls[0].params.get("element_id") == 39


@pytest.mark.asyncio
async def test_pointing_x_y_converted_and_not_stripped_with_target_name():
    """Holo-style x=258, y=320 in [0-1000] on 1920x1080 clicks screen (495, 346), with target_name present and not stripped."""
    gemma = AsyncMock()
    dispatcher = AsyncMock()
    perception = MagicMock()
    grounder = AsyncMock()

    snap = make_snapshot([], image_size=(1920, 1080), screen_size=(1920, 1080))
    perception.capture_snapshot.return_value = snap

    gemma.generate_text.side_effect = [
        '{"thought": "Click coordinates", "action": "cua_click", "params": {"x": 258, "y": 320, "target_name": "Signal song result entry"}, "is_completed": false, "user_update": "Clicking"}',
        '{"thought": "Done", "action": "converse", "params": {"response": "Playing"}, "is_completed": true, "final_response": "Playing"}',
    ]

    dispatcher.execute.return_value = {
        "status": "ok",
        "result": {"ok": True, "sent": True, "verified": True, "message": "Clicked"},
    }

    loop = AgentLoop(
        gemma=gemma,
        dispatcher=dispatcher,
        perception=perception,
        grounder=grounder,
        fleet_config=build_gpu_single_preset(),
    )

    result = await loop.run("click target")
    assert result["status"] == "ok"

    # Grounder was not called because coordinates were supplied
    assert grounder.locate.call_count == 0

    # Dispatcher received the scaled coordinates: x=495, y=346 and target_name was NOT stripped
    calls = [c[0][0] for c in dispatcher.execute.call_args_list if c[0][0].tool == "cua_click"]
    assert len(calls) == 1
    assert calls[0].params.get("x") == 495
    assert calls[0].params.get("y") == 346
    assert calls[0].params.get("target_name") == "Signal song result entry"


@pytest.mark.asyncio
async def test_unshown_element_id_without_fallback_gives_honest_error():
    """Unshown element_id=141 with no x/y and no target_name gives an honest error listing shown ids."""
    gemma = AsyncMock()
    dispatcher = AsyncMock()
    perception = MagicMock()
    grounder = AsyncMock()

    elements = [make_test_node(1, "Search", frame=FRAME_PAGE), make_test_node(2, "Home", frame=FRAME_PAGE)]
    snap = make_snapshot(elements)
    perception.capture_snapshot.return_value = snap

    # Step 1: Model asks for unshown id 141 with no fallback
    # Step 2: Model recovers with converse
    gemma.generate_text.side_effect = [
        '{"thought": "Click 141", "action": "cua_click", "params": {"element_id": 141}, "is_completed": false, "user_update": "Clicking 141"}',
        '{"thought": "Give up", "action": "converse", "params": {"response": "Cannot find it"}, "is_completed": true, "final_response": "Cannot find it"}',
    ]
    dispatcher.execute.return_value = {
        "status": "ok",
        "result": {"ok": True, "sent": True, "verified": True, "message": "Done"},
    }

    loop = AgentLoop(
        gemma=gemma,
        dispatcher=dispatcher,
        perception=perception,
        grounder=grounder,
        fleet_config=build_gpu_single_preset(),
    )

    result = await loop.run("test unshown id")
    assert result["status"] == "ok"

    # Step 1 should have failed honestly with error listing shown ids
    steps = result["steps"]
    assert len(steps) >= 1
    assert "Element [141] is not on screen" in steps[0]["result"]["error"]
    assert "[1, 2]" in steps[0]["result"]["error"]


async def _run_click(planner_params: str, preset, elements):
    gemma = AsyncMock()
    dispatcher = AsyncMock()
    perception = MagicMock()
    grounder = AsyncMock()
    perception.capture_snapshot.return_value = make_snapshot(elements, image_size=(1920, 1080))
    gemma.generate_text.side_effect = [
        '{"thought": "Click", "action": "cua_click", "params": ' + planner_params + ', "is_completed": false, "user_update": "Clicking"}',
        '{"thought": "Done", "action": "converse", "params": {"response": "Done"}, "is_completed": true, "final_response": "Done"}',
    ]
    dispatcher.execute.return_value = {
        "status": "ok",
        "result": {"ok": True, "sent": True, "verified": True, "message": "Clicked"},
    }
    loop = AgentLoop(gemma=gemma, dispatcher=dispatcher, perception=perception, grounder=grounder, fleet_config=preset)
    result = await loop.run("test both")
    assert result["status"] == "ok"
    calls = [c[0][0] for c in dispatcher.execute.call_args_list if c[0][0].tool == "cua_click"]
    assert len(calls) == 1
    return calls[0], grounder


@pytest.mark.asyncio
async def test_single_mode_xy_wins_over_element_id():
    """Cockpit 6 item 1: Holo step-2 params click at the scaled (100, 230), not element 4."""
    elements = [
        make_test_node(4, "Chats", rect=(10, 10, 50, 50)),
        make_test_node(5, "Mamma", rect=(300, 300, 400, 400)),
    ]
    call, grounder = await _run_click('{"element_id": 4, "x": 100, "y": 230}', build_gpu_single_preset(), elements)
    assert call.params.get("element_id") is None
    assert call.params.get("element_index") is None
    assert (round(call.params["x"]), round(call.params["y"])) == (192, 248)
    assert call.params.get("_raw_x") == 100.0
    grounder.locate.assert_not_called()


@pytest.mark.asyncio
async def test_single_mode_id_without_xy_resolves_by_id():
    elements = [make_test_node(5, "PlayButton", rect=(100, 200, 150, 250))]
    call, _ = await _run_click('{"element_id": 5}', build_gpu_single_preset(), elements)
    assert call.params.get("element_id") == 5
    assert call.params.get("x") is None


@pytest.mark.asyncio
async def test_fleet_mode_id_wins_over_xy():
    elements = [make_test_node(5, "PlayButton", rect=(100, 200, 150, 250))]
    call, _ = await _run_click('{"element_id": 5, "x": 500, "y": 500}', build_cpu_fleet_preset(), elements)
    assert call.params.get("element_id") == 5
    assert call.params.get("x") is None
    assert call.params.get("_id_won_over_xy") is True


@pytest.mark.asyncio
async def test_cua_screenshot_leads_to_next_step_prompt_saying_attached():
    """cua_screenshot leads to the next step's prompt saying the screenshot is attached."""
    gemma = AsyncMock()
    dispatcher = AsyncMock()
    perception = MagicMock()
    grounder = AsyncMock()

    elements = [make_test_node(1, "Search", frame=FRAME_PAGE)]
    snap = make_snapshot(elements)
    perception.capture_snapshot.return_value = snap

    # Step 1: Holo takes screenshot
    # Step 2: Holo sees prompt and concludes
    gemma.generate_text.side_effect = [
        '{"thought": "Take screenshot", "action": "cua_screenshot", "params": {}, "is_completed": false, "user_update": "Taking screenshot"}',
        '{"thought": "Done", "action": "converse", "params": {"response": "Saw screenshot"}, "is_completed": true, "final_response": "Saw screenshot"}',
    ]

    dispatcher.execute.return_value = {
        "status": "ok",
        "result": {
            "ok": True,
            "action": "screenshot",
            "png_b64": "fake_b64",
            "message": "screenshot attached to your next step",
            "text": "screenshot attached to your next step",
        },
    }

    loop = AgentLoop(
        gemma=gemma,
        dispatcher=dispatcher,
        perception=perception,
        grounder=grounder,
        fleet_config=build_gpu_single_preset(),
    )
    loop._verify_goal_completion = MagicMock(return_value=(True, None))

    result = await loop.run("inspect screen")
    assert result["status"] == "ok"

    # Verify gemma second call prompt contains "screenshot attached to your next step"
    assert gemma.generate_text.call_count == 2
    second_prompt = gemma.generate_text.call_args_list[1][1]["prompt"]
    assert "screenshot attached to your next step" in second_prompt



def test_step1_holo_reordered_title_acts_on_foreground(monkeypatch):
    """Step 1: Holo reordered title acts on foreground YouTube Music window without wrong_window."""
    from grace.automation.computer_use import ComputerUse

    fg = "YouTube Music and 1 more page - Personal - Microsoft\u200b Edge Beta"
    monkeypatch.setattr("grace.automation.computer_use._foreground_title", lambda: fg)
    monkeypatch.setattr("grace.automation.win32_driver.Win32Driver.click_at", lambda *a, **k: True)

    cu = ComputerUse()
    res = cu._click({
        "window": {"title": "YouTube Music - Microsoft Edge Beta - Personal"},
        "x": 100,
        "y": 100,
    })

    assert res["ok"] is True
    assert res.get("status") != "wrong_window"
    assert res.get("status") != "window_not_found"


def test_step6_shown_id_clicks_centre_and_carries_mismatch_note(monkeypatch):
    """Step 6: element_id=39 (shown) and target_name='SignalHome song entry' clicks 39 center and carries note."""
    from grace.automation.computer_use import ComputerUse

    el = make_test_node(39, "SignalHome • 93K plays•Signal", frame=FRAME_PAGE, rect=(100, 200, 300, 250))
    graph = ElementGraph(elements=[el])

    monkeypatch.setattr("grace.automation.computer_use._foreground_title", lambda: "YouTube Music")
    monkeypatch.setattr("grace.automation.win32_driver.Win32Driver.click_at", lambda *a, **k: True)

    cu = ComputerUse()
    res = cu._click({
        "element_id": 39,
        "target_name": "SignalHome song entry",
        "_graph": graph,
        "_rendered_element_ids": {39},
    })

    assert res["ok"] is True
    assert res["x"] == 200
    assert res["y"] == 225
    assert "(note: clicked [39] 'SignalHome • 93K plays•Signal'; you described it as 'SignalHome song entry')" in res["message"]


def test_truly_absent_window_returns_window_not_found(monkeypatch):
    """A genuinely absent window returns impossible error listing open windows."""
    from grace.automation.computer_use import ComputerUse

    monkeypatch.setattr("grace.automation.computer_use._foreground_title", lambda: "YouTube Music")
    cu = ComputerUse()
    monkeypatch.setattr(cu, "_list_windows", lambda *a, **k: {"ok": True, "windows": [{"id": 101, "title": "YouTube Music"}]})
    monkeypatch.setattr(cu, "_activate", lambda params: {"ok": False, "sent": True, "verified": False})

    res = cu._click({
        "window": {"title": "Calculator"},
        "x": 100,
        "y": 100,
    })

    assert res["ok"] is False
    assert res["status"] == "window_not_found"
    assert "Open windows: 'YouTube Music' (id: 101)" in res["error"]


@pytest.mark.asyncio
async def test_single_mode_parked_click_resumes_with_scaled_coordinates():
    """A click parked for confirmation dispatches the scaled x/y on 'yes', not the raw 0-1000 numbers."""
    gemma = AsyncMock()
    dispatcher = AsyncMock()
    perception = MagicMock()
    grounder = AsyncMock()
    perception.capture_snapshot.return_value = make_snapshot([], image_size=(1920, 1080))
    gemma.generate_text.side_effect = [
        '{"thought": "Click", "action": "cua_click", "params": {"x": 258, "y": 320, "target_name": "Delete"}, "is_completed": false, "user_update": "Clicking"}',
        '{"thought": "Done", "action": "converse", "params": {"response": "Done"}, "is_completed": true, "final_response": "Done"}',
    ]
    dispatcher.execute.return_value = {
        "status": "ok",
        "result": {"ok": True, "sent": True, "verified": True, "message": "Clicked"},
    }
    loop = AgentLoop(gemma=gemma, dispatcher=dispatcher, perception=perception, grounder=grounder,
                     fleet_config=build_gpu_single_preset())

    parked = await loop.run("delete it")
    assert parked["status"] == "safety_confirmation_required"
    assert dispatcher.execute.call_count == 0

    await loop.resume_pending(approved=True)
    calls = [c for c in dispatcher.execute.call_args_list if c[0][0].tool == "cua_click"]
    assert len(calls) == 1
    assert calls[0].kwargs.get("confirmed") is True
    assert (calls[0][0][0].params["x"], calls[0][0][0].params["y"]) == (495, 346)
