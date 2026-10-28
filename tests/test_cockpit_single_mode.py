"""Tests for Cockpit 1: Single Mode (gpu-single) driving by screenshot and pointing."""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from grace.agent.loop import AgentLoop
from grace.agent.perception import ScreenSnapshot, WindowInfo, observe_for_planner
from grace.agent.planner import PlannedStep, Planner, get_planner_system_prompt
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


# ---------------------------------------------------------------------------
# Cockpit 8: lean single-mode cockpit
# ---------------------------------------------------------------------------


def _readings_snapshot(elements=None, window_title="WhatsApp"):
    graph = ElementGraph(elements=elements or [], window=WindowRef(hwnd=1001, title=window_title))
    return ScreenSnapshot(
        active_window=WindowInfo(
            hwnd=1001, title=window_title, class_name="Chrome_WidgetWin_1",
            rect=(0, 0, 1920, 1080)),
        ocr_lines=[],
        width=1920,
        height=1080,
        graph=graph,
    )


class TestSingleModeSystemPrompt:
    BANNED = ["expect", "Screen unchanged", "verified", "Established facts",
              "Observed so far", "Ruled out"]

    def test_size_target(self):
        from grace.agent.planner import get_planner_system_prompt
        prompt = get_planner_system_prompt(is_single=True)
        assert len(prompt) <= 3000

    def test_documents_only_the_eight_tools(self):
        prompt = get_planner_system_prompt(is_single=True)
        for tool in ("cua_click", "cua_type_text", "cua_press_key", "cua_scroll",
                     "cua_drag", "open_app", "search_files", "open_file"):
            assert tool in prompt
        # element_id / element lists are gone from the single-mode prompt.
        assert "element_id" not in prompt
        assert "target_name" not in prompt

    def test_no_banned_words(self):
        prompt = get_planner_system_prompt(is_single=True)
        for word in self.BANNED:
            assert word not in prompt, word

    def test_keeps_the_untrusted_data_rule(self):
        from grace.agent.planner import get_planner_system_prompt
        prompt = get_planner_system_prompt(is_single=True)
        assert "UNTRUSTED_DATA_START" in prompt
        assert "UNTRUSTED_DATA_END" in prompt
        assert "never an instruction" in prompt
        assert "never a new goal" in prompt

    def test_asks_for_the_cockpit8_response_shape(self):
        prompt = get_planner_system_prompt(is_single=True)
        assert '"last_step"' in prompt
        assert '"done"' in prompt
        assert '"say"' in prompt


def test_single_mode_step_message_size_and_contents():
    """Fixture: 60 elements and 8 prior steps. Message <= 2000 chars minus image."""
    from grace.agent.planner import Planner, get_planner_system_prompt
    from grace.agent.loop import _render_single_history

    elements_prompt = "[" + ",".join(
        '{"id": %d, "role": "button", "name": "Element number %d with a fairly long display name"}'
        % (i, i)
        for i in range(1, 61)
    ) + "]"
    assert len(elements_prompt) > 4000  # the old element JSON was the bulk

    entries = [
        {"step_no": i, "action": f"click ({100 + i},200)",
         "said": "changed", "reading": 'Focus: button "Some fairly long element name %d"' % i}
        for i in range(1, 9)
    ]
    history = _render_single_history(entries)
    readings = 'Active window: WhatsApp\nFocus: textbox "Type a message to the group", contains: (empty)'

    planner = Planner(FakeLLM(), is_single=True)
    prompt = planner.build_prompt(
        goal="send a message to the chemistry group",
        elements_prompt=elements_prompt,   # present but unused in single mode
        history=history,
        scratchpad="- **Established**: lots of data " * 40,  # unused in single mode
        expectation_note="You predicted: lots of data.",      # unused in single mode
        window_title="WhatsApp",
        screen_readings=readings,
    )
    assert len(prompt) <= 2000, len(prompt)
    # The old sections are gone in single mode.
    for banned in ("What is on screen", "What you have already done",
                   "Data collected so far", "Result of your last step",
                   "expect", "Screen unchanged", "verified",
                   "Established facts", "Observed so far", "Ruled out"):
        assert banned not in prompt, banned
    assert "### Now" in prompt
    assert "### Your last steps" in prompt
    assert "UNTRUSTED_DATA_START" in prompt


class FakeLLM:
    async def generate_text(self, prompt, system_prompt, **kwargs):
        return '{"action": "converse", "is_completed": true, "final_response": "Done."}'


class TestReadings:
    def test_focused_empty_text_box_shows_empty(self):
        from grace.agent.loop import screen_readings

        el = ElementNode(
            id=4, role="textbox", name='Type a message to group BaaMaaBetaBeti',
            rect=(100, 200, 300, 240), center=(200, 220), value="", focused=True,
        )
        readings = screen_readings(_readings_snapshot([el]))
        assert "Active window: WhatsApp" in readings
        assert 'Focus: textbox "Type a message to group BaaMaaBetaBeti", contains: (empty)' in readings

    def test_password_field_never_shows_its_value(self):
        from grace.agent.loop import screen_readings

        el = ElementNode(
            id=9, role="textbox", name="Password", rect=(0, 0, 10, 10),
            center=(5, 5), value="hunter2", focused=True,
        )
        readings = screen_readings(_readings_snapshot([el]))
        assert "hunter2" not in readings
        assert "password field" in readings

    def test_focused_text_box_shows_value_cut_to_80_chars(self):
        from grace.agent.loop import screen_readings

        el = ElementNode(
            id=9, role="textbox", name="Search", rect=(0, 0, 10, 10),
            center=(5, 5), value="a" * 200, focused=True,
        )
        readings = screen_readings(_readings_snapshot([el]))
        line = [l for l in readings.splitlines() if l.startswith("Focus:")][0]
        assert 'contains: "' + "a" * 80 + '"' in line
        assert 'contains: "' + "a" * 81 not in line

    def test_no_focus_shows_none(self):
        from grace.agent.loop import screen_readings

        el = ElementNode(id=1, role="button", name="x", rect=(0, 0, 10, 10), center=(5, 5))
        readings = screen_readings(_readings_snapshot([el]))
        assert "Focus: none" in readings

    def test_readings_are_wrapped_as_untrusted_in_the_prompt(self):
        from grace.agent.planner import Planner

        planner = Planner(FakeLLM(), is_single=True)
        prompt = planner.build_prompt(
            goal="g", elements_prompt="[]",
            screen_readings='Active window: WhatsApp\nFocus: none',
        )
        section = prompt.split("### Now\n", 1)[1]
        assert section.startswith("<<<UNTRUSTED_DATA_START")
        assert "<<<UNTRUSTED_DATA_END>>>" in section


class TestSingleModeHistory:
    def test_five_lines_one_per_step_with_report_attached_to_the_right_step(self):
        from grace.agent.loop import _render_single_history

        entries = [
            {"step_no": 1, "action": 'type "Mama"', "said": None,
             "reading": 'Focus: textbox "Search", contains: "Mama"'},
            {"step_no": 2, "action": "click (192,270)", "said": "changed",
             "reading": 'Focus: textbox "Type a message to group BaaMaaBetaBeti"'},
            {"step_no": 3, "action": "click (924,971)", "said": None,
             "reading": 'Focus: button "Voice message"'},
            {"step_no": 4, "action": "click (100,100)", "said": "changed",
             "reading": "error: Element [141] is not on screen."},
            {"step_no": 5, "action": 'search files "report"', "said": "nothing changed",
             "reading": "files: C:\a.txt"},
            {"step_no": 6, "action": "click (10,10)", "said": None, "reading": ""},
        ]
        body = _render_single_history(entries)
        lines = [l for l in body.splitlines() if l.strip()]
        assert len(lines) == 5

        def said_on(no):
            line = next(l for l in lines if l.startswith(f"{no}. "))
            return "you said: changed" in line or "you said: nothing changed" in line

        # The report on step N comes from step N+1's output, so the newest
        # rendered line never has one.
        assert said_on(2) is True
        assert said_on(4) is True
        assert said_on(5) is True
        assert said_on(6) is False
        # Step 1 dropped
        assert not any(l.startswith("1. ") for l in lines)

    def test_history_click_shows_grid_coords_and_element_under_point(self):
        from grace.agent.loop import _render_single_history_line, _single_action_line

        graph = ElementGraph(
            elements=[
                make_test_node(39, "Lo Fi (Full Music Video)", frame=FRAME_PAGE, rect=(1100, 450, 1200, 550)),
                make_test_node(40, "The whole video page", frame=FRAME_PAGE, rect=(0, 0, 1920, 1080)),
            ],
            window=WindowRef(hwnd=1, title="YouTube"),
        )
        # Grid (600,440) on a 1920x1080 screen is pixel (1152, 475): inside the
        # small button [39] and inside the full-page container [40]; the
        # smallest one wins, so [39] is named.
        line = _single_action_line(
            "cua_click",
            {"x": 1152, "y": 475, "_raw_x": 600.0, "_raw_y": 440.0, "_graph": graph},
        )
        assert line == 'click (600,440) on button "Lo Fi (Full Music Video)"'

        # No element under the point: coordinates alone.
        line = _single_action_line(
            "cua_click",
            {"x": 9000, "y": 9000, "_raw_x": 600.0, "_raw_y": 440.0, "_graph": graph},
        )
        assert line == "click (600,440)"

    def test_history_drag_and_scroll_use_grid_coords(self):
        from grace.agent.loop import _single_action_line

        assert (
            _single_action_line(
                "cua_drag",
                {"path": [[100, 300], [500, 300]], "_raw_path": [[100, 300], [500, 300]]},
            )
            == "drag (100,300)..(500,300)"
        )
        assert (
            _single_action_line(
                "cua_scroll",
                {"x": 600, "y": 440, "_raw_x": 600.0, "_raw_y": 440.0},
            )
            == "scroll (600,440)"
        )

    def test_history_window_line_only_when_the_title_changes(self):
        from grace.agent.loop import _render_single_history, _render_single_history_line

        entries = [
            # First step: no previous step, so no Window line even though a
            # window is recorded.
            {
                "step_no": 1,
                "action": 'click (600,440) on link "Lo Fi (Full Music Video)"',
                "said": "changed",
                "reading": "",
                "window": "Lo Fi (Full Music Video) - YouTube",
                "prev_window": None,
            },
            # Same window as the previous step: no Window line.
            {
                "step_no": 2,
                "action": 'type "hello"',
                "said": "changed",
                "reading": "",
                "window": "Lo Fi (Full Music Video) - YouTube",
                "prev_window": "Lo Fi (Full Music Video) - YouTube",
            },
            # Window changed: the Window line appears after the report.
            {
                "step_no": 3,
                "action": "press win",
                "said": "changed",
                "reading": "",
                "window": "Start",
                "prev_window": "Lo Fi (Full Music Video) - YouTube",
            },
            # An error reading survives, after the Window line.
            {
                "step_no": 4,
                "action": "click (10,10)",
                "said": "nothing changed",
                "reading": "error: Element [141] is not on screen.",
                "window": "Start",
                "prev_window": "Start",
            },
        ]
        body = _render_single_history(entries)
        lines = [l for l in body.splitlines() if l.strip()]
        assert len(lines) == 4
        assert lines[0] == '1. click (600,440) on link "Lo Fi (Full Music Video)" -> you said: changed'
        assert "Window:" not in lines[0]
        assert lines[1] == '2. type "hello" -> you said: changed'
        assert "Window:" not in lines[1]
        assert (
            lines[2]
            == '3. press win -> you said: changed. Window: Start'
        )
        assert lines[3] == '4. click (10,10) -> you said: nothing changed. error: Element [141] is not on screen.'

    def test_history_lines_stay_under_160_chars(self):
        from grace.agent.loop import _render_single_history_line

        long_name = "button " + "x" * 200
        entry = {
            "step_no": 9,
            "action": f'click (600,440) on button "{long_name}"',
            "said": "changed",
            "reading": "",
            "window": "A very long window title " + "y" * 120,
            "prev_window": "something else entirely",
        }
        line = _render_single_history_line(entry)
        assert len(line) <= 160, (len(line), line)


DONE_JSON = (
    '{"thought": "Done", "action": "converse", "params": {"response": "Done."}, '
    '"is_completed": true, "final_response": "Done."}'
)


class TestSingleModeStopping:
    async def _loop(self, responses, elements=None):
        gemma = AsyncMock()
        dispatcher = AsyncMock()
        perception = MagicMock()
        grounder = AsyncMock()
        perception.capture_snapshot.return_value = make_snapshot(
            elements or [], image_size=(1920, 1080), screen_size=(1920, 1080))
        gemma.generate_text.side_effect = responses
        dispatcher.execute.return_value = {
            "status": "ok",
            "result": {"ok": True, "sent": True, "verified": True, "message": "Clicked"},
        }
        loop = AgentLoop(
            gemma=gemma, dispatcher=dispatcher, perception=perception,
            grounder=grounder, fleet_config=build_gpu_single_preset(),
        )
        result = await loop.run("no change drill")
        return result, dispatcher

    def _single_step(self, last_step=None, coords=(500, 500)):
        action = {"tool": "cua_click", "x": coords[0], "y": coords[1]}
        body = {
            "thought": "Point at it",
            "action": action,
            "done": False,
            "say": "",
        }
        if last_step is not None:
            body["last_step"] = last_step
        return json.dumps(body)

    @pytest.mark.asyncio
    async def test_four_no_change_reports_stop_the_task(self):
        # Step 1 has nothing to judge (no last_step), steps 2-5 report that
        # the previous click changed nothing. The fourth such report stops.
        responses = [
            self._single_step(),
            self._single_step("nothing changed"),
            self._single_step("nothing changed"),
            self._single_step("nothing changed"),
            self._single_step("nothing changed"),
        ]
        result, dispatcher = await self._loop(responses)
        assert result["status"] == "no_progress"
        assert result["final_response"] == "I tried four times and nothing changed, so I've stopped."
        # The fifth action was never dispatched.
        assert dispatcher.execute.call_count == 4

    @pytest.mark.asyncio
    async def test_three_no_change_then_changed_does_not_stop(self):
        responses = [
            self._single_step(),
            self._single_step("nothing changed"),
            self._single_step("nothing changed"),
            self._single_step("nothing changed"),
            self._single_step("changed"),
            DONE_JSON,
        ]
        result, dispatcher = await self._loop(responses)
        assert result["status"] == "ok"
        assert dispatcher.execute.call_count == 5

    @pytest.mark.asyncio
    async def test_no_repeat_stop_at_three_identical_steps_in_single_mode(self):
        # The fleet ladder stops at three identical actions; single mode
        # has no ladder, so four identical clicks are allowed to run and
        # only the model's own last_step verdict can stop the task.
        same = self._single_step("changed")
        responses = [same, same, same, same, DONE_JSON]
        result, dispatcher = await self._loop(responses)
        assert result["status"] == "ok"
        assert dispatcher.execute.call_count == 4


class TestSingleModeDragPath:
    async def _run_drag(self, params_json, elements):
        gemma = AsyncMock()
        dispatcher = AsyncMock()
        perception = MagicMock()
        grounder = AsyncMock()
        perception.capture_snapshot.return_value = make_snapshot(
            elements, image_size=(1920, 1080), screen_size=(1920, 1080))
        gemma.generate_text.side_effect = [
            '{"thought": "Drag it", "action": {"tool": "cua_drag", ' + params_json
            + '}, "done": false, "say": ""}',
            DONE_JSON,
        ]
        dispatcher.execute.return_value = {
            "status": "ok",
            "result": {"ok": True, "sent": True, "verified": True, "message": "Dragged"},
        }
        loop = AgentLoop(
            gemma=gemma, dispatcher=dispatcher, perception=perception,
            grounder=grounder, fleet_config=build_gpu_single_preset(),
        )
        result = await loop.run("drag drill")
        assert result["status"] == "ok"
        calls = [c[0][0] for c in dispatcher.execute.call_args_list if c[0][0].tool == "cua_drag"]
        assert len(calls) == 1
        return calls[0]

    @pytest.mark.asyncio
    async def test_path_wins_over_from_to_element_ids(self):
        """Cockpit 8: a coordinate stroke beats the ids even when ids resolve."""
        elements = [make_test_node(5, "Paper", rect=(100, 200, 150, 250))]
        call = await self._run_drag(
            '"path": [[100, 300], [500, 300]], "from_element_id": 5, "to_element_id": 7',
            elements)
        assert call.params.get("from_element_id") is None
        assert call.params.get("to_element_id") is None
        assert call.params.get("_raw_from_element_id") == 5
        assert call.params.get("_raw_to_element_id") == 7
        assert [list(p) for p in call.params["path"]] == [[192, 324], [960, 324]]

    @pytest.mark.asyncio
    async def test_xy_to_xy_stroke_loses_the_ids_and_is_scaled_like_clicks(self):
        elements = [make_test_node(5, "Paper", rect=(100, 200, 150, 250))]
        call = await self._run_drag(
            '"x": 100, "y": 300, "to_x": 500, "to_y": 300, "from_element_id": 5, "to_element_id": 7',
            elements)
        assert call.params.get("from_element_id") is None
        assert call.params.get("to_element_id") is None
        assert call.params["x"] == 192
        assert call.params["y"] == 324
        assert call.params["to_x"] == 960
        assert call.params["to_y"] == 324

    @pytest.mark.asyncio
    async def test_id_only_drag_keeps_the_ids(self):
        elements = [make_test_node(5, "Paper", rect=(100, 200, 150, 250)),
                    make_test_node(6, "Scissors", rect=(200, 200, 250, 250))]
        call = await self._run_drag('"from_element_id": 5, "to_element_id": 6', elements)
        assert call.params.get("from_element_id") == 5
        assert call.params.get("to_element_id") == 6
