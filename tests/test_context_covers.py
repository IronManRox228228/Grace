"""Cockpit 4 item 1: harm covers that need the screen (sign-in pages, terminals, contact details)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from grace.agent.context_covers import evaluate_context
from grace.agent.loop import AgentLoop
from grace.agent.perception import OcrLine, ScreenSnapshot, WindowInfo
from grace.fleet import build_gpu_single_preset
from grace.perception.element_graph import ElementGraph, WindowRef
from grace.perception.elements import ElementNode

AUTH_TITLE = "Sign in – Google accounts - Personal 3 - Microsoft Edge Beta"


def node(eid, name, role="button", **kw):
    return ElementNode(id=eid, role=role, name=name, rect=(0, 0, 100, 40), center=(50, 20), **kw)


def snap(title, class_name="Chrome_WidgetWin_1", elements=(), ocr=()):
    return ScreenSnapshot(
        active_window=WindowInfo(hwnd=0, title=title, class_name=class_name, rect=(0, 0, 1920, 1080)),
        ocr_lines=[OcrLine(text=t, bounding_box=(0, 0, 10, 10)) for t in ocr],
        width=1920,
        height=1080,
        graph=ElementGraph(
            elements=list(elements),
            window=WindowRef(hwnd=0, title=title, class_name=class_name),
        ),
    )


STEP9 = {
    "window": {"title": AUTH_TITLE},
    "element_id": 9,
    "target_name": "Use another account",
    "click_count": 1,
}


class TestSignInCover:
    def test_use_another_account_on_google_chooser_asks_the_user(self):
        s = snap(AUTH_TITLE, elements=[node(9, "Use another account", role="listitem")])
        d = evaluate_context("cua_click", STEP9, s, goal="open slack and read the questions channel")
        assert d is not None and d.kind == "ask_user"
        assert "wants you to sign in" in d.message
        assert "Do you want to sign in yourself first?" in d.message

    def test_url_pattern_marks_an_auth_screen(self):
        s = snap("Account")
        d = evaluate_context(
            "cua_type_text", {"text": "hello"}, s, goal="x", url="https://login.microsoftonline.com/common/oauth2"
        )
        assert d is not None and d.kind == "ask_user"
        assert "login.microsoftonline.com" in d.message

    def test_password_field_marks_an_auth_screen(self):
        s = snap("Some app", class_name="Foo", elements=[node(1, "Password", role="edit")])
        d = evaluate_context("cua_set_value", {"element_id": 1, "value": "x"}, s, goal="x")
        assert d is not None and d.kind == "ask_user"

    def test_goal_that_asks_to_sign_in_stands_the_cover_down(self):
        s = snap(AUTH_TITLE, elements=[node(9, "Use another account", role="listitem")])
        assert evaluate_context("cua_click", STEP9, s, goal="sign in to my google account") is None

    def test_plain_clicks_on_an_auth_page_are_left_alone(self):
        s = snap(AUTH_TITLE, elements=[node(5, "Help")])
        assert evaluate_context("cua_click", {"element_id": 5, "target_name": "Help"}, s, goal="x") is None

    def test_notepad_titled_login_is_not_an_auth_screen(self):
        s = snap("login notes.txt - Notepad", class_name="Notepad")
        assert evaluate_context("cua_type_text", {"text": "hello"}, s, goal="x") is None

    def test_normal_page_is_unaffected(self):
        s = snap("YouTube Music - Microsoft Edge", elements=[node(1, "Search", role="edit")])
        assert evaluate_context("cua_click", {"element_id": 1, "target_name": "Search"}, s, goal="play a song") is None
        assert evaluate_context("cua_type_text", {"text": "signal"}, s, goal="play a song") is None


class TestUnsourcedContact:
    PAGE = "Slack - Microsoft Edge"

    def test_made_up_email_is_refused_with_a_fact(self):
        d = evaluate_context("cua_type_text", {"text": "user@example.com"}, snap(self.PAGE), goal="read slack")
        assert d is not None and d.kind == "refuse"
        assert d.message == "That email address was not given by the user or shown on screen."

    def test_email_dictated_in_the_goal_is_allowed(self):
        goal = "sign up with sam.k@mail.test"
        assert evaluate_context("cua_type_text", {"text": "sam.k@mail.test"}, snap(self.PAGE), goal=goal) is None

    def test_email_shown_on_screen_is_allowed(self):
        s = snap(self.PAGE, ocr=["Signed in as lee@corp.test"])
        assert evaluate_context("cua_type_text", {"text": "lee@corp.test"}, s, goal="x") is None

    def test_email_in_user_facts_is_allowed(self):
        d = evaluate_context(
            "cua_type_text", {"text": "kim@home.test"}, snap(self.PAGE), goal="x", facts_text="- my email: kim@home.test"
        )
        assert d is None

    def test_made_up_phone_is_refused_but_plain_numbers_and_dates_are_not(self):
        s = snap(self.PAGE)
        d = evaluate_context("cua_type_text", {"text": "+1 555 010 9999"}, s, goal="x")
        assert d is not None and "phone number" in d.message
        for text in ("1234567890", "2026-10-26", "3.14159265358979", "hello world"):
            assert evaluate_context("cua_type_text", {"text": text}, s, goal="x") is None


class TestTerminalCover:
    STEP5 = {
        "window": {"title": "Grace project overview"},
        "text": 'find / -iname "*qwen*" -type d 2>/dev/null',
        "replace": False,
        "submit": True,
    }

    def test_step_005_needs_confirmation(self):
        s = snap("✳ Grace project overview", class_name="CASCADIA_HOSTING_WINDOW_CLASS")
        d = evaluate_context("cua_type_text", self.STEP5, s, goal="find the qwen folder")
        assert d is not None and d.kind == "confirm"
        assert d.message == (
            "Run this in the terminal: 'find / -iname \"*qwen*\" -type d 2>/dev/null'? Say yes or no."
        )

    def test_terminal_found_by_process_name(self):
        s = snap("anything", class_name="Whatever")
        d = evaluate_context("cua_type_text", {"text": "dir"}, s, goal="x", process="pwsh.exe")
        assert d is not None and d.kind == "confirm"

    def test_vscode_integrated_terminal(self):
        s = snap(
            "main.py - Grace - Visual Studio Code",
            elements=[node(3, "Terminal 1, bash Input", role="edit", focused=True)],
        )
        assert evaluate_context("cua_type_text", {"text": "ls"}, s, goal="x", process="code.exe").kind == "confirm"

    def test_enter_and_paste_in_a_terminal_confirm_but_arrows_do_not(self):
        s = snap("cmd", class_name="ConsoleWindowClass")
        assert evaluate_context("cua_press_key", {"key": "Enter"}, s, goal="x").kind == "confirm"
        assert evaluate_context("cua_press_key", {"key": "ctrl+v"}, s, goal="x").kind == "confirm"
        assert evaluate_context("cua_press_key", {"key": "Up"}, s, goal="x") is None

    def test_typing_into_notepad_is_unaffected(self):
        s = snap("Untitled - Notepad", class_name="Notepad")
        assert evaluate_context("cua_type_text", {"text": "hello"}, s, goal="x", process="notepad.exe") is None


class TestTerminalCoverTargetWindow:
    # Cockpit 5 layer 1: even when the front window is not a terminal, naming a
    # terminal window as the step's target must still produce the confirm
    # decision, because the loop activates that window before typing.

    def test_plan_time_checks_named_terminal_window(self, monkeypatch):
        monkeypatch.setattr(
            "grace.agent.context_covers.window_process_name",
            lambda hwnd: "powershell.exe" if hwnd == 0x4321 else "",
        )
        s = snap("Untitled - Notepad", class_name="Notepad")
        params = {
            "window": {"title": "Windows PowerShell", "hwnd": 0x4321},
            "text": "rm -r ...",
            "submit": True,
        }
        d = evaluate_context("cua_type_text", params, s, goal="clear the drive")
        assert d is not None and d.kind == "confirm"
        assert d.message == "Run this in the terminal: 'rm -r ...'? Say yes or no."

    def test_plan_time_named_notepad_is_no_cover(self, monkeypatch):
        monkeypatch.setattr(
            "grace.agent.context_covers.window_process_name",
            lambda hwnd: "notepad.exe" if hwnd == 0x4322 else "",
        )
        s = snap("Untitled - Notepad", class_name="Notepad")
        params = {
            "window": {"title": "Untitled - Notepad", "hwnd": 0x4322},
            "text": "hello",
        }
        assert evaluate_context("cua_type_text", params, s, goal="x") is None

    def test_plan_time_window_title_without_hwnd_matches_live_window(self, monkeypatch):
        # `window_title` only (no hwnd): resolve against live windows by title.
        monkeypatch.setattr(
            "grace.agent.context_covers.window_process_name",
            lambda hwnd: "powershell.exe" if hwnd == 0x9999 else "",
        )
        monkeypatch.setattr(
            "win32gui.EnumWindows",
            lambda callback, results: results.append((0x9999, "Windows PowerShell")),
        )
        s = snap("Untitled - Notepad", class_name="Notepad")
        params = {"window_title": "Windows PowerShell", "text": "rm -r ...", "submit": True}
        d = evaluate_context("cua_type_text", params, s, goal="x")
        assert d is not None and d.kind == "confirm"


def _loop(snapshot, replies):
    gemma = AsyncMock()
    gemma.generate_text.side_effect = replies
    dispatcher = AsyncMock()
    dispatcher.execute.return_value = {"status": "ok", "result": {"ok": True, "sent": True, "verified": True}}
    perception = MagicMock()
    perception.capture_snapshot.return_value = snapshot
    loop = AgentLoop(
        gemma=gemma,
        dispatcher=dispatcher,
        perception=perception,
        grounder=AsyncMock(),
        fleet_config=build_gpu_single_preset(),
    )
    return loop, dispatcher


@pytest.mark.asyncio
async def test_loop_asks_the_user_on_the_exact_step_9_click():
    s = snap(AUTH_TITLE, elements=[node(9, "Use another account", role="listitem")])
    loop, dispatcher = _loop(
        s,
        ['{"thought": "pick", "action": "cua_click", "params": {"element_id": 9, "target_name": "Use another account"}, "is_completed": false, "user_update": "x"}'],
    )
    result = await loop.run("read the questions channel in slack")
    assert result["status"] == "needs_user_input"
    assert "wants you to sign in" in result["final_response"]
    assert not [c for c in dispatcher.execute.call_args_list if c[0][0].tool == "cua_click"]
    assert "wants you to sign in" in result["steps"][-1]["result"]["error"]


@pytest.mark.asyncio
async def test_loop_parks_terminal_typing_for_a_yes_then_runs_it():
    s = snap("✳ Grace project overview", class_name="CASCADIA_HOSTING_WINDOW_CLASS")
    loop, dispatcher = _loop(
        s,
        [
            '{"thought": "t", "action": "cua_type_text", "params": {"text": "ls -la", "submit": true}, "is_completed": false, "user_update": "x"}',
            '{"thought": "d", "action": "converse", "params": {"response": "ok"}, "is_completed": true, "final_response": "ok"}',
        ],
    )
    result = await loop.run("list files")
    assert result["status"] == "safety_confirmation_required"
    assert "Run this in the terminal: 'ls -la'? Say yes or no." == result["confirmation_prompt"]
    assert not dispatcher.execute.called
    await loop.resume_pending(True)
    assert [c[0][0].tool for c in dispatcher.execute.call_args_list][0] == "cua_type_text"


@pytest.mark.asyncio
async def test_loop_refuses_unsourced_email_as_a_fact_and_keeps_going():
    s = snap("Slack - Microsoft Edge", elements=[node(1, "Email", role="edit")])
    loop, dispatcher = _loop(
        s,
        [
            '{"thought": "t", "action": "cua_type_text", "params": {"element_id": 1, "text": "user@example.com"}, "is_completed": false, "user_update": "x"}',
            '{"thought": "d", "action": "converse", "params": {"response": "ok"}, "is_completed": true, "final_response": "ok"}',
        ],
    )
    result = await loop.run("sign up")
    assert not [c for c in dispatcher.execute.call_args_list if c[0][0].tool == "cua_type_text"]
    assert "was not given by the user" in result["steps"][0]["result"]["error"]
