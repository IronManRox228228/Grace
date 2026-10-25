"""Tests for browser URL extraction and reporting (cockpit-2 Commit 4).

- Browser URL read from UIA Edit with frame=chrome.
- Page URL: <url> appears under ### Active window when present, absent otherwise.
- URL before: ... / URL now: ... reported as plain fact in result section.
- Omitted silently if it cannot be read.
- _screen_state never claims a URL it did not read.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest

from grace.agent.loop import AgentLoop
from grace.agent.perception import (
    ScreenSnapshot,
    WindowInfo,
    extract_browser_url,
)
from grace.agent.planner import PlannedStep, Planner
from grace.automation.computer_use import _change_verdict
from grace.perception.elements import ElementNode, FRAME_CHROME, FRAME_PAGE, FRAME_APP
from grace.perception.element_graph import ElementGraph, WindowRef


def test_extract_browser_url_reads_address_bar():
    addr = ElementNode(
        id=0, role="edit", name="Address and search bar",
        rect=(100, 50, 800, 90), center=(450, 70),
        frame=FRAME_CHROME, value="https://music.youtube.com/search?q=signal",
    )
    btn = ElementNode(
        id=1, role="button", name="Search",
        rect=(810, 50, 850, 90), center=(830, 70),
        frame=FRAME_CHROME,
    )
    graph = ElementGraph(elements=[addr, btn], window=WindowRef(hwnd=1, title="YouTube Music - Edge"))
    snap = ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title="YouTube Music - Edge", class_name="Chrome_WidgetWin_1", rect=(0, 0, 1920, 1080)),
        ocr_lines=[], width=1920, height=1080, graph=graph,
    )

    url = extract_browser_url(snap)
    assert url == "https://music.youtube.com/search?q=signal"


def test_extract_browser_url_returns_none_when_no_address_bar():
    app_btn = ElementNode(
        id=1, role="button", name="Close",
        rect=(10, 10, 30, 30), center=(20, 20),
        frame=FRAME_APP,
    )
    page_edit = ElementNode(
        id=2, role="edit", name="Search",
        rect=(100, 100, 300, 140), center=(200, 120),
        frame=FRAME_PAGE, value="signal",
    )
    graph = ElementGraph(elements=[app_btn, page_edit], window=WindowRef(hwnd=2, title="Calculator"))
    snap = ScreenSnapshot(
        active_window=WindowInfo(hwnd=2, title="Calculator", class_name="CalcFrame", rect=(0, 0, 800, 600)),
        ocr_lines=[], width=800, height=600, graph=graph,
    )

    assert extract_browser_url(snap) is None


def test_page_url_in_prompt_active_window():
    planner = Planner(llm=MagicMock())

    # With page_url
    prompt_with_url = planner.build_prompt(
        goal="find song",
        elements_prompt="elements",
        window_title="YouTube Music - Edge",
        page_url="https://music.youtube.com",
    )
    assert "### Active window\nYouTube Music - Edge\nPage URL: https://music.youtube.com" in prompt_with_url

    # Without page_url
    prompt_no_url = planner.build_prompt(
        goal="find song",
        elements_prompt="elements",
        window_title="YouTube Music - Edge",
        page_url=None,
    )
    assert "Page URL:" not in prompt_no_url
    assert "### Active window\nYouTube Music - Edge" in prompt_no_url


def test_url_before_and_now_in_expectation_note():
    loop = AgentLoop(gemma=AsyncMock(), dispatcher=AsyncMock(), perception=MagicMock())
    step = PlannedStep(action="cua_click", params={"element_id": 1}, expect="page navigates")
    snap = ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title="YouTube - Edge", class_name="Chrome_WidgetWin_1", rect=(0, 0, 100, 100)),
        ocr_lines=[], width=1920, height=1080, graph=None,
    )

    # Both URLs present: fact is reported
    note = loop._expectation_note(
        step, {"status": "ok"}, snap,
        url_before="https://music.youtube.com",
        url_now="https://music.youtube.com/search?q=signal",
    )
    assert "URL before: https://music.youtube.com / URL now: https://music.youtube.com/search?q=signal (note: some sites change content without changing the URL)" in note

    # url_before absent: silently omitted
    note_silent = loop._expectation_note(
        step, {"status": "ok"}, snap,
        url_before=None,
        url_now="https://music.youtube.com",
    )
    assert "URL before:" not in note_silent


def test_screen_state_and_change_verdict_url_handling():
    before = {"window": "Edge", "focus_id": 1, "focus_name": "", "focus_value": "", "elements": 10, "blind": False}
    after = {"window": "Edge", "focus_id": 1, "focus_name": "", "focus_value": "", "elements": 10, "blind": False}

    # When neither has a URL: never claims URL unchanged
    changed, verdict = _change_verdict(before, after)
    assert "Page URL" not in verdict
    assert "URL unchanged" not in verdict
    assert "Window title unchanged" in verdict

    # When URL changes
    before_url = {**before, "url": "https://example.com/1"}
    after_url = {**after, "url": "https://example.com/2"}
    changed2, verdict2 = _change_verdict(before_url, after_url)
    assert "URL changed to 'https://example.com/2'" in verdict2
