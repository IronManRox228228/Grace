"""Tests for effect text, browser OCR, and clean ruled out list (cockpit-1 Commit 5).

1. Effect text never contains "CONFIRMED", "FAILED", or "NO VISIBLE EFFECT".
2. "Ruled out approaches" never contains private `_`-prefixed params (e.g. `_graph`), and never contains duplicate lines.
3. Browser OCR runs for windows containing page elements (frame=FRAME_PAGE), and renders up to 15 lines in reading order inside wrap_untrusted.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from grace.agent.loop import AgentLoop, _short_params
from grace.agent.memory import AgentMemory
from grace.agent.perception import (
    ScreenSnapshot,
    WindowInfo,
    OcrLine,
    PerceptionEngine,
)
from grace.agent.planner import PlannedStep, wrap_untrusted
from grace.perception.element_graph import ElementGraph, WindowRef
from grace.perception.elements import ElementNode, FRAME_PAGE, FRAME_CHROME


def test_effect_text_never_contains_verdict_words():
    """Effect text never contains CONFIRMED, FAILED, or NO VISIBLE EFFECT."""
    loop = AgentLoop(gemma=AsyncMock(), dispatcher=AsyncMock(), perception=MagicMock())
    step = PlannedStep(action="cua_click", params={"element_id": 1}, expect="Search box opens")
    snap = ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title="Test Window", class_name="App", rect=(0, 0, 100, 100)),
        ocr_lines=[], width=1920, height=1080, graph=None, ui_elements=None,
    )

    # 1. Verified True
    res_confirmed = {"status": "ok", "result": {"ok": True, "sent": True, "verified": True, "evidence": "Focus moved to [8] 'Search'"}}
    note_confirmed = loop._expectation_note(step, res_confirmed, snap)
    assert "CONFIRMED" not in note_confirmed
    assert "Focus moved to [8] 'Search'" in note_confirmed

    # 2. Failed / error
    res_failed = {"status": "error", "error": "Element not found"}
    note_failed = loop._expectation_note(step, res_failed, snap)
    assert "FAILED" not in note_failed
    assert "Tool error: Element not found" in note_failed

    # 3. Repeated / screen unchanged
    note_repeated = loop._expectation_note(step, res_confirmed, snap, repeat_count=2)
    assert "NO VISIBLE EFFECT" not in note_repeated
    assert "The screen has not changed" not in note_repeated
    assert "You have now done this exact action 2 times on this same screen." in note_repeated


def test_ruled_out_approaches_has_no_private_keys_and_no_duplicates():
    """Ruled out approaches contains no _graph and no duplicate lines."""
    # Check _short_params ignores _-prefixed keys
    params = {
        "element_id": 39,
        "target_name": "Signal",
        "_graph": "<ElementGraph object at 0x123>",
        "_rendered_element_ids": {39},
        "_id_won_over_xy": True,
        "window": {"title": "YouTube Music"},
    }
    short = _short_params(params)
    assert "_graph" not in short
    assert "_rendered_element_ids" not in short
    assert "_id_won_over_xy" not in short
    assert "window" not in short
    assert "element_id" in short
    assert "target_name" in short

    # Check memory deduplication
    mem = AgentMemory("test goal")
    mem.rule_out("cua_click with {'element_id': 39}", "nothing changed")
    mem.rule_out("cua_click with {'element_id': 39}", "nothing changed")
    mem.rule_out("cua_click with {'element_id': 40}", "element not found")

    md = mem.format_established_ruled_out_markdown()
    lines = md.splitlines()
    ruled_out_lines = [l for l in lines if l.startswith("- cua_click with {'element_id': 39}")]
    assert len(ruled_out_lines) == 1, f"Expected 1 entry, got: {ruled_out_lines}"
    assert "_graph" not in md


def test_browser_ocr_triggered_and_rendered_in_reading_order():
    """Browser windows (with frame=FRAME_PAGE elements) always trigger OCR and wrap top 15 in untrusted data."""
    pe = PerceptionEngine(run_ocr=True)
    pe._winrt_ocr_available = True

    # Page elements present -> is_browser True
    el = ElementNode(
        id=1, role="button", name="Play", rect=(10, 10, 50, 50),
        center=(30, 30), frame=FRAME_PAGE,
    )
    graph = ElementGraph(elements=[el], window=WindowRef(hwnd=1, title="YouTube Music", class_name="Chrome_WidgetWin_1"))

    ocr_sample = [
        OcrLine(text="Line 2", bounding_box=(100, 200, 50, 20)),
        OcrLine(text="Line 1", bounding_box=(100, 50, 50, 20)),
        OcrLine(text="Line 3", bounding_box=(100, 350, 50, 20)),
    ]

    with patch.object(pe, "_perform_ocr", return_value=ocr_sample) as mock_ocr, \
         patch("grace.agent.perception._min_actionable", return_value=0), \
         patch("grace.agent.perception.PerceptionEngine.get_graph_builder") as mock_gb, \
         patch("win32gui.GetForegroundWindow", return_value=1), \
         patch("win32gui.GetWindowText", return_value="YouTube Music"), \
         patch("win32gui.GetClassName", return_value="Chrome_WidgetWin_1"), \
         patch("win32gui.GetWindowRect", return_value=(0, 0, 1920, 1080)):

        mock_gb.return_value.build.return_value = graph
        pe._take_screenshot = lambda: (1920, 1080, b"fake-png-bytes", 1280, 720)

        # Capture snapshot
        snap = pe.capture_snapshot()
        assert mock_ocr.call_count == 1
        assert len(snap.ocr_lines) == 3

        # Check rendering in reading order (Line 1 before Line 2 before Line 3) and wrap_untrusted
        md = snap.to_markdown()
        assert "Line 1" in md
        assert "Line 2" in md
        assert "Line 3" in md
        ocr_section = md[md.index("### Screen Text (OCR Output):"):]
        assert ocr_section.index("Line 1") < ocr_section.index("Line 2") < ocr_section.index("Line 3")
        # Double-wrap removed in Cockpit 2: outer prompt already wraps What is on screen
        assert "<<<UNTRUSTED_DATA_START" not in md
