"""Unit tests for Empirical Verification Guard and Window Focus Lock."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.agent.loop import AgentLoop
from grace.agent.memory import AgentMemory
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.automation.computer_use import ComputerUse


def _readable_graph():
    """A graph with enough actionable controls to count as readable.

    Without one, every snapshot here is `blind`, and the guard deliberately
    declines to contradict a claim it has no way to check - which would make
    the rejection tests below pass for the wrong reason.
    """
    from grace.perception.element_graph import ElementGraph, WindowRef
    from grace.perception.elements import ElementNode

    return ElementGraph(
        elements=[
            ElementNode(
                id=i, role="button", name=f"control-{i}",
                rect=(0, i * 30, 100, i * 30 + 20), center=(50, i * 30 + 10),
            )
            for i in range(1, 9)
        ],
        window=WindowRef(hwnd=123, title="test"),
    )


class TestVerificationEngine:
    """Test suite for the completion guard in AgentLoop."""

    def test_verify_goal_accepts_a_successful_interaction(self):
        # Renamed. This was called `..._fails_when_not_playing` and asserted
        # `verified is True`, i.e. it pinned the exact opposite of what its name
        # claimed, and would have gone on passing if the guard were deleted.
        #
        # What it actually establishes is the guard's deliberate limit: it does
        # not understand the goal, so a step that ran and reported success is
        # accepted even though nothing here is playing. Blocking on "the title
        # doesn't look like music" would be guessing.
        loop = AgentLoop(gemma=None, dispatcher=None)
        memory = AgentMemory(user_goal="Play Bohemian Rhapsody")
        memory.add_step("Click search", "cua_click", {"target_name": "Search"}, {"status": "ok"})

        snapshot = ScreenSnapshot(
            active_window=WindowInfo(hwnd=123, title="YouTube Search - Microsoft Edge", class_name="Chrome_WidgetWin_1", rect=(0,0,1000,800)),
            ocr_lines=[],
            width=1920,
            height=1080,
            graph=_readable_graph(),
        )

        verified, hint = loop._verify_goal_completion("Play Bohemian Rhapsody", snapshot, memory)
        assert verified is True

    def test_verify_goal_rejects_completion_after_a_failed_action(self):
        # The case the loop actually hit: a step came back `wrong_focus`, the
        # planner announced the goal complete anyway, and nothing disagreed.
        loop = AgentLoop(gemma=None, dispatcher=None)
        memory = AgentMemory(user_goal="Search for coordination compounds")
        memory.add_step(
            "Type the query", "cua_type_text", {"text": "coordination compounds"},
            {"status": "ok", "result": {"ok": False, "status": "wrong_focus",
                                        "error": "Focus is on 'Chat list', not 'Search'"}},
        )

        snapshot = ScreenSnapshot(
            active_window=WindowInfo(hwnd=123, title="WhatsApp", class_name="Chrome_WidgetWin_1", rect=(0,0,1000,800)),
            ocr_lines=[],
            width=1920,
            height=1080,
            graph=_readable_graph(),
        )

        verified, hint = loop._verify_goal_completion("Search for coordination compounds", snapshot, memory)
        assert verified is False
        assert "wrong_focus" in hint or "Focus is on" in hint

    def test_verify_goal_does_not_reject_on_a_window_it_cannot_read(self):
        # A blind window contradicts nothing, and manufacturing a verdict from
        # an absence of evidence is what stopped a run that was succeeding.
        loop = AgentLoop(gemma=None, dispatcher=None)
        memory = AgentMemory(user_goal="Open the chemistry group")
        memory.add_step(
            "Click the group", "cua_click", {"element_id": 3},
            {"status": "ok", "result": {"ok": False, "error": "element_not_found"}},
        )

        snapshot = ScreenSnapshot(
            active_window=WindowInfo(hwnd=123, title="WhatsApp", class_name="Chrome_WidgetWin_1", rect=(0,0,1000,800)),
            ocr_lines=[],
            width=1920,
            height=1080,
        )
        assert snapshot.is_blind

        verified, _ = loop._verify_goal_completion("Open the chemistry group", snapshot, memory)
        assert verified is True

    def test_verify_goal_media_play_passes_when_playing(self):
        loop = AgentLoop(gemma=None, dispatcher=None)
        memory = AgentMemory(user_goal="Play Bohemian Rhapsody")
        memory.add_step("Click play", "cua_click", {"target_name": "Play"}, {"status": "ok"})

        snapshot = ScreenSnapshot(
            active_window=WindowInfo(hwnd=123, title="Queen – Bohemian Rhapsody (Official Video) - YouTube - Microsoft Edge", class_name="Chrome_WidgetWin_1", rect=(0,0,1000,800)),
            ocr_lines=[],
            width=1920,
            height=1080,
        )

        verified, hint = loop._verify_goal_completion("Play Bohemian Rhapsody", snapshot, memory)
        assert verified is True
        assert hint is None

    def test_verify_goal_media_play_fails_on_duration_label_3_42(self):
        from grace.agent.perception import OcrLine
        loop = AgentLoop(gemma=None, dispatcher=None)
        memory = AgentMemory(user_goal="Play Signal by Home")

        snapshot = ScreenSnapshot(
            active_window=WindowInfo(hwnd=123, title="https://music.youtube.com/search?q=Signal+by+Home", class_name="Chrome_WidgetWin_1", rect=(0,0,1000,800)),
            ocr_lines=[OcrLine(text="Signal", bounding_box=(0,0,10,10)), OcrLine(text="Song • Home • 3:42", bounding_box=(0,0,10,10)), OcrLine(text="Play", bounding_box=(0,0,10,10))],
            width=1920,
            height=1080,
        )

        verified, hint = loop._verify_goal_completion("Play Signal by Home", snapshot, memory)
        assert verified is False
        assert "Goal observation check" in hint

    def test_ensure_foreground_window_does_not_raise(self):
        cu = ComputerUse()
        cu._ensure_foreground_window()
        assert cu is not None
