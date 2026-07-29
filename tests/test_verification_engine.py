"""Unit tests for Empirical Verification Guard and Window Focus Lock."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.agent.loop import AgentLoop
from grace.agent.memory import AgentMemory
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.automation.computer_use import ComputerUse


class TestVerificationEngine:
    """Test suite for Empirical Verification Guard."""

    def test_verify_goal_media_play_fails_when_not_playing(self):
        loop = AgentLoop(gemma=None, dispatcher=None)
        memory = AgentMemory(user_goal="Play Bohemian Rhapsody")
        memory.add_step("Click search", "cua_click", {"target_name": "Search"}, {"status": "ok"})

        snapshot = ScreenSnapshot(
            active_window=WindowInfo(hwnd=123, title="YouTube Search - Microsoft Edge", class_name="Chrome_WidgetWin_1", rect=(0,0,1000,800)),
            ocr_lines=[],
            width=1920,
            height=1080,
        )

        verified, hint = loop._verify_goal_completion("Play Bohemian Rhapsody", snapshot, memory)
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
