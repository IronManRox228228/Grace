"""Tests for media_control fast-path tool.

Asserts:
- Dispatcher dispatches VK_MEDIA_PLAY_PAUSE (0xB3), VK_MEDIA_NEXT_TRACK (0xB0),
  and VK_MEDIA_PREV_TRACK (0xB1) with mocked key sender. Never sends real keys.
- Spoken responses: 'Done.' for play_pause, 'Next track.' for next, 'Previous track.' for previous.
- Safety: media_control requires no user confirmation.
- Router: media_control routes to FAST_PATH for atomic requests, escalates on multi-step.
- Schema: registered in tools and intent prompt.
"""

from unittest.mock import MagicMock, patch
import pytest

from grace.agent.safety import SafetyGuard
from grace.intent.parser import Intent, VALID_TOOLS
from grace.intent.router import CapabilityRouter, TaskComplexity
from grace.intent.tools import format_tools_compact, format_tools_for_prompt
from grace.tools.dispatcher import (
    Dispatcher,
    VK_MEDIA_NEXT_TRACK,
    VK_MEDIA_PREV_TRACK,
    VK_MEDIA_PLAY_PAUSE,
)


class TestMediaControlDispatcher:
    @pytest.mark.asyncio
    @patch("grace.tools.dispatcher._send_media_key")
    async def test_play_pause_action_sends_0xb3_and_says_done(self, mock_send):
        dispatcher = Dispatcher(computer_use=None)
        res = await dispatcher.execute(Intent(tool="media_control", params={"action": "play_pause"}))
        assert res["status"] == "ok"
        assert res["action"] == "media_control"
        assert res["text"] == "Done."
        mock_send.assert_called_once_with(VK_MEDIA_PLAY_PAUSE)
        assert VK_MEDIA_PLAY_PAUSE == 0xB3

    @pytest.mark.asyncio
    @patch("grace.tools.dispatcher._send_media_key")
    async def test_default_action_is_play_pause(self, mock_send):
        dispatcher = Dispatcher(computer_use=None)
        res = await dispatcher.execute(Intent(tool="media_control", params={}))
        assert res["status"] == "ok"
        assert res["text"] == "Done."
        mock_send.assert_called_once_with(0xB3)

    @pytest.mark.asyncio
    @patch("grace.tools.dispatcher._send_media_key")
    async def test_next_action_sends_0xb0_and_says_next_track(self, mock_send):
        dispatcher = Dispatcher(computer_use=None)
        res = await dispatcher.execute(Intent(tool="media_control", params={"action": "next"}))
        assert res["status"] == "ok"
        assert res["text"] == "Next track."
        mock_send.assert_called_once_with(VK_MEDIA_NEXT_TRACK)
        assert VK_MEDIA_NEXT_TRACK == 0xB0

    @pytest.mark.asyncio
    @patch("grace.tools.dispatcher._send_media_key")
    async def test_previous_action_sends_0xb1_and_says_previous_track(self, mock_send):
        dispatcher = Dispatcher(computer_use=None)
        res = await dispatcher.execute(Intent(tool="media_control", params={"action": "previous"}))
        assert res["status"] == "ok"
        assert res["text"] == "Previous track."
        mock_send.assert_called_once_with(VK_MEDIA_PREV_TRACK)
        assert VK_MEDIA_PREV_TRACK == 0xB1

    @pytest.mark.asyncio
    @patch("grace.tools.dispatcher._send_media_key")
    async def test_synonyms_pause_resume_prev(self, mock_send):
        dispatcher = Dispatcher(computer_use=None)

        res_pause = await dispatcher.execute(Intent(tool="media_control", params={"action": "pause"}))
        assert res_pause["text"] == "Done."
        mock_send.assert_called_with(0xB3)

        res_resume = await dispatcher.execute(Intent(tool="media_control", params={"action": "resume"}))
        assert res_resume["text"] == "Done."
        mock_send.assert_called_with(0xB3)

        res_prev = await dispatcher.execute(Intent(tool="media_control", params={"action": "prev"}))
        assert res_prev["text"] == "Previous track."
        mock_send.assert_called_with(0xB1)

    @pytest.mark.asyncio
    @patch("grace.tools.dispatcher._send_media_key")
    async def test_unknown_action_returns_error(self, mock_send):
        dispatcher = Dispatcher(computer_use=None)
        res = await dispatcher.execute(Intent(tool="media_control", params={"action": "fast_forward"}))
        assert res["status"] == "error"
        assert "Unknown media action" in res["error"]
        mock_send.assert_not_called()


class TestMediaControlSafetyAndRouting:
    def test_media_control_is_safe_without_confirmation(self):
        is_safe, prompt = SafetyGuard.evaluate("media_control", {"action": "play_pause"})
        assert is_safe is True
        assert prompt is None

    def test_media_control_fast_path_routing(self):
        for text in ("pause the song", "pause the music", "next track", "previous song", "play"):
            intent = Intent(tool="media_control", params={"action": "play_pause"})
            assert CapabilityRouter.classify(text, intent) == TaskComplexity.FAST_PATH

    def test_media_control_chained_request_escalates(self):
        intent = Intent(tool="media_control", params={"action": "play_pause"})
        assert CapabilityRouter.classify(
            "pause the music and check my email", intent
        ) == TaskComplexity.AGENTIC_GOAL

    def test_media_control_registered_in_schema_and_prompts(self):
        assert "media_control" in VALID_TOOLS
        compact = format_tools_compact()
        assert "media_control" in compact
        assert "pause/play/resume the song/music/video" in compact
        assert "next song" in compact
        assert "previous track" in compact

        rich = format_tools_for_prompt()
        assert "### `media_control`" in rich


class TestWin32DriverSendVirtualKey:
    @patch("ctypes.windll.user32.SendInput", return_value=2)
    def test_send_virtual_key_sends_down_and_up_events(self, mock_send_input):
        from grace.automation.win32_driver import Win32Driver
        ok = Win32Driver.send_virtual_key(0xB3)
        assert ok is True
        assert mock_send_input.call_count == 1
        # 2 events sent (down + up)
        assert mock_send_input.call_args[0][0] == 2
