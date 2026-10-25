"""Tests for refusing click on wrong window (real-desktop-3.md Item 4)."""

import itertools
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grace.agent.loop import AgentLoop
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.automation.computer_use import ComputerUse


def make_loop(responses, dispatch_result=None, **kwargs):
    gemma = AsyncMock()
    gemma.generate_text.side_effect = responses
    dispatcher = AsyncMock()
    dispatcher.execute.return_value = dispatch_result or {"status": "ok"}
    perception = MagicMock()
    screens = (
        ScreenSnapshot(
            active_window=WindowInfo(hwnd=1, title=f"Terminal [{i}]", class_name="X", rect=(0, 0, 800, 600)),
            ocr_lines=[],
            width=1920,
            height=1080,
            graph=MagicMock(actionable_count=10),
            ui_elements=[],
        )
        for i in itertools.count()
    )
    perception.capture_snapshot.side_effect = lambda *a, **k: next(screens)
    perception.capture_snapshot_async.return_value = next(screens)
    loop = AgentLoop(gemma=gemma, dispatcher=dispatcher, perception=perception, **kwargs)
    return loop, gemma, dispatcher


class TestClickWrongWindow:
    """Test suite for Item 4 of real-desktop-3.md."""

    def test_click_names_window_refuses_when_not_verified_foreground(self):
        """When a click names a window and it is not verified foreground after activation, refuse click."""
        cu = ComputerUse()

        # Foreground is Terminal, not YouTube Music
        with patch("grace.automation.computer_use._foreground_title", return_value="Terminal"), \
             patch("win32gui.GetForegroundWindow", return_value=999), \
             patch("win32gui.GetWindowText", return_value="Terminal"), \
             patch.object(cu, "_activate", return_value={
                 "ok": False, "sent": True, "verified": False, "action": "activate",
                 "evidence": "the foreground window is 'Terminal'",
             }) as mock_act, \
             patch("grace.automation.win32_driver.Win32Driver.click_at") as mock_click:

            params = {
                "x": 192,
                "y": 78,
                "window": {"title": "YouTube Music - Microsoft Edge", "id": 1234},
            }
            res = cu._click(params)

        assert res["ok"] is False
        assert res["status"] == "wrong_window"
        assert "not the verified foreground after activation" in res["error"]
        assert mock_act.call_count == 1, "Must attempt activation when named window is not foreground"
        assert mock_click.call_count == 0, "Must not click whatever is in front when window is unverified"

    def test_click_names_window_proceeds_when_verified_foreground(self):
        """When a click names a window and activation verifies foreground, click proceeds."""
        cu = ComputerUse()

        # Initially Terminal, but activation brings YouTube Music to front
        fg_titles = ["Terminal"] + ["YouTube Music - Microsoft Edge"] * 10
        mock_state = {
            "window": "YouTube Music - Microsoft Edge",
            "elements": 5,
            "focus_id": 1,
            "focus_name": "Search",
            "focus_value": "",
            "blind": False,
        }
        with patch("grace.automation.computer_use._foreground_title", side_effect=fg_titles), \
             patch("win32gui.GetForegroundWindow", return_value=1234), \
             patch("win32gui.GetWindowText", return_value="YouTube Music - Microsoft Edge"), \
             patch.object(cu, "_activate", return_value={
                 "ok": True, "sent": True, "verified": True, "action": "activate",
                 "evidence": "the foreground window is 'YouTube Music - Microsoft Edge'",
             }) as mock_act, \
             patch("grace.automation.win32_driver.Win32Driver.click_at", return_value=True) as mock_click, \
             patch("grace.automation.computer_use._screen_state", return_value=mock_state):

            params = {
                "x": 192,
                "y": 78,
                "window": {"title": "YouTube Music - Microsoft Edge", "id": 1234},
            }
            res = cu._click(params)

        assert res["ok"] is True
        assert mock_act.call_count == 1
        assert mock_click.call_count == 1

    def test_click_refuses_when_foreground_is_empty(self):
        """When foreground window is empty (transient 0), refuse click."""
        cu = ComputerUse()

        with patch("grace.automation.computer_use._foreground_title", return_value=""), \
             patch.object(cu, "_activate", return_value={
                 "ok": False, "sent": True, "verified": False, "action": "activate",
                 "evidence": "the foreground window is ''",
             }), \
             patch("grace.automation.win32_driver.Win32Driver.click_at") as mock_click:

            params = {
                "x": 192,
                "y": 78,
                "window": {"title": "YouTube Music - Microsoft Edge"},
            }
            res = cu._click(params)

        assert res["ok"] is False
        assert res["status"] == "wrong_window"
        assert mock_click.call_count == 0

    @pytest.mark.asyncio
    async def test_agent_loop_switches_focus_for_dict_window(self):
        """When planned step has dict window target differing from current window, AgentLoop activates it."""
        plan_json = '{"thought": "click bar", "action": "cua_click", "params": {"window": {"title": "Microsoft Edge", "id": 1234}, "target_name": "Address bar"}}'
        loop, gemma, dispatcher = make_loop([plan_json, '{"thought": "done", "action": "done", "params": {}}'])

        res = await loop.run("test goal")

        activate_calls = [
            call for call in dispatcher.execute.call_args_list
            if call.args and getattr(call.args[0], "tool", None) == "cua_activate"
        ]
        assert len(activate_calls) >= 1
        assert activate_calls[0].args[0].params["window"] == {"title": "Microsoft Edge", "id": 1234}
