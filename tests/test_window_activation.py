"""Tests for robust window activation (real-desktop-3.md Item 2)."""

import time
from unittest.mock import MagicMock, call, patch

import pytest

from grace.automation.computer_use import ComputerUse


class TestWindowActivation:
    """Test suite for Item 2 of real-desktop-3.md."""

    def test_transient_zero_re_polls_and_resolves(self):
        """GetForegroundWindow returning 0 is treated as transient switch and resolves within 500ms."""
        cu = ComputerUse()
        target_hwnd = 12345

        # Mock win32gui and user32
        with patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.GetWindowText", return_value="Target App"), \
             patch("win32process.GetWindowThreadProcessId", return_value=(999, 1234)), \
             patch("win32api.GetCurrentThreadId", return_value=888), \
             patch("win32gui.SetForegroundWindow") as mock_set_fg, \
             patch("win32gui.BringWindowToTop") as mock_bring_top:

            # GetForegroundWindow returns 0 twice, then target_hwnd
            fg_returns = [0, 0, target_hwnd]

            def mock_get_fg():
                if fg_returns:
                    return fg_returns.pop(0)
                return target_hwnd

            with patch("win32gui.GetForegroundWindow", side_effect=mock_get_fg):
                # Using technique 2 (AttachThreadInput)
                with patch("ctypes.windll.user32.AttachThreadInput", return_value=1):
                    res = cu._activate({"hwnd": target_hwnd, "title": "Target App"})

            assert res["ok"] is True, f"Expected ok: True, got: {res}"
            assert res["verified"] is True
            assert res["evidence"] == "the foreground window is 'Target App'"

    def test_activation_fallback_order_2_3_5_4(self):
        """Techniques are tried in order 2 -> 3 -> 5 -> 4, stopping at the first verified success."""
        cu = ComputerUse()
        target_hwnd = 54321

        call_log = []

        def fake_attach_thread_input(*args):
            call_log.append("technique_2_attach_thread")
            return 1

        def fake_alt_key(*args):
            call_log.append("technique_3_alt_key")

        def fake_switch_to_this_window(*args):
            call_log.append("technique_5_switch_to_this_window")

        def fake_show_window(*args):
            call_log.append("technique_4_show_window")

        # Let technique 2 and 3 fail (fg stays 99999), technique 5 succeed (fg becomes target_hwnd)
        fg_state = {"current": 99999}

        def mock_get_fg():
            return fg_state["current"]

        def on_switch_to_this_window(hwnd, alt_tab):
            fake_switch_to_this_window(hwnd, alt_tab)
            fg_state["current"] = target_hwnd  # Succeed on technique 5!

        with patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.GetWindowText", return_value="Target App"), \
             patch("win32process.GetWindowThreadProcessId", return_value=(999, 1234)), \
             patch("win32api.GetCurrentThreadId", return_value=888), \
             patch("win32gui.SetForegroundWindow"), \
             patch("win32gui.BringWindowToTop"), \
             patch("win32gui.GetForegroundWindow", side_effect=mock_get_fg), \
             patch("ctypes.windll.user32.AttachThreadInput", side_effect=fake_attach_thread_input), \
             patch("ctypes.windll.user32.keybd_event", side_effect=fake_alt_key), \
             patch("ctypes.windll.user32.SwitchToThisWindow", side_effect=on_switch_to_this_window), \
             patch("win32gui.ShowWindow", side_effect=fake_show_window):

            res = cu._activate({"hwnd": target_hwnd, "title": "Target App"})

        assert res["ok"] is True
        assert res["verified"] is True
        assert "technique_2_attach_thread" in call_log
        assert "technique_3_alt_key" in call_log
        assert "technique_5_switch_to_this_window" in call_log
        assert "technique_4_show_window" not in call_log, "Should have stopped at technique 5 upon success"

    def test_unverified_activation_returns_ok_false(self):
        """When all techniques fail to bring target to foreground, ok: False is returned."""
        cu = ComputerUse()
        target_hwnd = 11111
        other_hwnd = 22222

        with patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.GetWindowText", side_effect=lambda h: "Other Window" if h == other_hwnd else "Target App"), \
             patch("win32process.GetWindowThreadProcessId", return_value=(0, 999)), \
             patch("win32api.GetCurrentThreadId", return_value=888), \
             patch("win32gui.SetForegroundWindow"), \
             patch("win32gui.BringWindowToTop"), \
             patch("win32gui.GetForegroundWindow", return_value=other_hwnd), \
             patch("ctypes.windll.user32.AttachThreadInput", return_value=1), \
             patch("ctypes.windll.user32.keybd_event"), \
             patch("ctypes.windll.user32.SwitchToThisWindow"), \
             patch("win32gui.ShowWindow"):

            res = cu._activate({"hwnd": target_hwnd, "title": "Target App"})

        # Must return ok: False (never ok: True with verified: False)
        assert res["ok"] is False, f"Expected ok: False for unverified activation, got: {res}"
        assert res["verified"] is False

    def test_hallucinated_window_id_ignored_and_matches_by_title(self):
        """Item 3: Hallucinated window id is ignored if unknown; matches by normalised title."""
        cu = ComputerUse()
        fake_id = 1234  # Hallucinated
        real_id = 9999
        real_title = "YouTube Music - Personal - Microsoft Edge"

        mock_windows = [
            {"id": 100, "title": "Terminal", "pid": 1},
            {"id": real_id, "title": real_title, "pid": 2},
        ]
        cu._list_windows = MagicMock(return_value={"ok": True, "windows": mock_windows})

        # Mock _activate_hwnd to record what hwnd was actually activated
        activated_hwnds = []
        cu._activate_hwnd = MagicMock(side_effect=lambda h, t: activated_hwnds.append(h) or {
            "ok": True, "sent": True, "verified": True, "action": "activate", "hwnd": h, "message": f"Activated {t}"
        })

        # Pass window dict with hallucinated id and valid title
        res = cu._activate({"window": {"title": "YouTube Music", "id": fake_id}})
        assert res["ok"] is True
        assert activated_hwnds == [real_id], f"Expected real hwnd {real_id} to be activated, got {activated_hwnds}"

    def test_hallucinated_window_id_and_unknown_title_returns_real_windows_error(self):
        """Item 3: When neither matches, error lists real windows (title + id, capped at 8)."""
        cu = ComputerUse()
        fake_id = 1234
        mock_windows = [
            {"id": i, "title": f"App {i}", "pid": i}
            for i in range(1, 15)  # 14 windows
        ]
        cu._list_windows = MagicMock(return_value={"ok": True, "windows": mock_windows})

        res = cu._activate({"window": {"title": "GhostApp", "id": fake_id}})
        assert res["ok"] is False
        assert "error" in res
        error_msg = res["error"]
        # Error must list real windows with title plus id
        assert "'App 1' (id: 1)" in error_msg
        assert "'App 8' (id: 8)" in error_msg
        # Capped at 8
        assert "'App 9' (id: 9)" not in error_msg
