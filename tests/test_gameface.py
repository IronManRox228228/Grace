"""Unit tests for Project Gameface coexistence guard (Task 10, PLAN §7)."""

import time
from unittest.mock import MagicMock, patch
import pytest

from grace.automation.gameface import GamefaceCoexistence


class TestGamefaceCoexistence:
    def test_process_detection(self):
        coex = GamefaceCoexistence()

        # Mock psutil with no gameface process
        with patch("psutil.process_iter") as mock_iter:
            proc1 = MagicMock()
            proc1.info = {"name": "chrome.exe"}
            proc2 = MagicMock()
            proc2.info = {"name": "notepad.exe"}
            mock_iter.return_value = [proc1, proc2]

            assert coex.is_running(force_refresh=True) is False

        # Mock psutil with gameface running
        with patch("psutil.process_iter") as mock_iter:
            proc1 = MagicMock()
            proc1.info = {"name": "project_gameface.exe"}
            mock_iter.return_value = [proc1]

            assert coex.is_running(force_refresh=True) is True

    def test_cache_ttl(self):
        coex = GamefaceCoexistence(cache_ttl_seconds=10.0)
        coex._last_check_time = time.perf_counter()
        coex._is_running_cached = True

        # Should return cached value without querying psutil
        with patch("psutil.process_iter") as mock_iter:
            assert coex.is_running() is True
            mock_iter.assert_not_called()

    def test_quiet_window_not_running(self):
        coex = GamefaceCoexistence()
        with patch.object(coex, "is_running", return_value=False):
            # Not running -> safe to click immediately
            assert coex.wait_for_quiet_window(timeout_seconds=0.1) is True

    def test_quiet_window_settles(self):
        coex = GamefaceCoexistence()
        with patch.object(coex, "is_running", return_value=True):
            # Simulate stationary cursor at (100, 100)
            with patch("grace.automation.gameface._get_cursor_pos", return_value=(100, 100)):
                assert coex.wait_for_quiet_window(timeout_seconds=0.2, quiet_window_ms=50.0) is True

    def test_quiet_window_moving_times_out(self):
        coex = GamefaceCoexistence()
        with patch.object(coex, "is_running", return_value=True):
            # Simulate constantly moving cursor
            positions = [(i, i) for i in range(100)]
            pos_iter = iter(positions)

            def _moving_pos():
                return next(pos_iter, (999, 999))

            with patch("grace.automation.gameface._get_cursor_pos", side_effect=_moving_pos):
                # Should detect motion contention and return False on timeout
                res = coex.wait_for_quiet_window(timeout_seconds=0.08, quiet_window_ms=50.0)
                assert res is False
