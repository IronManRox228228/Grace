"""Test CUA bridge enhancements and parameter extractions."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.automation.computer_use import ComputerUse


class TestCuaEnhancements:
    def setup_method(self):
        self.cua = ComputerUse()
        self.cua.start()

    def test_press_key_mapping(self):
        """Verify Return, Control_L, and hotkeys map without throwing PyAutoGUI key errors."""
        res1 = self.cua.perform("press_key", {"key": "Return"})
        assert res1.get("ok") is True

        res2 = self.cua.perform("press_key", {"key": "Control_L+a"})
        assert res2.get("ok") is True

    def test_activate_heuristics(self):
        """Verify activate handles window dicts, titles, and hwnds safely."""
        res1 = self.cua.perform("activate", {"window": {"id": 999999, "title": "NonExistentWindow"}})
        assert "ok" in res1

        res2 = self.cua.perform("activate", {"app": "NonExistentAppTitle"})
        assert "ok" in res2

    def test_click_window_extraction(self):
        """Verify click extracts x/y and click_count properly."""
        res = self.cua.perform("click", {"window": {"x": 10, "y": 10}, "click_count": 2})
        assert res.get("ok") is True
        assert "2 time(s)" in res.get("message", "")
