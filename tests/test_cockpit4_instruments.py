"""Cockpit 4 item 2: scroll position as a plain fact, off-screen elements reported not clicked."""

import pyautogui

from grace.automation import computer_use as cu_mod
from grace.automation.computer_use import ComputerUse, _scroll_position_fact


class TestScrollPositionFact:
    def test_at_the_bottom(self):
        assert (
            _scroll_position_fact((100.0, 0.0), (100.0, 0.0))
            == "Scroll position 100% -> 100% (at the bottom)"
        )

    def test_mid_page_has_no_edge_note(self):
        assert _scroll_position_fact((40.0, 0.0), (62.2, 0.0)) == "Scroll position 40% -> 62%"

    def test_top_and_horizontal(self):
        assert _scroll_position_fact((3.0, 0.0), (0.0, 0.0)).endswith("(at the top)")
        assert _scroll_position_fact((0.0, 10.0), (0.0, 100.0), vertical=False).endswith("(at the right edge)")

    def test_unavailable_or_not_scrollable_is_silent(self):
        assert _scroll_position_fact(None, (1.0, 1.0)) == ""
        assert _scroll_position_fact((1.0, 1.0), None) == ""
        assert _scroll_position_fact((-1.0, 0.0), (-1.0, 0.0)) == ""


class TestScrollReportsPosition:
    def _run(self, monkeypatch, readings, params):
        calls = []
        monkeypatch.setattr(pyautogui, "scroll", lambda *a, **k: calls.append((a, k)))
        monkeypatch.setattr(cu_mod, "_screen_state", lambda: None)
        monkeypatch.setattr(cu_mod, "_settled_state", lambda: None)
        monkeypatch.setattr(cu_mod, "_invalidate_graph", lambda: None)
        monkeypatch.setattr(ComputerUse, "_foreground_centre", staticmethod(lambda: (500, 400)))
        feed = iter(readings)
        monkeypatch.setattr(cu_mod, "_read_scroll_percent", lambda x, y: next(feed))
        result = ComputerUse()._scroll(params)
        assert calls, "the scroll itself must still be sent"
        return result

    def test_result_reports_before_and_after_and_bottom(self, monkeypatch):
        res = self._run(monkeypatch, [(100.0, 0.0), (100.0, 0.0)], {"scrollY": 500})
        assert res["ok"] is True
        assert "Scroll position 100% -> 100% (at the bottom)" in res["message"]

    def test_moving_scroll_reports_both_numbers(self, monkeypatch):
        res = self._run(monkeypatch, [(40.0, 0.0), (62.0, 0.0)], {"scrollY": 500})
        assert "Scroll position 40% -> 62%" in res["message"]
        assert "bottom" not in res["message"]

    def test_unavailable_pattern_is_omitted(self, monkeypatch):
        res = self._run(monkeypatch, [None, None], {"scrollY": 500})
        assert "Scroll position" not in res["message"]


class TestOffscreenElementIsReportedNotClicked:
    def test_minimised_window_search_box(self, monkeypatch):
        # runs/2026-10-26_03-28-19: WhatsApp minimised, search box at (-31750, -31901).
        clicked = []
        monkeypatch.setattr(cu_mod, "_monitor_rects", lambda: [(0, 0, 1920, 1080)])
        monkeypatch.setattr(
            "grace.automation.win32_driver.Win32Driver.click_at",
            staticmethod(lambda *a, **k: clicked.append(a) or True),
        )
        res = ComputerUse()._click_at(-31750, -31901, 1, "graph:edit", "Search or start new chat", element_id=7)
        assert res["ok"] is False and res["sent"] is False
        assert res["error"] == (
            "Element [7] 'Search or start new chat' is off-screen at (-31750, -31901); "
            "its window may be minimised."
        )
        assert clicked == []

    def test_on_screen_and_second_monitor_points_still_click(self, monkeypatch):
        clicked = []
        monkeypatch.setattr(cu_mod, "_monitor_rects", lambda: [(0, 0, 1920, 1080), (-1920, 0, 0, 1080)])
        monkeypatch.setattr(cu_mod, "_settled_state", lambda: None)
        monkeypatch.setattr(cu_mod, "_invalidate_graph", lambda: None)
        monkeypatch.setattr(cu_mod, "_wrong_window", lambda *a, **k: None)
        monkeypatch.setattr(
            "grace.automation.win32_driver.Win32Driver.click_at",
            staticmethod(lambda *a, **k: clicked.append(a) or True),
        )
        cu = ComputerUse()
        assert cu._click_at(500, 500, 1, "m", "a")["ok"] is True
        assert cu._click_at(-500, 500, 1, "m", "a")["ok"] is True
        assert len(clicked) == 2

    def test_unknown_monitors_never_block(self, monkeypatch):
        monkeypatch.setattr(cu_mod, "_monitor_rects", lambda: [])
        assert cu_mod._offscreen_fact(-99999, -99999, "x") is None
