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
