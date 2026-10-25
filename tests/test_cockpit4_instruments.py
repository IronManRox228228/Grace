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


# -- item 3a: cua_drag with coordinates and a path ---------------------------

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from grace.agent.loop import AgentLoop
from grace.agent.planner import parse_planned_step
from grace.automation.win32_driver import Win32Driver
from grace.fleet import build_gpu_single_preset
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.intent.tools import ALL_TOOLS


class _Mouse:
    """Records the SendInput-level calls; nothing reaches the desktop."""

    def __init__(self, monkeypatch):
        self.events = []
        monkeypatch.setattr(Win32Driver, "mouse_down", classmethod(lambda c, x, y: self.events.append(("down", x, y)) or True))
        monkeypatch.setattr(Win32Driver, "mouse_move", classmethod(lambda c, x, y: self.events.append(("move", x, y)) or True))
        monkeypatch.setattr(Win32Driver, "mouse_up", classmethod(lambda c, x, y: self.events.append(("up", x, y)) or True))
        monkeypatch.setattr(Win32Driver, "_is_foreground", classmethod(lambda c, h: True))
        monkeypatch.setattr("win32gui.WindowFromPoint", lambda p: 0)
        monkeypatch.setattr("ctypes.windll.user32.SetCursorPos", lambda *a: 1, raising=False)
        monkeypatch.setattr("time.sleep", lambda s: None)

    @property
    def kinds(self):
        return [e[0] for e in self.events]


@pytest.fixture
def quiet_desktop(monkeypatch):
    monkeypatch.setattr(cu_mod, "_monitor_rects", lambda: [(0, 0, 1920, 1080)])
    monkeypatch.setattr(cu_mod, "_screen_state", lambda: None)
    monkeypatch.setattr(cu_mod, "_settled_state", lambda: None)
    monkeypatch.setattr(cu_mod, "_invalidate_graph", lambda: None)


class TestDragExecutor:
    def test_coordinate_drag_presses_moves_and_releases(self, monkeypatch, quiet_desktop):
        mouse = _Mouse(monkeypatch)
        res = ComputerUse()._drag({"x": 400, "y": 300, "to_x": 900, "to_y": 600})
        assert res["ok"] is True
        assert mouse.kinds[0] == "down" and mouse.kinds[-1] == "up"
        assert "move" in mouse.kinds
        assert mouse.events[0][1:] == (400, 300) and mouse.events[-1][1:] == (900, 600)

    def test_five_point_path_makes_five_or_more_moves(self, monkeypatch, quiet_desktop):
        mouse = _Mouse(monkeypatch)
        path = [[100, 100], [300, 200], [500, 100], [700, 200], [900, 100]]
        res = ComputerUse()._drag({"path": path})
        assert res["ok"] is True
        assert mouse.kinds.count("move") >= 5
        assert mouse.kinds.count("down") == 1 and mouse.kinds.count("up") == 1
        assert "path of 5 points" in res["message"]

    def test_element_id_drag_is_unchanged(self, monkeypatch, quiet_desktop):
        mouse = _Mouse(monkeypatch)
        calls = []
        monkeypatch.setattr(pyautogui, "moveTo", lambda *a, **k: calls.append(("moveTo", a)))
        monkeypatch.setattr(pyautogui, "drag", lambda *a, **k: calls.append(("drag", a)))
        elements = {1: types_ns((10, 20)), 2: types_ns((310, 220))}
        monkeypatch.setattr(ComputerUse, "_graph_element", staticmethod(lambda i: elements.get(i)))
        res = ComputerUse()._drag({"from_element_id": 1, "to_element_id": 2})
        assert res["ok"] is True
        assert mouse.events == []
        assert calls == [("moveTo", (10, 20)), ("drag", (300, 200))]

    def test_path_with_one_point_sends_nothing(self, monkeypatch, quiet_desktop):
        mouse = _Mouse(monkeypatch)
        res = ComputerUse()._drag({"path": [[1, 1]]})
        assert res["ok"] is False and mouse.events == []


def types_ns(center):
    import types
    return types.SimpleNamespace(center=center)


class TestDragSchemaAndParsing:
    def test_tool_offers_coordinates_and_path(self):
        drag = next(t for t in ALL_TOOLS if t.name == "cua_drag")
        names = {p.name for p in drag.params}
        assert {"from_element_id", "to_element_id", "x", "y", "to_x", "to_y", "path"} <= names
        assert "path" in drag.description

    def test_flat_and_nested_forms_keep_to_x_to_y_and_path(self):
        flat = parse_planned_step('{"thought": "t", "action": "cua_drag", "params": {"x": 100, "y": 200, "to_x": 800, "to_y": 600}}')
        assert flat.params["to_x"] == 800 and flat.params["to_y"] == 600
        nested = parse_planned_step(
            '{"thought": "t", "action": {"tool": "cua_drag", "params": {"path": [[1, 2], [3, 4]], "to_x": 5, "to_y": 6}}}'
        )
        assert nested.params["path"] == [[1, 2], [3, 4]] and nested.params["to_x"] == 5


@pytest.mark.asyncio
async def test_loop_converts_drag_points_from_1000_space_to_screen():
    snap = ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title="Paint", class_name="MSPaintApp", rect=(0, 0, 1920, 1080)),
        ocr_lines=[], width=1920, height=1080, png_bytes=b"x", image_width=1280, image_height=720,
    )
    gemma = AsyncMock()
    gemma.generate_text.side_effect = [
        '{"thought": "t", "action": "cua_drag", "params": {"x": 500, "y": 500, "to_x": 1000, "to_y": 0, "path": [[0, 0], [500, 500]]}, "is_completed": false, "user_update": "x"}',
        '{"thought": "d", "action": "converse", "params": {"response": "ok"}, "is_completed": true, "final_response": "ok"}',
    ]
    dispatcher = AsyncMock()
    dispatcher.execute.return_value = {"status": "ok", "result": {"ok": True, "sent": True, "verified": True}}
    perception = MagicMock()
    perception.capture_snapshot.return_value = snap
    loop = AgentLoop(gemma=gemma, dispatcher=dispatcher, perception=perception, grounder=AsyncMock(), fleet_config=build_gpu_single_preset())
    await loop.run("draw a line")
    sent = [c[0][0] for c in dispatcher.execute.call_args_list if c[0][0].tool == "cua_drag"][0].params
    assert (sent["x"], sent["y"]) == (960, 540)
    assert (sent["to_x"], sent["to_y"]) == (1920, 0)
    assert sent["path"] == [[0, 0], [960, 540]]
