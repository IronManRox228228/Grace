"""A grounded pixel point must beat a name lookup in the element graph.

Real run evidence: Holo's grounder located "Play" accurately, but `_click`
resolved `target_name="Play"` against the graph first and clicked the first
element whose text contained "play" - ignoring the point entirely.
"""

from unittest.mock import MagicMock, patch

from grace.automation.computer_use import ComputerUse
from grace.perception.element_graph import ElementGraph, WindowRef
from grace.perception.elements import FRAME_PAGE, ElementNode

STATE = {"window": "W", "elements": 5, "focus_id": 1, "focus_name": "", "focus_value": "", "blind": False}


def _graph():
    wrong = ElementNode(id=7, role="button", name="Play next", rect=(10, 10, 50, 50),
                        center=(30, 30), frame=FRAME_PAGE)
    return ElementGraph(elements=[wrong], window=WindowRef(hwnd=1, title="W"))


def _run(params):
    cu = ComputerUse()
    resolver = MagicMock()
    resolver.resolve.return_value = MagicMock(x=800, y=420, method="explicit", element=None)
    with patch("grace.automation.computer_use._foreground_title", return_value=""), \
         patch("grace.automation.computer_use._screen_state", return_value=STATE), \
         patch("grace.automation.computer_use._settled_state", return_value=STATE), \
         patch("grace.automation.coordinate_resolver.CoordinateResolver", return_value=resolver), \
         patch("grace.automation.win32_driver.Win32Driver.click_at", return_value=True) as click:
        res = cu._click(params)
    return res, click, resolver


def test_grounded_point_beats_graph_name_match():
    res, click, resolver = _run({"target_name": "Play", "x": 800, "y": 420, "_graph": _graph()})
    assert res["ok"] is True
    assert click.call_args.args[:2] == (800, 420)
    assert resolver.resolve.call_args.kwargs["target_name"] is None


def test_name_only_click_still_uses_graph():
    res, click, _ = _run({"target_name": "Play next", "_graph": _graph()})
    assert res["ok"] is True
    assert click.call_args.args[:2] == (30, 30)
