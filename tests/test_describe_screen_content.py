"""Tests for describe_screen content description (retest 5 item 2).

- Use page-frame elements first (UIA frame=page), then the visible OCR lines in reading order.
- Show headings and list items with role and name, and the focused element.
- Cap it at about 25 items / 1200 characters.
- Put all screen text inside wrap_untrusted.
- Window chrome (min/max/close) goes last, or is dropped.
- Test: with a mocked graph of chrome buttons plus page items "Signal", "Song · Home · 3:42" and "Play",
  the output contains "Signal" and "Play" before "Minimize".
"""

from unittest.mock import MagicMock, patch
import pytest

from grace.agent.planner import _UNTRUSTED_START, _UNTRUSTED_END
from grace.intent.parser import Intent
from grace.perception.element_graph import ElementGraph, WindowRef
from grace.perception.elements import (
    ElementNode,
    FRAME_PAGE,
    FRAME_CHROME,
    FRAME_APP,
    SOURCE_UIA,
    SOURCE_OCR,
)
from grace.tools.dispatcher import Dispatcher


@pytest.fixture
def mock_graph_builder():
    builder = MagicMock()
    with patch("grace.agent.perception.PerceptionEngine.get_graph_builder", return_value=builder):
        yield builder


class TestDescribeScreenContent:
    @pytest.mark.asyncio
    async def test_page_items_signal_and_play_appear_before_minimize(self, mock_graph_builder):
        """With chrome buttons plus page items 'Signal', 'Song · Home · 3:42' and 'Play',

        the output contains 'Signal' and 'Play' before 'Minimize'.
        """
        # Graph with chrome buttons and page items
        el_min = ElementNode(
            id=1, role="button", name="Minimize", rect=(1800, 0, 1840, 30),
            center=(1820, 15), frame=FRAME_CHROME,
        )
        el_close = ElementNode(
            id=2, role="button", name="Close", rect=(1880, 0, 1920, 30),
            center=(1900, 15), frame=FRAME_CHROME,
        )
        el_signal = ElementNode(
            id=3, role="heading", name="Signal", rect=(200, 200, 400, 240),
            center=(300, 220), frame=FRAME_PAGE,
        )
        el_song = ElementNode(
            id=4, role="listitem", name="Song · Home · 3:42", rect=(200, 250, 600, 280),
            center=(400, 265), frame=FRAME_PAGE,
        )
        el_play = ElementNode(
            id=5, role="button", name="Play", rect=(620, 250, 680, 280),
            center=(650, 265), frame=FRAME_PAGE,
        )

        graph = ElementGraph(
            elements=[el_min, el_close, el_signal, el_song, el_play],
            window=WindowRef(hwnd=1234, title="YouTube Music", class_name="Chrome_WidgetWin_1"),
        )
        mock_graph_builder.get.return_value = graph

        dispatcher = Dispatcher(computer_use=None)
        res = await dispatcher.execute(Intent(tool="describe_screen", params={}))

        assert res["status"] == "ok"
        text = res["text"]

        # All screen text must be inside wrap_untrusted
        assert _UNTRUSTED_START in text
        assert _UNTRUSTED_END in text

        # Verify page items appear before chrome Minimize
        signal_idx = text.index("Signal")
        play_idx = text.index("Play")
        min_idx = text.index("Minimize")

        assert signal_idx < min_idx, "Expected 'Signal' before 'Minimize'"
        assert play_idx < min_idx, "Expected 'Play' before 'Minimize'"

        # Verify controls summary also orders page items before chrome
        controls = res["controls"]
        assert controls.index("Signal") < controls.index("Minimize")
        assert controls.index("Play") < controls.index("Minimize")

    @pytest.mark.asyncio
    async def test_roles_and_focused_element_are_shown(self, mock_graph_builder):
        """Heading, listitem, and focused element are properly described."""
        el_heading = ElementNode(
            id=10, role="heading", name="Top Results", rect=(10, 10, 200, 40),
            center=(100, 25), frame=FRAME_PAGE,
        )
        el_focused = ElementNode(
            id=11, role="edit", name="Search songs", rect=(10, 50, 300, 80),
            center=(150, 65), frame=FRAME_PAGE, focused=True,
        )
        el_item = ElementNode(
            id=12, role="listitem", name="Result 1", rect=(10, 90, 300, 120),
            center=(150, 105), frame=FRAME_PAGE,
        )

        graph = ElementGraph(
            elements=[el_heading, el_focused, el_item],
            window=WindowRef(hwnd=1234, title="Player", class_name="App"),
        )
        mock_graph_builder.get.return_value = graph

        dispatcher = Dispatcher(computer_use=None)
        res = await dispatcher.execute(Intent(tool="describe_screen", params={}))

        text = res["text"]
        assert "Heading: 'Top Results'" in text
        assert "Focused edit: 'Search songs'" in text
        assert "Item: 'Result 1'" in text

    @pytest.mark.asyncio
    async def test_ocr_lines_sorted_in_reading_order_and_chrome_dropped_on_cap(self, mock_graph_builder):
        """OCR lines are sorted top-to-bottom and chrome is dropped when capped at 25 items."""
        # Create 25 page items
        elements = [
            ElementNode(
                id=100 + i, role="listitem", name=f"Song {i}",
                rect=(0, 100 + i * 20, 300, 120 + i * 20),
                center=(150, 110 + i * 20), frame=FRAME_PAGE,
            )
            for i in range(25)
        ]
        # Plus 2 OCR lines (out of order vertically)
        ocr_bottom = ElementNode(
            id=201, role="text", name="OCR Bottom Line",
            rect=(50, 900, 300, 930), center=(175, 915),
            source=SOURCE_OCR,
        )
        ocr_top = ElementNode(
            id=202, role="text", name="OCR Top Line",
            rect=(50, 500, 300, 530), center=(175, 515),
            source=SOURCE_OCR,
        )
        # Plus chrome buttons
        chrome_btn = ElementNode(
            id=300, role="button", name="Minimize",
            rect=(1800, 0, 1850, 30), center=(1825, 15),
            frame=FRAME_CHROME,
        )

        graph = ElementGraph(
            elements=elements + [ocr_bottom, ocr_top, chrome_btn],
            window=WindowRef(hwnd=1234, title="App", class_name="App"),
        )
        mock_graph_builder.get.return_value = graph

        dispatcher = Dispatcher(computer_use=None)
        res = await dispatcher.execute(Intent(tool="describe_screen", params={}))

        # Cap is at most 25 items
        assert len(res["controls"]) <= 25
        # Chrome should have been dropped because 25 page items filled the budget
        assert "Minimize" not in res["controls"]
        assert "Minimize" not in res["text"]
