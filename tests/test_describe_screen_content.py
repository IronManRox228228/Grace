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
        detail = res["screen_detail"]
        text = res["text"]

        # All screen detail text must be inside wrap_untrusted
        assert _UNTRUSTED_START in detail
        assert _UNTRUSTED_END in detail

        # Spoken text is short sentence with no markers or element dump
        assert _UNTRUSTED_START not in text
        assert _UNTRUSTED_END not in text
        assert "Button:" not in text and "Link:" not in text
        assert len(text) <= 160
        assert text == "You're on YouTube Music."

        # Verify page items appear before chrome Minimize
        signal_idx = detail.index("Signal")
        play_idx = detail.index("Play")
        min_idx = detail.index("Minimize")

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

        detail = res["screen_detail"]
        assert "Heading: 'Top Results'" in detail
        assert "Focused edit: 'Search songs'" in detail
        assert "Item: 'Result 1'" in detail
        assert "Heading:" not in res["text"]
        assert res["text"] == "You're on Player."

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
        assert "Minimize" not in res["screen_detail"]
        assert "Minimize" not in res["text"]

    @pytest.mark.asyncio
    async def test_spoken_text_sanitizes_title_and_excludes_raw_dump(self, mock_graph_builder):
        """Spoken text is built deterministically, has no markers or button/link lists, and strips < >."""
        btn = ElementNode(
            id=1, role="button", name="Guide", rect=(10, 10, 50, 50),
            center=(30, 30), frame=FRAME_PAGE,
        )
        link = ElementNode(
            id=2, role="link", name="Home", rect=(10, 60, 50, 100),
            center=(30, 80), frame=FRAME_PAGE,
        )
        graph = ElementGraph(
            elements=[btn, link],
            window=WindowRef(
                hwnd=999,
                title="<Imagine (Ultimate Mix) | YouTube Music and 2 more pages> - Personal - Microsoft Edge",
                class_name="Chrome_WidgetWin_1",
            ),
        )
        mock_graph_builder.get.return_value = graph

        dispatcher = Dispatcher(computer_use=None)
        res = await dispatcher.execute(Intent(tool="describe_screen", params={}))

        text = res["text"]
        detail = res["screen_detail"]

        # Detail has untrusted markers and elements
        assert _UNTRUSTED_START in detail
        assert _UNTRUSTED_END in detail
        assert "Button: 'Guide'" in detail
        assert "Link: 'Home'" in detail

        # Spoken text is short and clean
        assert text == 'You\'re on "Imagine (Ultimate Mix) | YouTube Music" in Microsoft Edge.'
        assert _UNTRUSTED_START not in text
        assert _UNTRUSTED_END not in text
        assert "Button:" not in text
        assert "Link:" not in text
        assert "Guide" not in text
        assert "Home" not in text
        assert "<" not in text and ">" not in text
        assert len(text) <= 160

    @pytest.mark.asyncio
    async def test_spoken_text_edge_beta_with_more_pages_and_profile(self, mock_graph_builder):
        """Live title with 'and 1 more page', 'Personal', and zero-width char in 'Microsoft Edge Beta'."""
        btn = ElementNode(
            id=1, role="button", name="Play", rect=(10, 10, 50, 50),
            center=(30, 30), frame=FRAME_PAGE,
        )
        title = "Imagine (Ultimate Mix) | YouTube Music and 1 more page - Personal - Microsoft\u200b Edge Beta"
        graph = ElementGraph(
            elements=[btn],
            window=WindowRef(
                hwnd=1001,
                title=title,
                class_name="Chrome_WidgetWin_1",
            ),
        )
        mock_graph_builder.get.return_value = graph

        dispatcher = Dispatcher(computer_use=None)
        res = await dispatcher.execute(Intent(tool="describe_screen", params={}))

        assert res["text"] == 'You\'re on "Imagine (Ultimate Mix) | YouTube Music" in Microsoft Edge Beta.'


def test_spoken_window_description_exact_live_edge_beta():
    from grace.tools.dispatcher import _spoken_window_description
    title = "Imagine (Ultimate Mix) | YouTube Music and 1 more page - Personal - Microsoft\u200b Edge Beta"
    assert _spoken_window_description(title) == 'You\'re on "Imagine (Ultimate Mix) | YouTube Music" in Microsoft Edge Beta.'
