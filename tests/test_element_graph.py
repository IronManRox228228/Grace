"""Tests for the structured element graph.

The headline case is the one Grace used to get wrong: told to search on
YouTube, it clicked the browser's address bar instead of the page's own search
box. Both are edit controls with search-ish names; the `frame` field is what
separates them.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.perception.element_graph import ElementGraph, WindowRef, _drop_overlapping
from grace.perception.elements import (
    FRAME_APP,
    FRAME_CHROME,
    FRAME_PAGE,
    SOURCE_DOM,
    SOURCE_UIA,
    ElementNode,
    elements_to_json,
    elements_to_prompt,
    find_at_point,
    find_by_name,
)
from grace.perception.uia_provider import _parse_aria_properties, is_browser_window


def node(id, role, name, rect, **kw):
    left, top, right, bottom = rect
    kw.setdefault("center", ((left + right) // 2, (top + bottom) // 2))
    return ElementNode(id=id, role=role, name=name, rect=rect, **kw)


@pytest.fixture
def youtube_graph():
    """A browser showing YouTube: omnibox in chrome, search box in the page."""
    elements = [
        node(1, "edit", "Address and search bar", (300, 60, 1500, 92),
             frame=FRAME_CHROME, placeholder="Search or enter web address",
             value="https://www.youtube.com", focusable=True, container="Toolbar"),
        node(2, "button", "Refresh", (250, 60, 282, 92), frame=FRAME_CHROME),
        node(3, "tab", "YouTube", (100, 20, 400, 55), frame=FRAME_CHROME),
        node(4, "searchbox", "Search", (600, 130, 1200, 168),
             frame=FRAME_PAGE, placeholder="Search YouTube",
             focusable=True, container="YouTube - masthead"),
        node(5, "button", "Search", (1200, 130, 1264, 168), frame=FRAME_PAGE),
        node(6, "link", "Home", (40, 200, 180, 232), frame=FRAME_PAGE),
    ]
    return ElementGraph(
        elements=elements,
        window=WindowRef(hwnd=1, title="YouTube - Microsoft Edge", class_name="Chrome_WidgetWin_1"),
        sources=("uia",),
    )


class TestFrameDisambiguation:
    def test_page_search_box_preferred_over_address_bar(self, youtube_graph):
        # The regression this whole phase exists to fix.
        found = youtube_graph.resolve(target_name="Search")
        assert found is not None
        assert found.id == 4
        assert found.frame == FRAME_PAGE
        assert found.placeholder == "Search YouTube"

    def test_explicit_chrome_frame_still_reaches_address_bar(self, youtube_graph):
        found = youtube_graph.resolve(target_name="Address", frame=FRAME_CHROME)
        assert found is not None and found.id == 1

    def test_frame_filter_excludes_other_frame(self, youtube_graph):
        # "Refresh" only exists in chrome, so a page-scoped lookup must miss.
        assert youtube_graph.resolve(target_name="Refresh", frame=FRAME_PAGE) is None

    def test_role_filter(self, youtube_graph):
        found = youtube_graph.resolve(target_name="Search", role="button")
        assert found is not None and found.id == 5

    def test_non_browser_window_does_not_prefer_page(self):
        graph = ElementGraph(
            elements=[node(1, "edit", "Search", (0, 0, 100, 20), frame=FRAME_APP)],
            window=WindowRef(hwnd=1, title="Notepad", class_name="Notepad"),
        )
        assert graph.window.is_browser is False
        assert graph.resolve(target_name="Search").id == 1


class TestResolution:
    def test_id_wins_over_name(self, youtube_graph):
        found = youtube_graph.resolve(element_id=2, target_name="Search")
        assert found.id == 2

    def test_unknown_id_falls_back_to_name(self, youtube_graph):
        found = youtube_graph.resolve(element_id=999, target_name="Home")
        assert found is not None and found.id == 6

    def test_unknown_target_returns_none(self, youtube_graph):
        assert youtube_graph.resolve(target_name="NonExistentThing9999") is None

    def test_no_criteria_returns_none(self, youtube_graph):
        assert youtube_graph.resolve() is None

    def test_focused_element(self, youtube_graph):
        assert youtube_graph.focused() is None
        youtube_graph.elements[3].focused = True
        assert youtube_graph.focused().id == 4

    def test_exact_name_beats_substring(self):
        elements = [
            node(1, "button", "Search settings", (0, 0, 50, 20)),
            node(2, "button", "Search", (0, 30, 50, 50)),
        ]
        assert find_by_name(elements, "Search").id == 2

    def test_placeholder_match(self):
        elements = [node(1, "edit", "", (0, 0, 50, 20), placeholder="Search YouTube")]
        assert find_by_name(elements, "Search YouTube").id == 1

    def test_disabled_and_offscreen_are_skipped(self):
        elements = [
            node(1, "button", "Go", (0, 0, 50, 20), enabled=False),
            node(2, "button", "Go", (0, 30, 50, 50), offscreen=True),
            node(3, "button", "Go", (0, 60, 50, 80)),
        ]
        assert find_by_name(elements, "Go").id == 3


class TestPointSnapping:
    def test_snaps_to_containing_small_control(self):
        elements = [node(1, "button", "OK", (100, 100, 160, 130))]
        assert find_at_point(elements, 130, 115).id == 1

    def test_snaps_to_nearby_control_within_tolerance(self):
        elements = [node(1, "button", "OK", (100, 100, 160, 130))]
        assert find_at_point(elements, 135, 140, tolerance_px=40).id == 1

    def test_does_not_snap_beyond_tolerance(self):
        elements = [node(1, "button", "OK", (100, 100, 160, 130))]
        assert find_at_point(elements, 600, 600) is None

    def test_large_containers_are_not_snapped_to(self):
        # Snapping onto the centre of a full-width panel is worse than leaving
        # the predicted coordinate alone.
        elements = [node(1, "pane", "Content", (0, 0, 1920, 1080))]
        assert find_at_point(elements, 500, 500) is None

    def test_prefers_closest_of_several(self):
        elements = [
            node(1, "button", "A", (100, 100, 140, 130)),
            node(2, "button", "B", (200, 100, 240, 130)),
        ]
        assert find_at_point(elements, 215, 115).id == 2


class TestSerialisation:
    def test_compact_json_drops_defaults(self, youtube_graph):
        data = json.loads(elements_to_json(youtube_graph.elements, compact=True))
        home = next(d for d in data if d["name"] == "Home")
        assert "value" not in home and "placeholder" not in home
        assert home["frame"] == FRAME_PAGE

    def test_compact_json_keeps_meaningful_fields(self, youtube_graph):
        data = json.loads(elements_to_json(youtube_graph.elements, compact=True))
        search = next(d for d in data if d["id"] == 4)
        assert search["placeholder"] == "Search YouTube"
        assert search["container"] == "YouTube - masthead"

    def test_full_json_has_every_field(self, youtube_graph):
        data = youtube_graph.elements[0].to_dict(compact=False)
        for key in ("value", "placeholder", "container", "focused", "focusable",
                    "enabled", "offscreen", "automation_id", "source"):
            assert key in data

    def test_prompt_explains_frame_when_both_present(self, youtube_graph):
        prompt = youtube_graph.to_prompt()
        assert "frame" in prompt
        assert "always frame=\"page\"" in prompt
        assert json.loads(prompt.split("\n")[-1])  # last line is valid JSON

    def test_prompt_omits_frame_note_for_plain_app(self):
        elements = [node(1, "button", "OK", (0, 0, 50, 20), frame=FRAME_APP)]
        prompt = elements_to_prompt(elements)
        assert "always frame=" not in prompt

    def test_prompt_reports_truncation(self):
        elements = [node(i, "button", f"B{i}", (0, i, 50, i + 20)) for i in range(1, 60)]
        prompt = elements_to_prompt(elements, limit=10)
        assert "49 further elements not shown" in prompt

    def test_empty_prompt(self):
        assert "none detected" in elements_to_prompt([])


class TestBrowserDetection:
    @pytest.mark.parametrize("title,cls", [
        ("YouTube - Google Chrome", "Chrome_WidgetWin_1"),
        ("anything", "MozillaWindowClass"),
        ("Docs - Microsoft Edge", "SomeClass"),
        ("News - Brave", ""),
        ("Site - Opera", ""),
    ])
    def test_browsers_detected(self, title, cls):
        assert is_browser_window(title, cls) is True

    @pytest.mark.parametrize("title,cls", [
        ("Untitled - Notepad", "Notepad"),
        ("", ""),
        ("Calculator", "ApplicationFrameWindow"),
    ])
    def test_non_browsers(self, title, cls):
        assert is_browser_window(title, cls) is False


class TestAriaProperties:
    def test_parses_placeholder(self):
        parsed = _parse_aria_properties("placeholder=Search YouTube;expanded=false")
        assert parsed["placeholder"] == "Search YouTube"
        assert parsed["expanded"] == "false"

    def test_empty_and_malformed(self):
        assert _parse_aria_properties("") == {}
        assert _parse_aria_properties("novalue;also") == {}

    def test_ignores_blank_keys(self):
        assert _parse_aria_properties("=x;a=1") == {"a": "1"}


class TestDomUiaMerge:
    def test_overlapping_page_elements_deduped(self):
        dom = [node(1, "button", "Play", (100, 100, 160, 130),
                    frame=FRAME_PAGE, source=SOURCE_DOM)]
        uia = [
            node(2, "button", "Play", (101, 101, 161, 131),
                 frame=FRAME_PAGE, source=SOURCE_UIA),
            node(3, "edit", "Address bar", (0, 0, 900, 30),
                 frame=FRAME_CHROME, source=SOURCE_UIA),
        ]
        kept = _drop_overlapping(uia, dom)
        # The duplicated page control goes; browser chrome, which CDP cannot
        # see at all, must survive.
        assert [e.id for e in kept] == [3]

    def test_distinct_page_elements_survive(self):
        dom = [node(1, "button", "Play", (100, 100, 160, 130), frame=FRAME_PAGE)]
        uia = [node(2, "button", "Pause", (400, 400, 460, 430), frame=FRAME_PAGE)]
        assert len(_drop_overlapping(uia, dom)) == 1


class TestElementNode:
    def test_geometry(self):
        element = node(1, "button", "OK", (10, 20, 110, 70))
        assert element.width == 100
        assert element.height == 50
        assert element.center == (60, 45)

    def test_contains_and_distance(self):
        element = node(1, "button", "OK", (0, 0, 100, 100))
        assert element.contains(50, 50)
        assert not element.contains(150, 50)
        assert element.distance_to(50, 50) == 0

    def test_is_interactive(self):
        assert node(1, "button", "OK", (0, 0, 10, 10)).is_interactive
        assert not node(2, "pane", "Panel", (0, 0, 10, 10)).is_interactive

    def test_search_text_includes_placeholder_and_id(self):
        element = node(1, "edit", "Name", (0, 0, 10, 10),
                       placeholder="Your name", automation_id="name-input")
        text = element.search_text()
        assert "name" in text and "your name" in text and "name-input" in text


class TestGraphBasics:
    def test_len_and_age(self, youtube_graph):
        assert len(youtube_graph) == 6
        assert youtube_graph.age_seconds >= 0

    def test_empty_graph_resolves_to_none(self):
        graph = ElementGraph()
        assert graph.resolve(element_id=1) is None
        assert graph.focused() is None
        assert len(graph) == 0
