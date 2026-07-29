"""Test suite for UIInspector and Zero-Guessing Target Resolution."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.automation.ui_inspector import (
    UIInspector,
    UIElement,
    MAX_ELEMENTS_DEFAULT,
    MAX_ELEMENTS_BROWSER,
    MAX_DEPTH_BROWSER,
)
from grace.automation.computer_use import ComputerUse


class TestUIInspector:
    """Test suite for UIInspector."""

    def test_inspect_active_window(self):
        inspector = UIInspector()
        elements = inspector.inspect_active_window()
        assert isinstance(elements, list)

    def test_find_element_by_index(self):
        inspector = UIInspector()
        elem1 = UIElement(index=1, name="Shorts", control_type="Button", bounds=(100, 100, 200, 150), center=(150, 125))
        inspector._last_elements = [elem1]

        found = inspector.find_element(element_index=1)
        assert found is not None
        assert found.name == "Shorts"
        assert found.center == (150, 125)

    def test_find_element_by_name(self):
        inspector = UIInspector()
        elem1 = UIElement(index=1, name="Shorts", control_type="Button", bounds=(100, 100, 200, 150), center=(150, 125))
        inspector._last_elements = [elem1]

        found = inspector.find_element(target_name="shorts")
        assert found is not None
        assert found.name == "Shorts"

    def test_zero_guessing_guard(self):
        cu = ComputerUse()
        cu.start()
        res = cu.perform("click", {"target_name": "NonExistentFakeButton9999"})
        assert res["status"] == "element_not_found"
        assert "not found" in res["error"]


class TestBrowserDetection:
    """Bug #4 fix: browser windows get deeper traversal and a higher element cap."""

    def test_browser_cap_and_depth_constants(self):
        assert MAX_ELEMENTS_BROWSER == 250
        assert MAX_ELEMENTS_DEFAULT == 100
        assert MAX_DEPTH_BROWSER == 8

    def test_is_browser_window_by_class(self):
        assert UIInspector._is_browser_window(0, "YouTube Music", "Chrome_WidgetWin_1") is True
        assert UIInspector._is_browser_window(0, "A tab", "MozillaWindowClass") is True

    def test_is_browser_window_by_title_keyword(self):
        # Newer Edge variants sometimes report different class names; title is the fallback.
        assert UIInspector._is_browser_window(0, "Something - Microsoft Edge", "Win32Class") is True
        assert UIInspector._is_browser_window(0, "Inbox - Google Chrome", "Other") is True

    def test_is_browser_window_false_for_native_app(self):
        assert UIInspector._is_browser_window(0, "Untitled - Notepad", "Notepad") is False
        assert UIInspector._is_browser_window(0, "Calculator", "Windows.UI.Core.CoreWindow") is False

    def test_walk_uia_tree_recurses_and_respects_cap(self):
        """Bug #4 fix: the recursive TreeWalker visits children up to depth/cap.

        Driven with a stub `uia` whose tree is a single chain of named buttons;
        the walker must collect them until the element cap is hit.
        """
        from unittest.mock import MagicMock

        def make_elem(name, ctl=50000):  # 50000 = Button
            e = MagicMock()
            e.CurrentName = name
            e.CurrentControlType = ctl
            r = MagicMock()
            r.left, r.top, r.right, r.bottom = 10, 10, 50, 50
            e.CurrentBoundingRectangle = r
            return e

        # Build a flat list of N buttons; each element's sibling is the next.
        buttons = [make_elem(f"Btn{i}") for i in range(5)]

        walker = MagicMock()
        # First child of the root is buttons[0]; sibling chain links the rest.
        walker.GetFirstChildElement.return_value = buttons[0]
        walker.GetNextSiblingElement.side_effect = buttons[1:] + [None]

        uia = MagicMock()
        uia.RawViewWalker = walker

        inspector = UIInspector()
        elements = []
        inspector._walk_uia_tree(uia, MagicMock(), elements, depth=0, max_depth=8, cap=3)

        # Cap=3 must stop collection at 3 elements despite 5 being available.
        assert len(elements) == 3
        assert elements[0].name == "Btn0"


# Expose patch for the import-free test file.
from unittest.mock import patch


