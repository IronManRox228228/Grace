"""Unit tests for the CoordinateResolver cascade."""

import unittest
from grace.automation.coordinate_resolver import CoordinateResolver
from grace.automation.ui_inspector import UIInspector


class _EmptyInspector(UIInspector):
    """An inspector that reports an empty screen and never touches the desktop.

    Without this these tests consulted the real foreground window. `resolve()`
    with a bare x/y runs the UIA point-snap pass, and `find_element_at_point`
    calls `inspect_active_window()` whenever its cache is empty - so a small
    control happening to sit within 30px of the asserted coordinate on whoever's
    machine is running the suite would snap the result somewhere else.

    It passed for a long time and then failed once, mid-suite, because a window
    had moved. A unit test that reads the developer's screen is not testing the
    resolver.
    """

    def inspect_active_window(self, *args, **kwargs):
        self._last_elements = []
        return []

    def find_element_at_point(self, *args, **kwargs):
        return None


class TestCoordinateResolver(unittest.TestCase):

    def setUp(self):
        self.resolver = CoordinateResolver(inspector=_EmptyInspector())

    def test_explicit_coordinate_resolution(self):
        target = self.resolver.resolve(x=100, y=200)
        self.assertIsNotNone(target)
        self.assertEqual(target.x, 100)
        self.assertEqual(target.y, 200)
        self.assertEqual(target.method, "explicit")

    def test_window_relative_coordinate_normalization(self):
        # Window at left=500, top=300, width=800, height=600
        window_bounds = (500, 300, 1300, 900)
        # Relative coord (x=100, y=150) inside window -> Absolute (600, 450)
        target = self.resolver.resolve(x=100, y=150, window_bounds=window_bounds)
        self.assertIsNotNone(target)
        self.assertEqual(target.x, 600)
        self.assertEqual(target.y, 450)

    def test_already_absolute_coordinate(self):
        # Window at left=500, top=300, width=800, height=600
        window_bounds = (500, 300, 1300, 900)
        # Absolute coord x=950, y=700 (outside relative range 0..800/0..600)
        target = self.resolver.resolve(x=950, y=700, window_bounds=window_bounds)
        self.assertIsNotNone(target)
        self.assertEqual(target.x, 950)
        self.assertEqual(target.y, 700)

    def test_clamp_center_to_window(self):
        """Bug #5 fix: out-of-window centers are clamped inside the window."""
        window_bounds = (100, 100, 1000, 800)
        # Point far outside on every edge gets clamped to the bottom-right corner.
        cx, cy = CoordinateResolver._clamp_center_to_window((5000, 5000), window_bounds)
        self.assertEqual((cx, cy), (1000, 800))
        # Point far to the upper-left gets clamped to the top-left corner.
        cx, cy = CoordinateResolver._clamp_center_to_window((-50, -50), window_bounds)
        self.assertEqual((cx, cy), (100, 100))
        # Point already inside is unchanged.
        cx, cy = CoordinateResolver._clamp_center_to_window((500, 400), window_bounds)
        self.assertEqual((cx, cy), (500, 400))

    def test_clamp_no_op_without_window_bounds(self):
        cx, cy = CoordinateResolver._clamp_center_to_window((5000, 5000), None)
        self.assertEqual((cx, cy), (5000, 5000))

    def test_explicit_pass_clamps_out_of_window_coords(self):
        """Pass 5 (explicit) must clamp the resolved coordinates into the window."""
        window_bounds = (100, 100, 1000, 800)
        target = self.resolver.resolve(x=9000, y=9000, window_bounds=window_bounds)
        self.assertIsNotNone(target)
        self.assertEqual(target.method, "explicit")
        # (9000,9000) is outside the relative range and is treated as absolute,
        # then clamped to the bottom-right corner.
        self.assertEqual(target.x, 1000)
        self.assertEqual(target.y, 800)


if __name__ == "__main__":
    unittest.main()
