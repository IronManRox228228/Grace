"""Tests for the vision grounder's point/coordinate parser.

Covers:
- each coordinate format the code supports (grounder.py:scale_to_screen)
- malformed model output (grounder.py:parse_grounder_response -> None)
- out-of-bounds points (scale_to_screen + Grounder.locate)
- empty input (parse_grounder_response -> None)
- extra text around the point (parse_grounder_response still extracts coords)
"""

import asyncio
import unittest

from grace.agent.grounder import (
    Grounder,
    GroundedPoint,
    parse_grounder_response,
    scale_to_screen,
)


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def generate_text(self, prompt, system_prompt, **kwargs):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, **kwargs})
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class TestCoordinateFormats(unittest.TestCase):
    """scale_to_screen for every supported coordinate_format value."""

    def test_format_0_to_1000(self):
        # 500/1000 -> centre of a 1920x1080 screen
        res = scale_to_screen(500, 500, (1280, 720), (1920, 1080), coordinate_format="0-1000")
        self.assertEqual(res, (960, 540))

    def test_format_norm_1000_alias(self):
        res = scale_to_screen(500, 500, (1280, 720), (1920, 1080), coordinate_format="norm_1000")
        self.assertEqual(res, (960, 540))

    def test_format_1000_alias(self):
        res = scale_to_screen(500, 500, (1280, 720), (1920, 1080), coordinate_format="1000")
        self.assertEqual(res, (960, 540))

    def test_format_0_to_1(self):
        # 0.5/1.0 -> centre of a 1920x1080 screen
        res = scale_to_screen(0.5, 0.5, (1280, 720), (1920, 1080), coordinate_format="0-1")
        self.assertEqual(res, (960, 540))

    def test_format_norm_1_alias(self):
        res = scale_to_screen(0.5, 0.5, (1280, 720), (1920, 1080), coordinate_format="norm_1")
        self.assertEqual(res, (960, 540))

    def test_format_percent(self):
        # 50% / 100 -> centre of a 1920x1080 screen
        res = scale_to_screen(50, 50, (1280, 720), (1920, 1080), coordinate_format="percent")
        self.assertEqual(res, (960, 540))

    def test_format_pixels(self):
        # 640x360 of a 1280x720 image -> centre of a 1920x1080 screen
        res = scale_to_screen(640, 360, (1280, 720), (1920, 1080), coordinate_format="pixels")
        self.assertEqual(res, (960, 540))

    def test_format_pixels_downscaled_screenshot_maps_to_screen(self):
        # Item 11: explicit coordinates from a downscaled screenshot (SCREENSHOT_MAX_WIDTH=1280)
        # must be mapped back to physical screen pixels. Image (640, 40) on a 1280x720
        # downscaled screenshot of a 1920x1080 screen lands at screen (960, 60) (scale 1.5).
        res = scale_to_screen(640, 40, (1280, 720), (1920, 1080), coordinate_format="pixels")
        self.assertEqual(res, (960, 60))


class TestMalformedModelOutput(unittest.TestCase):
    """parse_grounder_response must return None for output it cannot ground."""

    def test_none_input(self):
        self.assertIsNone(parse_grounder_response(None))

    def test_empty_string_input(self):
        self.assertIsNone(parse_grounder_response(""))

    def test_whitespace_only_input(self):
        self.assertIsNone(parse_grounder_response("   \n  \t"))

    def test_garbage_text_returns_none(self):
        self.assertIsNone(parse_grounder_response("I will click the button for you."))
        self.assertIsNone(parse_grounder_response("Thought: I should do something."))

    def test_three_numbers_is_not_a_point(self):
        self.assertIsNone(parse_grounder_response("click(point=(100, 200, 300))"))

    def test_nested_json_returns_none(self):
        # The JSON regex is flat-only (grounder.py:69); nested braces are not parsed.
        self.assertIsNone(parse_grounder_response('{"point": [100, 200], "meta": {"a": 1}}'))


class TestOutOfBoundsPoints(unittest.TestCase):
    """Points outside the valid range for a format must not be grounded."""

    def test_out_of_bounds_0_to_1000(self):
        self.assertIsNone(scale_to_screen(-1, 500, (1280, 720), (1920, 1080), coordinate_format="0-1000"))
        self.assertIsNone(scale_to_screen(1001, 500, (1280, 720), (1920, 1080), coordinate_format="0-1000"))
        self.assertIsNone(scale_to_screen(500, -5, (1280, 720), (1920, 1080), coordinate_format="0-1000"))
        self.assertIsNone(scale_to_screen(500, 1050, (1280, 720), (1920, 1080), coordinate_format="0-1000"))

    def test_out_of_bounds_0_to_1(self):
        self.assertIsNone(scale_to_screen(-0.1, 0.5, (1280, 720), (1920, 1080), coordinate_format="0-1"))
        self.assertIsNone(scale_to_screen(1.1, 0.5, (1280, 720), (1920, 1080), coordinate_format="0-1"))
        self.assertIsNone(scale_to_screen(0.5, -0.01, (1280, 720), (1920, 1080), coordinate_format="0-1"))
        self.assertIsNone(scale_to_screen(0.5, 1.05, (1280, 720), (1920, 1080), coordinate_format="0-1"))

    def test_out_of_bounds_percent(self):
        self.assertIsNone(scale_to_screen(-1, 50, (1280, 720), (1920, 1080), coordinate_format="percent"))
        self.assertIsNone(scale_to_screen(101, 50, (1280, 720), (1920, 1080), coordinate_format="percent"))
        self.assertIsNone(scale_to_screen(50, -0.5, (1280, 720), (1920, 1080), coordinate_format="percent"))
        self.assertIsNone(scale_to_screen(50, 100.5, (1280, 720), (1920, 1080), coordinate_format="percent"))

    def test_out_of_bounds_pixels(self):
        self.assertIsNone(scale_to_screen(-10, 100, (1280, 720), (1920, 1080), coordinate_format="pixels"))
        # Exceeds image bounds and 1000, so the overflow net cannot apply either.
        self.assertIsNone(scale_to_screen(2500, 500, (1280, 720), (1920, 1080), coordinate_format="pixels"))

    def test_grounder_locate_out_of_bounds_returns_none(self):
        llm = FakeLLM(["Action: click(point=(1500, 500))"])
        grounder = Grounder(llm, coordinate_format="0-1000")
        res = asyncio.run(grounder.locate("button", b"png", (1280, 720), (1920, 1080)))
        self.assertIsNone(res)


class TestEmptyInput(unittest.TestCase):
    """Empty or missing input must never raise and must return None."""

    def test_parse_grounder_response_empty(self):
        self.assertIsNone(parse_grounder_response(None))
        self.assertIsNone(parse_grounder_response(""))
        self.assertIsNone(parse_grounder_response("   "))

    def test_scale_to_screen_zero_dimensions_no_crash(self):
        # image_size or screen_size of 0 returns the raw rounded coords without bounds-checking
        res = scale_to_screen(10, 20, (0, 720), (1920, 1080), coordinate_format="0-1000")
        self.assertEqual(res, (10, 20))


class TestExtraTextAroundPoint(unittest.TestCase):
    """The parser must find coordinates embedded in surrounding prose."""

    def test_extra_text_before_and_after(self):
        text = "blah click(point=(300, 400)) blah"
        parsed = parse_grounder_response(text)
        self.assertIsNotNone(parsed)
        x, y, _ = parsed
        self.assertEqual((x, y), (300.0, 400.0))

    def test_point_in_prose_sentence(self):
        text = "The button is at point: (300, 400) now"
        parsed = parse_grounder_response(text)
        self.assertIsNotNone(parsed)
        x, y, _ = parsed
        self.assertEqual((x, y), (300.0, 400.0))

    def test_thought_and_action_with_surrounding_text(self):
        text = "Preamble. Thought: Here is the submit button.\nAction: click(point=(469, 472)). Postscript."
        parsed = parse_grounder_response(text)
        self.assertIsNotNone(parsed)
        x, y, thought = parsed
        self.assertEqual((x, y), (469.0, 472.0))
        self.assertIn("submit button", thought)


class TestGrounderEndToEnd(unittest.TestCase):
    """Grounder.locate parses and scales a coordinate response."""

    def test_valid_locates_point(self):
        llm = FakeLLM(["Action: click(point=(500, 500))"])
        grounder = Grounder(llm, coordinate_format="0-1000")
        res = asyncio.run(grounder.locate("button", b"png", (1280, 720), (1920, 1080)))
        self.assertIsInstance(res, GroundedPoint)
        self.assertEqual(res.x, 960)
        self.assertEqual(res.y, 540)

    def test_empty_png_or_description_returns_none(self):
        llm = FakeLLM(["Action: click(point=(500, 500))"])
        grounder = Grounder(llm, coordinate_format="0-1000")
        self.assertIsNone(asyncio.run(grounder.locate("button", b"", (1280, 720), (1920, 1080))))
        self.assertIsNone(asyncio.run(grounder.locate("", b"png", (1280, 720), (1920, 1080))))


if __name__ == "__main__":
    unittest.main()
