"""Unit tests for the model-agnostic vision grounder and parser."""

import asyncio
import unittest

from grace.agent.grounder import Grounder, GroundedPoint, parse_grounder_response, scale_to_screen


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


class TestGrounderParser(unittest.TestCase):
    """Test suite for parsing vision grounding outputs into coordinates."""

    def test_parse_action_click_point(self):
        text = "Thought: Here is the submit button.\nAction: click(point=(469, 472))"
        parsed = parse_grounder_response(text)
        self.assertIsNotNone(parsed)
        x, y, thought = parsed
        self.assertEqual(x, 469.0)
        self.assertEqual(y, 472.0)
        self.assertIn("submit button", thought)

    def test_parse_action_click_tuple(self):
        text = "Action: click((100, 200))"
        parsed = parse_grounder_response(text)
        self.assertIsNotNone(parsed)
        x, y, _ = parsed
        self.assertEqual(x, 100.0)
        self.assertEqual(y, 200.0)

    def test_parse_action_click_bare_numbers(self):
        text = "Action: click(150, 250)"
        parsed = parse_grounder_response(text)
        self.assertIsNotNone(parsed)
        x, y, _ = parsed
        self.assertEqual(x, 150.0)
        self.assertEqual(y, 250.0)

    def test_parse_action_click_start_box(self):
        text = "Action: click(start_box='(50, 60)')"
        parsed = parse_grounder_response(text)
        self.assertIsNotNone(parsed)
        x, y, _ = parsed
        self.assertEqual(x, 50.0)
        self.assertEqual(y, 60.0)

    def test_parse_box_tokens(self):
        text = "Action: click(start_box='<|box_start|>(320, 180)<|box_end|>')"
        parsed = parse_grounder_response(text)
        self.assertIsNotNone(parsed)
        x, y, _ = parsed
        self.assertEqual(x, 320.0)
        self.assertEqual(y, 180.0)

    def test_parse_json_point(self):
        text = '{"thought": "Located button", "point": [640, 360]}'
        parsed = parse_grounder_response(text)
        self.assertIsNotNone(parsed)
        x, y, thought = parsed
        self.assertEqual(x, 640.0)
        self.assertEqual(y, 360.0)
        self.assertEqual(thought, "Located button")

    def test_parse_json_xy(self):
        text = '```json\n{"x": 500, "y": 300}\n```'
        parsed = parse_grounder_response(text)
        self.assertIsNotNone(parsed)
        x, y, _ = parsed
        self.assertEqual(x, 500.0)
        self.assertEqual(y, 300.0)

    def test_parse_float_coordinates(self):
        text = "Action: click(point=(0.469, 0.472))"
        parsed = parse_grounder_response(text)
        self.assertIsNotNone(parsed)
        x, y, _ = parsed
        self.assertAlmostEqual(x, 0.469)
        self.assertAlmostEqual(y, 0.472)

    def test_parse_wait_returns_none(self):
        self.assertIsNone(parse_grounder_response("Action: wait()"))
        self.assertIsNone(parse_grounder_response("Thought: not visible\nAction: wait()"))

    def test_malformed_returns_none(self):
        self.assertIsNone(parse_grounder_response("I will click the button for you."))
        self.assertIsNone(parse_grounder_response("Thought: I should do something."))
        self.assertIsNone(parse_grounder_response(""))
        self.assertIsNone(parse_grounder_response(None))


class TestScaleToScreenCoordinates(unittest.TestCase):
    """Test scaling across different coordinate formats and out-of-bounds checks."""

    def test_norm_1000_format(self):
        # 500/1000 -> center of 1920x1080 is (960, 540)
        res = scale_to_screen(500, 500, (1280, 720), (1920, 1080), coordinate_format="0-1000")
        self.assertEqual(res, (960, 540))

    def test_norm_1_format(self):
        # 0.5/1.0 -> center of 1920x1080 is (960, 540)
        res = scale_to_screen(0.5, 0.5, (1280, 720), (1920, 1080), coordinate_format="0-1")
        self.assertEqual(res, (960, 540))

    def test_pixels_format(self):
        # 640/1280 -> center of 1920x1080 is (960, 540)
        res = scale_to_screen(640, 360, (1280, 720), (1920, 1080), coordinate_format="pixels")
        self.assertEqual(res, (960, 540))

    def test_out_of_bounds_norm_1000(self):
        self.assertIsNone(scale_to_screen(-1, 500, (1280, 720), (1920, 1080), coordinate_format="0-1000"))
        self.assertIsNone(scale_to_screen(1001, 500, (1280, 720), (1920, 1080), coordinate_format="0-1000"))
        self.assertIsNone(scale_to_screen(500, -5, (1280, 720), (1920, 1080), coordinate_format="0-1000"))
        self.assertIsNone(scale_to_screen(500, 1050, (1280, 720), (1920, 1080), coordinate_format="0-1000"))

    def test_out_of_bounds_norm_1(self):
        self.assertIsNone(scale_to_screen(-0.1, 0.5, (1280, 720), (1920, 1080), coordinate_format="0-1"))
        self.assertIsNone(scale_to_screen(1.1, 0.5, (1280, 720), (1920, 1080), coordinate_format="0-1"))
        self.assertIsNone(scale_to_screen(0.5, -0.01, (1280, 720), (1920, 1080), coordinate_format="0-1"))
        self.assertIsNone(scale_to_screen(0.5, 1.05, (1280, 720), (1920, 1080), coordinate_format="0-1"))

    def test_out_of_bounds_pixels(self):
        self.assertIsNone(scale_to_screen(-10, 100, (1280, 720), (1920, 1080), coordinate_format="pixels"))
        # Exceeds both image bounds and 1000
        self.assertIsNone(scale_to_screen(2500, 500, (1280, 720), (1920, 1080), coordinate_format="pixels"))

    def test_pixels_overflow_fallback_to_thousand(self):
        # When model emits 500 on a 400x300 image, it is treated as 0-1000
        res = scale_to_screen(500, 500, (400, 300), (1920, 1080), coordinate_format="pixels")
        self.assertEqual(res, (960, 540))


class TestGrounderClass(unittest.TestCase):
    def test_grounder_out_of_bounds_returns_none(self):
        llm = FakeLLM(["Action: click(point=(1500, 500))"])
        grounder = Grounder(llm, coordinate_format="0-1000")
        res = asyncio.run(grounder.locate("button", b"png", (1280, 720), (1920, 1080)))
        self.assertIsNone(res)

    def test_grounder_valid_locates_point(self):
        llm = FakeLLM(["Action: click(point=(500, 500))"])
        grounder = Grounder(llm, coordinate_format="0-1000")
        res = asyncio.run(grounder.locate("button", b"png", (1280, 720), (1920, 1080)))
        self.assertIsNotNone(res)
        self.assertEqual(res.x, 960)
        self.assertEqual(res.y, 540)


if __name__ == "__main__":
    unittest.main()
