"""Unit tests for UI-TARS Action Parser and LlamaServerManager."""

import unittest
from grace.agent.ui_tars_parser import UITarsParser


class TestUITarsParser(unittest.TestCase):
    """Test suite for parsing UI-TARS 7B output responses into Grace intents."""

    def test_parse_click_coordinates(self):
        text = "Thought: Need to click search box\nAction: click(start_box='(929, 72)')"
        parsed = UITarsParser.parse_response(text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["action"], "cua_click")
        self.assertEqual(parsed["params"]["x"], 929)
        self.assertEqual(parsed["params"]["y"], 72)
        self.assertFalse(parsed["is_completed"])

    def test_parse_click_target_name(self):
        text = "Action: click(target='Play')"
        parsed = UITarsParser.parse_response(text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["action"], "cua_click")
        self.assertEqual(parsed["params"]["target_name"], "Play")

    def test_parse_type_text(self):
        text = "Action: type(text='Signal by Home')"
        parsed = UITarsParser.parse_response(text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["action"], "cua_type_text")
        self.assertEqual(parsed["params"]["text"], "Signal by Home")

    def test_parse_press_key(self):
        text = "Action: press_key(key='Return')"
        parsed = UITarsParser.parse_response(text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["action"], "cua_press_key")
        self.assertEqual(parsed["params"]["key"], "Return")

    def test_parse_finished(self):
        text = "Thought: Playback started\nAction: finished(response='Playing Signal by Home')"
        parsed = UITarsParser.parse_response(text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["action"], "converse")
        self.assertTrue(parsed["is_completed"])
        self.assertEqual(parsed["final_response"], "Playing Signal by Home")


if __name__ == "__main__":
    unittest.main()
