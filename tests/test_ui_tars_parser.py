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


class TestNativeBoxTokens(unittest.TestCase):
    """UI-TARS's own coordinate syntax wraps points in box tokens.

    The original regex did not match it, so every click in the model's native
    format fell through to "unrecognized" and burned a retry turn.
    """

    def test_box_token_click(self):
        text = "Thought: click it\nAction: click(start_box='<|box_start|>(345,678)<|box_end|>')"
        parsed = UITarsParser.parse_response(text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["params"]["x"], 345)
        self.assertEqual(parsed["params"]["y"], 678)

    def test_box_token_without_quotes(self):
        parsed = UITarsParser.parse_response("Action: click(start_box=<|box_start|>(10,20)<|box_end|>)")
        self.assertIsNotNone(parsed)
        self.assertEqual((parsed["params"]["x"], parsed["params"]["y"]), (10, 20))

    def test_positional_coordinates(self):
        parsed = UITarsParser.parse_response("Action: click((100, 200))")
        self.assertIsNotNone(parsed)
        self.assertEqual((parsed["params"]["x"], parsed["params"]["y"]), (100, 200))


class TestClickKindPreserved(unittest.TestCase):
    """double / right click both used to collapse into a plain left click."""

    def test_double_click_sets_click_count(self):
        parsed = UITarsParser.parse_response("Action: left_double_click(start_box='(50,60)')")
        self.assertEqual(parsed["action"], "cua_click")
        self.assertEqual(parsed["params"]["click_count"], 2)

    def test_right_click_routes_to_secondary_action(self):
        parsed = UITarsParser.parse_response("Action: right_click(start_box='(50,60)')")
        self.assertEqual(parsed["action"], "cua_secondary_action")
        self.assertEqual(parsed["params"]["button"], "right")

    def test_named_right_click(self):
        parsed = UITarsParser.parse_response("Action: right_click(target='Desktop')")
        self.assertEqual(parsed["action"], "cua_secondary_action")
        self.assertEqual(parsed["params"]["target_name"], "Desktop")

    def test_plain_click_is_single(self):
        parsed = UITarsParser.parse_response("Action: click(start_box='(1,2)')")
        self.assertEqual(parsed["params"]["click_count"], 1)


class TestPreviouslyUnparsedActions(unittest.TestCase):
    """scroll / drag / wait / positional hotkey were all wasted retry turns."""

    def test_scroll_down(self):
        parsed = UITarsParser.parse_response(
            "Action: scroll(start_box='(500,400)', direction='down')"
        )
        self.assertEqual(parsed["action"], "cua_scroll")
        self.assertEqual(parsed["params"]["scrollY"], 500)
        self.assertEqual(parsed["params"]["x"], 500)

    def test_scroll_up_is_negative(self):
        parsed = UITarsParser.parse_response("Action: scroll(start_box='(0,0)', direction='up')")
        self.assertEqual(parsed["params"]["scrollY"], -500)

    def test_scroll_defaults_to_down(self):
        parsed = UITarsParser.parse_response("Action: scroll(start_box='(0,0)')")
        self.assertEqual(parsed["params"]["scrollY"], 500)

    def test_drag_between_two_points(self):
        parsed = UITarsParser.parse_response(
            "Action: drag(start_box='(10,20)', end_box='(30,40)')"
        )
        self.assertEqual(parsed["action"], "cua_drag")
        self.assertEqual(parsed["params"]["from_x"], 10)
        self.assertEqual(parsed["params"]["to_y"], 40)

    def test_drag_with_one_point_is_rejected(self):
        self.assertIsNone(UITarsParser.parse_response("Action: drag(start_box='(10,20)')"))

    def test_wait_becomes_a_harmless_observation(self):
        parsed = UITarsParser.parse_response("Action: wait()")
        self.assertEqual(parsed["action"], "cua_screenshot")
        self.assertFalse(parsed["is_completed"])

    def test_positional_hotkey(self):
        parsed = UITarsParser.parse_response("Action: hotkey('ctrl a')")
        self.assertEqual(parsed["action"], "cua_press_key")
        self.assertEqual(parsed["params"]["key"], "ctrl+a")

    def test_hotkey_with_plus_is_left_alone(self):
        parsed = UITarsParser.parse_response("Action: hotkey('ctrl+c')")
        self.assertEqual(parsed["params"]["key"], "ctrl+c")

    def test_type_with_content_argument(self):
        # `content` is UI-TARS's own argument name for typed text.
        parsed = UITarsParser.parse_response("Action: type(content='hello world')")
        self.assertEqual(parsed["action"], "cua_type_text")
        self.assertEqual(parsed["params"]["text"], "hello world")

    def test_finished_with_content_argument(self):
        parsed = UITarsParser.parse_response("Action: finished(content='All set')")
        self.assertTrue(parsed["is_completed"])
        self.assertEqual(parsed["final_response"], "All set")


class TestParserRejection(unittest.TestCase):
    def test_prose_is_rejected(self):
        self.assertIsNone(UITarsParser.parse_response("I will now click the button for you."))

    def test_empty_input(self):
        self.assertIsNone(UITarsParser.parse_response(""))
        self.assertIsNone(UITarsParser.parse_response(None))

    def test_thought_without_action_is_rejected(self):
        self.assertIsNone(UITarsParser.parse_response("Thought: I should probably do something."))


if __name__ == "__main__":
    unittest.main()
