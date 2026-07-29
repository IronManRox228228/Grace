"""Test intent parser."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.intent.parser import IntentParser, Intent, IntentParseError


class TestIntentParser:
    """Test suite for IntentParser."""

    def setup_method(self):
        self.parser = IntentParser()

    def test_parse_open_app(self):
        """Parse an open_app intent."""
        json_str = '{"tool": "open_app", "params": {"name": "Calculator"}}'
        intent = self.parser.parse(json_str)
        assert intent.tool == "open_app"
        assert intent.params["name"] == "Calculator"
        assert not intent.is_conversation

    def test_parse_close_app(self):
        """Parse a close_app intent."""
        json_str = '{"tool": "close_app", "params": {"name": "Microsoft Edge"}}'
        intent = self.parser.parse(json_str)
        assert intent.tool == "close_app"
        assert intent.params["name"] == "Microsoft Edge"

    def test_parse_converse(self):
        """Parse a converse intent."""
        json_str = '{"tool": "converse", "params": {"response": "Hello!"}}'
        intent = self.parser.parse(json_str)
        assert intent.tool == "converse"
        assert intent.is_conversation
        assert self.parser.extract_response_text(intent) == "Hello!"

    def test_parse_adjust_volume(self):
        """Parse an adjust_volume intent."""
        json_str = '{"tool": "adjust_volume", "params": {"amount": 5, "mode": "increase"}}'
        intent = self.parser.parse(json_str)
        assert intent.tool == "adjust_volume"
        assert intent.params["amount"] == 5
        assert intent.params["mode"] == "increase"

    def test_parse_lock_computer(self):
        """Parse a lock_computer intent."""
        json_str = '{"tool": "lock_computer", "params": {}}'
        intent = self.parser.parse(json_str)
        assert intent.tool == "lock_computer"
        assert intent.params == {}

    def test_parse_with_markdown_fences(self):
        """Parse intent with markdown code fences."""
        json_str = '```json\n{"tool": "open_calculator", "params": {}}\n```'
        intent = self.parser.parse(json_str)
        assert intent.tool == "open_calculator"
        assert intent.params == {}

    def test_parse_with_backticks(self):
        """Parse intent with just backticks (no json tag)."""
        json_str = '```\n{"tool": "search_files", "params": {"query": "notes"}}\n```'
        intent = self.parser.parse(json_str)
        assert intent.tool == "search_files"
        assert intent.params["query"] == "notes"

    def test_parse_missing_tool_raises(self):
        """Parsing without 'tool' field should raise."""
        json_str = '{"params": {}}'
        try:
            self.parser.parse(json_str)
            assert False, "Should have raised IntentParseError"
        except IntentParseError:
            pass

    def test_parse_invalid_tool_raises(self):
        """Parsing with unknown tool should raise."""
        json_str = '{"tool": "invalid_tool", "params": {}}'
        try:
            self.parser.parse(json_str)
            assert False, "Should have raised IntentParseError"
        except IntentParseError:
            pass

    def test_parse_invalid_json_raises(self):
        """Parsing invalid JSON should raise."""
        json_str = '{tool: open_app}'  # Invalid JSON
        try:
            self.parser.parse(json_str)
            assert False, "Should have raised IntentParseError"
        except IntentParseError:
            pass

    def test_extract_response_converse(self):
        """Extract response from converse intent."""
        intent = Intent(tool="converse", params={"response": "Hello there!"})
        assert self.parser.extract_response_text(intent) == "Hello there!"

    def test_extract_response_non_converse(self):
        """Non-converse intent should return empty string."""
        intent = Intent(tool="open_app", params={"name": "Calculator"})
        assert self.parser.extract_response_text(intent) == ""

    def test_intent_needs_cua(self):
        """Test CUA tool detection."""
        cua_intent = Intent(tool="cua_click", params={})
        assert cua_intent.needs_cua is True

        sys_intent = Intent(tool="open_app", params={})
        assert sys_intent.needs_cua is False
