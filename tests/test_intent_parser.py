"""Test intent parser."""

import asyncio
import os
import types
from unittest.mock import MagicMock

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.intent.parser import IntentParser, Intent, IntentParseError
from grace.intent.prompt import get_system_prompt
from grace.intent.tools import site_url
from grace.main import GraceApp


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

    def test_parse_flat_params_open_app(self):
        """Holo emitted flat params `{"tool": "open_app", "name": "Edge"}`."""
        json_str = '{"tool": "open_app", "name": "Edge"}'
        intent = self.parser.parse(json_str)
        assert intent.tool == "open_app"
        assert intent.params["name"] == "Edge"

    def test_parse_flat_and_nested_params_merged(self):
        """Flat parameters merge with nested params without overwriting existing nested keys."""
        json_str = '{"tool": "open_app", "params": {"url": "https://example.com"}, "name": "Edge"}'
        intent = self.parser.parse(json_str)
        assert intent.tool == "open_app"
        assert intent.params["name"] == "Edge"
        assert intent.params["url"] == "https://example.com"

    def test_flat_param_cannot_override_tool(self):
        """A flat parameter cannot override the tool field."""
        json_str = '{"tool": "open_app", "name": "Edge"}'
        intent = self.parser.parse(json_str)
        assert intent.tool == "open_app"
        assert "tool" not in intent.params


class TestSiteUrl:
    """Known-site URL resolution for the "open X in my browser" shortcut."""

    def test_youtube_music_resolves_to_music_url(self):
        assert site_url("youtube music") == "https://music.youtube.com"

    def test_site_url_is_case_insensitive_and_trims(self):
        assert site_url("  YouTube Music  ") == "https://music.youtube.com"
        assert site_url("YOUTUBE") == "https://www.youtube.com"

    def test_unknown_site_returns_none(self):
        assert site_url("no such site xyzzy") is None
        assert site_url("") is None


class TestIntentPromptKeepsBrowserUrl:
    def test_prompt_guides_open_app_with_url_for_browser(self):
        prompt = get_system_prompt()
        assert "open_app" in prompt
        assert "url" in prompt
        assert "address bar" in prompt


class _FakeWS:
    def __init__(self):
        self.emitted = []

    async def emit(self, event):
        self.emitted.append(event)


class _FakeDispatcher:
    def __init__(self):
        self.calls = []

    async def execute(self, intent, confirmed=False):
        self.calls.append(intent)
        return {
            "status": "ok",
            "text": f"I've opened {intent.params.get('name')} in your browser.",
        }


class _Stage:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return False


class _Trace:
    def stage(self, name):
        return _Stage()


class _Stub:
    def __init__(self):
        self.ws_server = _FakeWS()
        self.dispatcher = _FakeDispatcher()
        self.spoken = None

    async def _speak_response(self, text):
        self.spoken = text


class TestBrowserShortcut:
    """Task 12: "open X in my browser" opens the site URL directly, before the
    agent loop, so it never drives the address bar."""

    def _make_stub(self):
        return _Stub()

    def test_shortcut_opens_site_url_directly(self):
        stub = self._make_stub()
        method = types.MethodType(GraceApp._open_site_in_browser_shortcut, stub)
        result = asyncio.run(method(_Trace(), "open youtube music in my browser"))

        assert result is True
        assert len(stub.dispatcher.calls) == 1
        intent = stub.dispatcher.calls[0]
        assert intent.tool == "open_app"
        assert intent.params["name"] == "youtube music"
        assert intent.params["url"] == "https://music.youtube.com"
        # The shortcut answers directly instead of running the agent loop.
        assert stub.spoken is not None
        assert any(e["type"] == "ConversationFinished" for e in stub.ws_server.emitted)

    def test_shortcut_unknown_site_falls_through(self):
        stub = self._make_stub()
        method = types.MethodType(GraceApp._open_site_in_browser_shortcut, stub)
        result = asyncio.run(method(_Trace(), "open no such site xyzzy in my browser"))

        assert result is None  # fall through to normal routing
        assert stub.dispatcher.calls == []

    def test_shortcut_only_matches_whole_browser_request(self):
        stub = self._make_stub()
        method = types.MethodType(GraceApp._open_site_in_browser_shortcut, stub)
        # Multi-step: opens the site, then searches - must reach the agent loop,
        # not be shortcut to just opening the site.
        result = asyncio.run(
            method(
                _Trace(),
                "open youtube music in my browser, search for the song signal by home, and play it",
            )
        )
        assert result is None
        assert stub.dispatcher.calls == []

    def test_shortcut_does_not_match_plain_open_app(self):
        stub = self._make_stub()
        method = types.MethodType(GraceApp._open_site_in_browser_shortcut, stub)
        result = asyncio.run(method(_Trace(), "open calculator"))
        assert result is None
        assert stub.dispatcher.calls == []
