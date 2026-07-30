"""Tests for the capability router.

The router used to substring-match ~80 keywords *before* looking at the parsed
intent, and defaulted to AGENTIC_GOAL. The practical result was that saying
hello took a screenshot, an OCR pass and a vision call.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.intent.parser import Intent
from grace.intent.router import CapabilityRouter, TaskComplexity


def route(text, intent=None):
    return CapabilityRouter.classify(text, intent)


class TestIntentIsTrustedFirst:
    def test_single_deterministic_tool_takes_the_fast_path(self):
        intent = Intent(tool="open_app", params={"name": "Notepad"})
        assert route("open Notepad", intent) == TaskComplexity.FAST_PATH

    def test_volume_takes_the_fast_path(self):
        intent = Intent(tool="adjust_volume", params={"amount": 20, "mode": "increase"})
        assert route("turn the volume up", intent) == TaskComplexity.FAST_PATH

    def test_screen_manipulation_stays_agentic(self):
        intent = Intent(tool="cua_click", params={"x": 1, "y": 2})
        assert route("click the button", intent) == TaskComplexity.AGENTIC_GOAL

    def test_pdf_work_stays_agentic(self):
        intent = Intent(tool="summarize_pdf", params={"path": "a.pdf"})
        assert route("summarize this", intent) == TaskComplexity.AGENTIC_GOAL

    def test_chained_request_overrides_the_fast_path(self):
        # "open YouTube" alone is one step; "and play something" is not.
        intent = Intent(tool="open_app", params={"name": "YouTube"})
        assert route("open YouTube and play a lo-fi video", intent) == TaskComplexity.AGENTIC_GOAL


class TestConversationShortCircuit:
    @pytest.mark.parametrize("text", ["hello", "hi there", "hey Grace", "thanks", "goodbye"])
    def test_greetings_do_not_take_the_agentic_path(self, text):
        assert route(text) == TaskComplexity.CONVERSATION

    def test_conversational_intent_is_honoured(self):
        intent = Intent(tool="converse", params={}, response="It's sunny.")
        assert route("what's the weather", intent) == TaskComplexity.CONVERSATION


class TestWordBoundaries:
    """Substring matching turned innocuous words into agentic triggers."""

    @pytest.mark.parametrize("text", [
        "what happened yesterday",   # contains "happen" -> "open"
        "that was a typo",           # contains "typo"   -> "type"
        "I feel happy today",        # contains "happy"  -> "app"
        "explain it plainly",        # contains "plain"
    ])
    def test_incidental_substrings_do_not_force_agentic(self, text):
        intent = Intent(tool="converse", params={}, response="ok")
        assert route(text, intent) == TaskComplexity.CONVERSATION

    def test_real_action_words_still_match(self):
        assert route("click the blue button") == TaskComplexity.AGENTIC_GOAL
        assert route("scroll down the page") == TaskComplexity.AGENTIC_GOAL


class TestDefaults:
    def test_unknown_request_defaults_to_agentic(self):
        # Safer than guessing: the loop can look at the screen and find out.
        assert route("do the thing with the stuff") == TaskComplexity.AGENTIC_GOAL

    def test_empty_input_does_not_raise(self):
        assert route("") == TaskComplexity.AGENTIC_GOAL
        assert route(None) == TaskComplexity.AGENTIC_GOAL

    def test_unknown_tool_falls_through_to_text(self):
        intent = Intent(tool="some_future_tool", params={})
        assert route("click something", intent) == TaskComplexity.AGENTIC_GOAL
