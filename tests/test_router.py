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


class TestFastPathIsOptIn:
    """Agentic is the default; a request has to be one short command to skip it.

    The old rule promoted a fast-path intent to the loop only if the utterance
    contained a listed verb. "search" was not listed, so "open chemistry group
    in WhatsApp and search for the PDF" ran open_app, dropped the rest, and
    said "I've opened WhatsApp."
    """

    def test_the_search_request_that_used_to_be_truncated(self):
        intent = Intent(tool="open_app", params={"name": "WhatsApp"})
        assert route(
            "open chemistry group in WhatsApp and search for the PDF called electrochemistry",
            intent,
        ) == TaskComplexity.AGENTIC_GOAL

    @pytest.mark.parametrize("text", [
        "open Edge and search for cat videos",
        "open notepad then save it",
        "open the folder also rename the file",
        "open Spotify next play my playlist",
    ])
    def test_a_second_clause_always_escalates(self, text):
        intent = Intent(tool="open_app", params={"name": "X"})
        assert route(text, intent) == TaskComplexity.AGENTIC_GOAL

    def test_verbs_no_list_would_have_contained_still_escalate(self):
        # The point of the rewrite: this works without "download" being known.
        intent = Intent(tool="open_app", params={"name": "Edge"})
        assert route("open Edge and download the attachment", intent) == TaskComplexity.AGENTIC_GOAL

    def test_a_long_request_escalates_even_without_a_conjunction(self):
        # Speech drops conjunctions; the intent model still answers with the
        # first tool it recognises.
        intent = Intent(tool="open_app", params={"name": "WhatsApp"})
        assert route(
            "open whatsapp find the electrochemistry pdf in my chemistry group",
            intent,
        ) == TaskComplexity.AGENTIC_GOAL

    @pytest.mark.parametrize("text,tool", [
        ("open notepad", "open_app"),
        ("close notepad", "close_app"),
        ("turn the volume up a bit", "adjust_volume"),
        ("lock my computer", "lock_computer"),
        ("find my tax return", "search_files"),
        ("list my open windows", "cua_list_windows"),
        ("get rid of the draft file", "delete_file"),
        ("open the budget spreadsheet", "open_file"),
    ])
    def test_genuinely_atomic_commands_keep_the_fast_path(self, text, tool):
        # Every fast-path utterance in the recorded corpus. Escalating these
        # would trade the whole latency benefit for nothing.
        assert route(text, Intent(tool=tool, params={})) == TaskComplexity.FAST_PATH


class TestMultiStepBackstop:
    """The intent model folds two-action sentences into one JSON object.

    On the live Holo model (recorded in PROGRESS.md), "click the search box
    and type hello" was answered with `converse` ("I'll click the search box
    and then type hello...") and the turn went to chat, stranding the second
    action. "read my report pdf to me" was answered with `search_files` and
    the turn went down the fast path, truncating the request. The backstop
    counts action clauses: two or more of them beat a `converse` label, and
    beat the fast path even when the chain is comma-joined.
    """

    # The recorded agentic_read_pdf tape utterance, with the intent the live
    # model actually produced for it.
    def test_a_search_intent_with_no_search_ask_goes_agentic(self):
        intent = Intent(tool="search_files", params={"query": "report.pdf"})
        assert route(
            "read my report pdf to me", intent,
        ) == TaskComplexity.AGENTIC_GOAL

    def test_a_search_intent_that_matches_the_ask_keeps_the_fast_path(self):
        intent = Intent(tool="search_files", params={"query": "tax documents"})
        assert route("find my tax documents", intent) == TaskComplexity.FAST_PATH
        assert route("search for my tax documents", intent) == TaskComplexity.FAST_PATH

    # The recorded agentic_two_step tape utterance, with the intent the live
    # model actually produced for it.
    def test_converse_intent_on_the_tape_utterance_goes_agentic(self):
        intent = Intent(tool="converse", params={},
                        response="I'll click on a search box and type 'hello'.")
        assert route(
            "click the search box and type hello", intent,
        ) == TaskComplexity.AGENTIC_GOAL

    def test_comma_joined_actions_escalate_a_fast_path_intent(self):
        # No separator word this time: commas alone have to count.
        intent = Intent(tool="open_app", params={"name": "Notepad"})
        assert route(
            "open notepad, type hello", intent,
        ) == TaskComplexity.AGENTIC_GOAL

    def test_the_demo_multi_app_command_goes_agentic(self):
        intent = Intent(tool="open_app", params={"name": "Edge"})
        assert route(
            "open edge, go to youtube and search lofi", intent,
        ) == TaskComplexity.AGENTIC_GOAL

    def test_greeting_before_two_actions_still_goes_agentic(self):
        # No intent parsed; the greeting keyword would have eaten it.
        assert route(
            "hey grace, open notepad and reply to my email",
        ) == TaskComplexity.AGENTIC_GOAL

    def test_one_action_clause_interrupts_nothing(self):
        # The boundary: exactly two action clauses trigger the backstop. A
        # single action is what the fast path and conversation are for.
        intent = Intent(tool="converse", params={}, response="Sure.")
        assert route("thanks grace, open notepad", intent) == TaskComplexity.CONVERSATION

    @pytest.mark.parametrize("text", [
        "thanks and goodbye",
        "tell me a haiku and then say goodnight",
        "what can you do and how do you work",
        "i feel great today, how about you?",
    ])
    def test_chit_chat_with_conjunctive_bits_stays_conversational(self, text):
        intent = Intent(tool="converse", params={}, response="Hi!")
        assert route(text, intent) == TaskComplexity.CONVERSATION

    def test_no_intent_greetings_are_untouched_by_the_backstop(self):
        # A keyword greeting with no parsed intent is still conversation;
        # the backstop only fires on action clauses it can find.
        assert route("what can you do") == TaskComplexity.CONVERSATION



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
