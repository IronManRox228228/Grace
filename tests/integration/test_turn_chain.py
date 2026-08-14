"""The activation turn and the follow-up turn must route the same way.

`main.py` holds two near-copies of the intent -> router -> dispatch chain: one
in `_handle_activation_turn`, one in `_handle_followup_transcript`. They are
maintained by hand and have already drifted - the safety-confirmation park was
added to one before the other, and the agentic pre-execution step exists in both
only because someone remembered.

Nothing tests the follow-up copy at all, so a fix applied to the activation path
and not the follow-up path is invisible until a user says something in the ten
seconds after a turn - which, for a user who cannot reach a keyboard to retry,
is most of how they talk to Grace.

This does not assert the two are textually identical; they legitimately differ
in what they return and in the events they emit around the work. It asserts they
make the same *decisions*: same router call, same pre-execution rule, same
confirmation handling, same dispatch.
"""

import ast
import os

import pytest

pytestmark = pytest.mark.integration

MAIN = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "grace", "main.py")
)

ACTIVATION = "_handle_activation_turn"
FOLLOWUP = "_handle_followup_transcript"


def function_body(name: str) -> ast.AST:
    with open(MAIN, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=MAIN)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} no longer exists in main.py")


def calls_in(name: str) -> set[str]:
    """Dotted names of everything the function calls, however deeply nested."""
    found: set[str] = set()
    for node in ast.walk(function_body(name)):
        if isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            if dotted:
                found.add(dotted)
    return found


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def strings_in(name: str) -> set[str]:
    return {
        node.value for node in ast.walk(function_body(name))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


class TestBothTurnsMakeTheSameDecisions:

    @pytest.mark.parametrize("call", [
        "CapabilityRouter.classify",
        "self.intent_parser.parse",
        "self.gemma.generate_intent",
        "self.dispatcher.execute",
        "self.agent_loop.run",
    ])
    def test_both_paths_make_the_call(self, call):
        assert call in calls_in(ACTIVATION), f"{ACTIVATION} no longer calls {call}"
        assert call in calls_in(FOLLOWUP), (
            f"{FOLLOWUP} does not call {call}, so a follow-up utterance takes a "
            f"different route from the same words said after the wake word"
        )

    def test_both_paths_resolve_a_pending_confirmation_first(self):
        # A "yes" said in the follow-up window is the *normal* way a
        # confirmation gets answered - the activation path only sees one if the
        # user waited out the window and said the wake word again.
        for name in (ACTIVATION, FOLLOWUP):
            assert "self._resolve_pending_confirmation" in calls_in(name), name

    def test_both_paths_park_a_refused_dispatch(self):
        # Without this the fast path speaks the confirmation question and then
        # throws away the action it was asking about, so "yes" answers nothing.
        for name in (ACTIVATION, FOLLOWUP):
            assert "self._park_if_confirmation_required" in calls_in(name), name

    def test_both_paths_pre_execute_the_same_setup_intents(self):
        # `open_app` and `cua_launch` run before the loop starts, so the agent
        # observes the app it was asked about rather than the desktop. The two
        # copies kept their own literal tuples of tool names.
        for tool in ("open_app", "cua_launch"):
            assert tool in strings_in(ACTIVATION), f"{ACTIVATION} lost {tool}"
            assert tool in strings_in(FOLLOWUP), (
                f"{FOLLOWUP} does not pre-execute {tool}, so an agentic "
                f"follow-up starts by looking at the wrong window"
            )

    def test_both_paths_handle_a_rate_limit(self):
        for name in (ACTIVATION, FOLLOWUP):
            handlers = [
                _dotted(node.type)
                for node in ast.walk(function_body(name))
                if isinstance(node, ast.ExceptHandler) and node.type is not None
            ]
            assert "RateLimitError" in handlers, (
                f"{name} does not catch RateLimitError; a rate-limited turn "
                f"there becomes an unhandled exception instead of an apology"
            )
