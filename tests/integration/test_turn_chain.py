"""The activation turn and the follow-up turn must route the same way.

`main.py` used to hold two near-copies of the intent -> router -> dispatch
chain: one in `_handle_activation_turn`, one in `_handle_followup_transcript`.
They were maintained by hand and had already drifted - the safety-confirmation
park was added to one before the other, and the agentic pre-execution step
exists in both only because someone remembered.

The activation copy is now extracted into `_route_transcript`, shared by the
wake-word turn and by the typed-command entrypoint (`--text`), so those two
cannot drift from each other any more - they are literally one body. That
still leaves the follow-up window holding its own hand-maintained copy, so
this class's remaining job is to hold that copy to the same decisions the
shared body makes.

Nothing tests the follow-up copy at all, so a fix applied to the shared path
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
# The extracted body: one intent -> router -> dispatch chain shared by the
# wake-word path and the typed path (`--text`). Hold the follow-up copy to it.
ROUTE = "_route_transcript"
FOLLOWUP = "_handle_followup_transcript"
TYPED = "_handle_text_activation"

# Every caller that must route through the shared body instead of a copy.
ROUTERS = (ACTIVATION, TYPED)


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
        # Not "self.agent_loop.run" directly any more (R5): _run_agent_goal
        # arms the wake-word cancel watch around that call so a running goal
        # can be stopped by voice.
        "self._run_agent_goal",
    ])
    def test_both_paths_make_the_call(self, call):
        assert call in calls_in(ROUTE), f"{ROUTE} no longer calls {call}"
        for name in ROUTERS:
            assert "self._route_transcript" in calls_in(name), (
                f"{name} stopped routing through the shared body; a typed string "
                f"and a heard string could then take different paths"
            )
        assert call in calls_in(FOLLOWUP), (
            f"{FOLLOWUP} does not call {call}, so a follow-up utterance takes a "
            f"different route from the same words said after the wake word"
        )

    def test_both_paths_resolve_a_pending_confirmation_first(self):
        # A "yes" said in the follow-up window is the *normal* way a
        # confirmation gets answered - the activation path only sees one if the
        # user waited out the window and said the wake word again. The typed
        # path sees it whenever the entrypoint's stdin reader gets the answer.
        for name in (ROUTE, FOLLOWUP):
            assert "self._resolve_pending_confirmation" in calls_in(name), name
        for name in ROUTERS:
            assert "self._route_transcript" in calls_in(name), name

    def test_both_paths_park_a_refused_dispatch(self):
        # Without this the fast path speaks the confirmation question and then
        # throws away the action it was asking about, so "yes" answers nothing.
        for name in (ROUTE, FOLLOWUP):
            assert "self._park_if_confirmation_required" in calls_in(name), name
        for name in ROUTERS:
            assert "self._route_transcript" in calls_in(name), name

    def test_both_paths_pre_execute_the_same_setup_intents(self):
        # `open_app` and `cua_launch` run before the loop starts, so the agent
        # observes the app it was asked about rather than the desktop. The
        # shared body and the follow-up copy must each keep the rule.
        for tool in ("open_app", "cua_launch"):
            assert tool in strings_in(ROUTE), f"{ROUTE} lost {tool}"
            assert tool in strings_in(FOLLOWUP), (
                f"{FOLLOWUP} does not pre-execute {tool}, so an agentic "
                f"follow-up starts by looking at the wrong window"
            )

    def test_both_paths_handle_a_rate_limit(self):
        for name in (ROUTE, FOLLOWUP):
            handlers = [
                _dotted(node.type)
                for node in ast.walk(function_body(name))
                if isinstance(node, ast.ExceptHandler) and node.type is not None
            ]
            assert "RateLimitError" in handlers, (
                f"{name} does not catch RateLimitError; a rate-limited turn "
                f"there becomes an unhandled exception instead of an apology"
            )
        for name in ROUTERS:
            assert "self._route_transcript" in calls_in(name), name
