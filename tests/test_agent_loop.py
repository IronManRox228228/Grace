"""Tests for the rewritten agentic loop.

Covers the three things that were broken rather than merely slow: completion
was "verified" by a substring check that matched the word "which", a step
parked for safety confirmation could never be resumed, and a failed step gave
the model no signal that it had failed.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.agent.loop import AgentLoop
from grace.agent.memory import AgentMemory
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.agent.planner import PlannedStep


def snapshot(title="Untitled - Notepad", graph=None):
    return ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title=title, class_name="X", rect=(0, 0, 800, 600)),
        ocr_lines=[],
        width=1920,
        height=1080,
        graph=graph,
    )


def make_loop(responses, dispatch_result=None, **kwargs):
    gemma = AsyncMock()
    gemma.generate_text.side_effect = responses
    dispatcher = AsyncMock()
    dispatcher.execute.return_value = dispatch_result or {"status": "ok"}
    perception = MagicMock()
    perception.capture_snapshot.return_value = snapshot()
    perception.capture_snapshot_async.return_value = snapshot()
    loop = AgentLoop(gemma=gemma, dispatcher=dispatcher, perception=perception, **kwargs)
    return loop, gemma, dispatcher


DONE = '{"action": "converse", "params": {}, "is_completed": true, "final_response": "All done."}'


class TestIterationCap:
    def test_default_cap_is_twelve_not_a_hundred(self):
        loop, _, _ = make_loop([])
        assert loop._max_iterations == 12

    def test_cap_is_respected(self):
        # A model that never finishes must stop, not run for 100 screenshots.
        click = '{"action": "cua_click", "params": {"x": 1, "y": 2}, "expect": "something"}'
        loop, _, _ = make_loop([click] * 20)
        res = asyncio.run(loop.run(user_goal="click forever", max_iterations=3))
        assert res["status"] in ("max_iterations_reached", "planner_budget_exceeded")
        assert len(res["steps"]) <= 3


class TestVerification:
    """The guard used to return True for any goal containing the substring 'hi'."""

    @pytest.mark.parametrize("goal", ["which file is open", "make this white", "finish this"])
    def test_substring_hi_no_longer_counts_as_conversational(self, goal):
        loop, _, _ = make_loop([])
        verified, hint = loop._verify_goal_completion(goal, snapshot(), AgentMemory(goal))
        assert verified is False
        assert "Goal observation check" in hint

    @pytest.mark.parametrize("goal", [
        "what is the weather", "who is Ada Lovelace", "hi there",
        "tell me a joke", "explain recursion", "calculate 2 plus 2",
    ])
    def test_real_questions_still_pass(self, goal):
        loop, _, _ = make_loop([])
        verified, hint = loop._verify_goal_completion(goal, snapshot(), AgentMemory(goal))
        assert verified is True and hint is None

    def test_completion_allowed_after_an_interactive_step(self):
        loop, _, _ = make_loop([])
        memory = AgentMemory("open notepad")
        memory.add_step("t", "open_app", {"name": "Notepad"}, {"status": "ok"})
        assert loop._verify_goal_completion("open notepad", snapshot(), memory)[0] is True

    def test_completion_blocked_when_only_observations_happened(self):
        loop, _, _ = make_loop([])
        memory = AgentMemory("open notepad")
        memory.add_step("t", "cua_list_windows", {}, {"status": "ok"})
        assert loop._verify_goal_completion("open notepad", snapshot(), memory)[0] is False

    def test_rejected_completion_keeps_the_loop_going(self):
        # First the model claims done with nothing performed; it must be pushed
        # back, act, and only then be allowed to finish.
        loop, _, dispatcher = make_loop([
            DONE,
            '{"action": "open_app", "params": {"name": "Notepad"}, "expect": "Notepad is open"}',
            DONE,
        ])
        res = asyncio.run(loop.run(user_goal="open Notepad"))
        assert res["status"] == "ok"
        assert res["final_response"] == "All done."
        assert any(s["action"] == "open_app" for s in res["steps"])


class TestExpectationFeedback:
    """A failed step must come back as an explicit correction, not a silent retry."""

    def test_failure_is_reported_to_the_planner(self):
        loop, _, _ = make_loop([])
        step = PlannedStep(action="cua_type_text", params={}, expect="the search box has my query")
        result = {"status": "ok", "result": {"ok": False, "error": "Focus is on 'Address bar'"}}
        note = loop._expectation_note(step, result, snapshot())
        assert "IT FAILED" in note
        assert "Address bar" in note
        assert "Do not repeat this action unchanged" in note

    def test_success_reports_the_new_state(self):
        loop, _, _ = make_loop([])
        step = PlannedStep(action="cua_click", params={}, expect="the box is focused")
        note = loop._expectation_note(step, {"status": "ok", "result": {"message": "Clicked"}},
                                      snapshot(title="YouTube - Edge"))
        assert "the box is focused" in note
        assert "YouTube - Edge" in note
        assert "IT FAILED" not in note

    def test_error_status_counts_as_failure(self):
        loop, _, _ = make_loop([])
        step = PlannedStep(action="open_app", params={}, expect="app opens")
        note = loop._expectation_note(step, {"status": "error", "error": "not found"}, snapshot())
        assert "IT FAILED" in note and "not found" in note

    def test_focused_element_is_included(self):
        graph = MagicMock()
        focused = MagicMock(id=4, role="searchbox", name="Search", placeholder="", frame="page")
        graph.focused.return_value = focused
        loop, _, _ = make_loop([])
        note = loop._expectation_note(
            PlannedStep(action="cua_click", params={}, expect="focus"),
            {"status": "ok", "result": {}},
            snapshot(graph=graph),
        )
        assert "frame=page" in note and "Search" in note

    def test_no_note_on_the_first_step(self):
        loop, _, _ = make_loop([])
        assert loop._expectation_note(None, None, snapshot()) == ""

    def test_note_reaches_the_next_planner_call(self):
        loop, gemma, _ = make_loop([
            '{"action": "cua_click", "params": {"x": 1, "y": 2}, "expect": "the box is focused"}',
            DONE,
        ], dispatch_result={"status": "ok", "result": {"ok": False, "error": "nothing there"}})
        asyncio.run(loop.run(user_goal="click the box"))
        second_prompt = gemma.generate_text.call_args_list[1].kwargs["prompt"]
        assert "IT FAILED" in second_prompt


class TestSafetyResumption:
    """SafetyGuard parked the step and nothing could ever un-park it."""

    def test_dangerous_action_pauses_and_is_recorded(self):
        loop, _, _ = make_loop(['{"action": "delete_file", "params": {"name": "notes.txt"}}'])
        res = asyncio.run(loop.run(user_goal="delete notes.txt"))
        assert res["status"] == "safety_confirmation_required"
        assert "notes.txt" in res["confirmation_prompt"]
        assert loop.has_pending_confirmation is True

    def test_yes_executes_the_parked_action(self):
        loop, _, dispatcher = make_loop([
            '{"action": "delete_file", "params": {"name": "notes.txt"}}',
            DONE,
        ])

        async def drive():
            await loop.run(user_goal="delete notes.txt")
            return await loop.resume_pending(approved=True)

        res = asyncio.run(drive())
        assert res["status"] == "ok"
        executed = [c.args[0].tool for c in dispatcher.execute.call_args_list]
        assert "delete_file" in executed
        assert loop.has_pending_confirmation is False

    def test_no_cancels_without_executing(self):
        loop, _, dispatcher = make_loop(['{"action": "delete_file", "params": {"name": "notes.txt"}}'])

        async def drive():
            await loop.run(user_goal="delete notes.txt")
            return await loop.resume_pending(approved=False)

        res = asyncio.run(drive())
        assert res["status"] == "ok"
        assert "won't" in res["final_response"]
        assert dispatcher.execute.call_count == 0

    def test_resume_without_anything_pending(self):
        loop, _, _ = make_loop([])
        res = asyncio.run(loop.resume_pending(approved=True))
        assert res["status"] == "error"

    def test_cancel_clears_the_parked_step(self):
        loop, _, _ = make_loop(['{"action": "delete_file", "params": {"name": "x"}}'])
        asyncio.run(loop.run(user_goal="delete x"))
        loop.cancel_pending()
        assert loop.has_pending_confirmation is False


class TestGrounding:
    """UI-TARS must be a last resort, not the default path."""

    def test_resolvable_target_never_calls_the_grounder(self):
        grounder = AsyncMock()
        loop, _, _ = make_loop([
            '{"action": "cua_click", "params": {"element_id": 4}, "expect": "focused"}',
            DONE,
        ], grounder=grounder)
        asyncio.run(loop.run(user_goal="click the search box"))
        assert grounder.locate.await_count == 0

    def test_unresolvable_target_calls_the_grounder_and_uses_its_point(self):
        from grace.agent.grounder import GroundedPoint

        grounder = AsyncMock()
        grounder.locate.return_value = GroundedPoint(x=640, y=480)
        loop, _, dispatcher = make_loop([
            '{"action": "cua_click", "params": {"target_name": "the red play triangle"}, "expect": "playing"}',
            DONE,
        ], grounder=grounder)

        # A screenshot must be present for grounding to be attempted.
        with_image = ScreenSnapshot(
            active_window=None, ocr_lines=[], width=1920, height=1080,
            png_bytes=b"png", image_width=1280, image_height=720,
        )
        loop._perception.capture_snapshot.return_value = with_image
        loop._perception.capture_snapshot_async.return_value = with_image

        asyncio.run(loop.run(user_goal="play the video"))
        assert grounder.locate.await_count == 1
        params = dispatcher.execute.call_args_list[0].args[0].params
        assert params["x"] == 640 and params["y"] == 480


class TestPlannerFailureModes:
    def test_unparseable_plan_does_not_abort_the_goal(self):
        loop, _, _ = make_loop(["not json at all", DONE])
        res = asyncio.run(loop.run(user_goal="what is the weather"))
        assert res["status"] == "ok"

    def test_budget_exhaustion_returns_what_is_known(self):
        from grace.agent.planner import Planner

        click = '{"action": "cua_click", "params": {"x": 1, "y": 2}, "expect": "x"}'
        gemma = AsyncMock()
        gemma.generate_text.side_effect = [click] * 10
        dispatcher = AsyncMock()
        dispatcher.execute.return_value = {"status": "ok"}
        perception = MagicMock()
        perception.capture_snapshot_async.return_value = snapshot()

        loop = AgentLoop(
            gemma=gemma, dispatcher=dispatcher, perception=perception,
            planner=Planner(gemma, max_calls=2),
        )
        res = asyncio.run(loop.run(user_goal="click things"))
        assert res["status"] == "planner_budget_exceeded"
        assert gemma.generate_text.await_count == 2
        assert res["final_response"]

    def test_rate_limit_is_surfaced_to_the_user(self):
        from grace.llm.gemma_client import RateLimitError

        loop, gemma, _ = make_loop([RateLimitError("429 quota")])
        res = asyncio.run(loop.run(user_goal="open notepad"))
        assert res["status"] == "rate_limited"
        assert "limit" in res["final_response"].lower()
