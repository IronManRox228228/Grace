"""Tests for the rewritten agentic loop.

Covers the three things that were broken rather than merely slow: completion
was "verified" by a substring check that matched the word "which", a step
parked for safety confirmation could never be resumed, and a failed step gave
the model no signal that it had failed.
"""

import asyncio
import itertools
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.agent.loop import AgentLoop
from grace.agent.memory import AgentMemory
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.agent.planner import PlannedStep


def snapshot(title="Untitled - Notepad", graph=None, ui_elements=None):
    return ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title=title, class_name="X", rect=(0, 0, 800, 600)),
        ocr_lines=[],
        width=1920,
        height=1080,
        graph=graph,
        ui_elements=ui_elements,
    )


def make_loop(responses, dispatch_result=None, **kwargs):
    gemma = AsyncMock()
    gemma.generate_text.side_effect = responses
    dispatcher = AsyncMock()
    dispatcher.execute.return_value = dispatch_result or {"status": "ok"}
    perception = MagicMock()
    # A task that is getting somewhere changes the screen, and the loop treats
    # an unchanging screen as evidence that it isn't (TestNoProgressGuard). So
    # the shared fixture hands back a different snapshot each time; tests that
    # want the frozen-screen pathology ask for it with freeze_screen().
    screens = (snapshot(title=f"Untitled - Notepad [{i}]") for i in itertools.count())
    perception.capture_snapshot.side_effect = lambda *a, **k: next(screens)
    perception.capture_snapshot_async.return_value = snapshot()
    loop = AgentLoop(gemma=gemma, dispatcher=dispatcher, perception=perception, **kwargs)
    return loop, gemma, dispatcher


def pin_snapshot(loop, snap):
    """Make perception return exactly this snapshot every time.

    Clearing `side_effect` first is required: make_loop sets one to vary the
    screen, and a mock's side_effect takes precedence over its return_value.
    """
    loop._perception.capture_snapshot.side_effect = None
    loop._perception.capture_snapshot.return_value = snap
    loop._perception.capture_snapshot_async.return_value = snap


def freeze_screen(loop, title="Untitled - Notepad", graph=None):
    """Pin perception to one unchanging snapshot: a blind or frozen app."""
    pin_snapshot(loop, snapshot(title=title, graph=graph))


DONE = '{"action": "converse", "params": {}, "is_completed": true, "final_response": "All done."}'


class TestIterationCap:
    """Uncapped by default; an explicit positive cap is still honoured.

    A step cap only ever abandons a real task halfway, leaving the desktop in a
    half-changed state the user then has to undo by hand.
    """

    def test_there_is_no_default_step_limit(self):
        loop, _, _ = make_loop([])
        assert loop._max_iterations == 0
        assert AgentMemory("g", max_iterations=0).is_exceeded is False

    def test_an_explicit_cap_is_still_respected(self):
        click = '{"action": "cua_click", "params": {"x": 1, "y": 2}, "expect": "something"}'
        loop, _, _ = make_loop([click] * 20)
        res = asyncio.run(loop.run(user_goal="click forever", max_iterations=3))
        assert res["status"] in ("max_iterations_reached", "planner_budget_exceeded")
        assert len(res["steps"]) <= 3

    def test_uncapped_loop_runs_past_the_old_twelve_step_limit(self):
        click = '{"action": "cua_click", "params": {"x": 1, "y": 2}, "expect": "something"}'
        loop, _, _ = make_loop([click] * 19 + [DONE])
        res = asyncio.run(loop.run(user_goal="do a long thing"))
        assert res["status"] == "ok"
        assert len(res["steps"]) > 12

    def test_planner_is_uncapped_by_default(self):
        loop, _, _ = make_loop([])
        assert loop._planner.is_unlimited is True


class TestPlannerFailureCircuitBreaker:
    """An unreachable planner must end the turn, not retry forever.

    The loop is uncapped by default, and that cap-less-ness used to apply to
    failing steps too: with the LLM unreachable or the API key rejected, the
    planner returned nothing on every call and the loop retried several times a
    second indefinitely, leaving the overlay stuck mid-task. For a user who
    cannot reach a keyboard to kill it, that is a lockout rather than a slow
    response.
    """

    def test_a_planner_that_never_answers_gives_up(self):
        # What an unreachable LLM looks like from here: generate_text returns
        # None every time, so no plan ever parses.
        loop, _, _ = make_loop([None] * 200)
        res = asyncio.run(loop.run(user_goal="open my browser"))

        assert res["status"] == "planner_failed"
        assert len(res["steps"]) == loop._max_consecutive_failures

    def test_it_says_the_fault_is_graces_not_the_users(self):
        loop, _, _ = make_loop([None] * 200)
        res = asyncio.run(loop.run(user_goal="open my browser"))
        assert "language model" in res["final_response"]

    def test_isolated_failures_do_not_end_a_working_goal(self):
        # One bad plan between good ones must not count towards the breaker,
        # or a single malformed response would abandon a healthy task.
        click = '{"action": "cua_click", "params": {"x": 1, "y": 2}, "expect": "something"}'
        loop, _, _ = make_loop([click, None, click, None, click, None, DONE])
        res = asyncio.run(loop.run(user_goal="do a thing with hiccups"))
        assert res["status"] == "ok"


class TestNoProgressGuard:
    """A planner that repeats itself must stop; one making progress must not.

    Against WhatsApp - which reports four elements, its window frame, and the
    same four whatever happens inside - the loop once ran 130 steps and twelve
    minutes cycling Ctrl+F / type / Enter, and stopped only when the LLM
    provider began returning 429s.

    The obvious measure is wrong, and cost a working run: judging by "the
    observation did not change" killed a sequence that had correctly opened
    WhatsApp, searched for a group and opened it, because none of that success
    was visible through UIA. Repetition is the signal, not stillness.
    """

    CLICK = '{"action": "cua_click", "params": {"x": 1, "y": 2}, "expect": "something"}'

    def test_repeating_one_action_ends_the_loop(self):
        loop, _, _ = make_loop([self.CLICK] * 200)
        freeze_screen(loop)
        res = asyncio.run(loop.run(user_goal="find the pdf in my chemistry group"))

        assert res["status"] == "no_progress"
        # Bounded by the guard, not by the 200 plans it was offered.
        assert len(res["steps"]) < loop._max_repeated_actions + 2

    def test_distinct_actions_on_a_blind_app_are_not_interrupted(self):
        # Exactly the run that was wrongly killed: WhatsApp's four-element
        # frame never changes, but every step is different and productive.
        steps = [
            '{"action": "cua_click", "params": {"x": 300, "y": 300}, "expect": "focus"}',
            '{"action": "cua_press_key", "params": {"key": "Control_L+f"}, "expect": "search"}',
            '{"action": "cua_type_text", "params": {"text": "chemistry"}, "expect": "typed"}',
            '{"action": "cua_press_key", "params": {"key": "Return"}, "expect": "opened"}',
            DONE,
        ]
        loop, _, _ = make_loop(steps)
        freeze_screen(loop, title="WhatsApp")

        res = asyncio.run(loop.run(user_goal="open my chemistry group"))
        assert res["status"] == "ok"
        assert len(res["steps"]) == 5

    def test_the_same_key_twice_is_still_allowed(self):
        # One retry is reasonable - a window may not have been ready yet.
        key = '{"action": "cua_press_key", "params": {"key": "Return"}, "expect": "x"}'
        loop, _, _ = make_loop([key, key, DONE])
        freeze_screen(loop)

        res = asyncio.run(loop.run(user_goal="press enter twice"))
        assert res["status"] == "ok"

    def test_it_names_the_app_it_cannot_read(self):
        loop, _, _ = make_loop([self.CLICK] * 200)
        freeze_screen(loop, title="WhatsApp")

        res = asyncio.run(loop.run(user_goal="open my chemistry group"))
        assert "WhatsApp" in res["final_response"]
        assert "doesn't report its contents" in res["final_response"]

    def test_a_changing_screen_resets_the_repeat_count(self):
        # The same action against a screen that keeps changing is not a repeat:
        # each result taught the loop something.
        loop, _, _ = make_loop([self.CLICK] * 19 + [DONE])
        res = asyncio.run(loop.run(user_goal="do a long thing"))
        assert res["status"] == "ok"
        assert len(res["steps"]) > 12


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

    def test_completion_rejected_when_only_the_last_step_failed(self):
        # A regression required BOTH of the last two interactive steps to have
        # failed before a completion claim was rejected, so a real failure
        # slipped through whenever the step before it had succeeded.
        # `snapshot(ui_elements=...)` keeps the guard's own is_blind shortcut
        # ("nothing to check against, trust the claim") from masking this.
        loop, _, _ = make_loop([])
        memory = AgentMemory("open notepad")
        memory.add_step("t", "open_app", {"name": "Notepad"}, {"status": "ok"})
        memory.add_step("t", "cua_click", {}, {"status": "error", "error": "not found"})
        readable = snapshot(ui_elements=list(range(10)))
        assert loop._verify_goal_completion("open notepad", readable, memory)[0] is False

    def test_a_successful_final_action_completes_on_its_own_dispatch(self):
        # `_verify_goal_completion` used to run before `memory.add_step`, so a
        # single step that both acts and claims completion was judged with no
        # history at all and rejected - even though it had just succeeded.
        step_json = (
            '{"action": "cua_type_text", "params": {"text": "banana"}, '
            '"is_completed": true, "final_response": "Typed banana."}'
        )
        loop, _, dispatcher = make_loop([step_json], dispatch_result={"status": "ok"})
        res = asyncio.run(loop.run(user_goal="type banana"))
        assert res["status"] == "ok"
        assert res["final_response"] == "Typed banana."
        assert dispatcher.execute.await_count == 1

    def test_a_failing_final_action_is_still_rejected(self):
        # The other half: seeing the claiming step's own result must not turn
        # into always trusting it - a failed final action is still rejected.
        loop, _, _ = make_loop([])
        memory = AgentMemory("type banana")
        step = PlannedStep(action="cua_type_text", params={"text": "banana"}, is_completed=True)
        exec_result = {"status": "error", "error": "no focused field"}
        readable = snapshot(ui_elements=list(range(10)))
        verified, hint = loop._verify_goal_completion(
            "type banana", readable, memory, claiming_step=step, claiming_result=exec_result
        )
        assert verified is False
        assert "no focused field" in hint


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


class TestActionVerdict:
    """`sent` and `verified` are two claims, and `null` is not failure."""

    def note_for(self, inner):
        loop, _, _ = make_loop([])
        return loop._expectation_note(
            PlannedStep(action="cua_click", params={}, expect="the menu opens"),
            {"status": "ok", "result": inner},
            snapshot(),
        )

    def test_a_confirmed_effect_is_reported(self):
        note = self.note_for({"ok": True, "sent": True, "verified": True,
                              "evidence": "keyboard focus moved to 'Search'"})
        assert "EFFECT CONFIRMED" in note
        assert "Search" in note

    def test_no_effect_is_reported_as_no_effect(self):
        note = self.note_for({"ok": True, "sent": True, "verified": False,
                              "evidence": "nothing changed after settling"})
        assert "NO EFFECT" in note
        assert "IT FAILED" not in note, (
            "the input was dispatched; calling that a failed action tells the "
            "planner to stop using a tool that works"
        )

    def test_unknown_is_not_reported_as_failure(self):
        # The distinction the whole of A2 exists for. An app that reports
        # nothing cannot contradict a claim, and treating silence as a negative
        # verdict is what stopped a run that was succeeding.
        note = self.note_for({"ok": True, "sent": True, "verified": None,
                              "evidence": "this window does not report its contents"})
        assert "EFFECT UNKNOWN" in note
        assert "Do not treat that as failure" in note
        assert "NO EFFECT" not in note and "IT FAILED" not in note

    def test_a_refused_action_is_still_a_failure(self):
        note = self.note_for({"ok": False, "sent": False, "verified": False,
                              "error": "Focus is on 'Address bar'",
                              "evidence": "no keystrokes were sent"})
        assert "IT FAILED" in note
        assert "NO EFFECT" not in note, (
            "an action that never left the building has no effect to report on"
        )


class TestEscalationLadder:
    """A repeated action escalates through different strategies before stopping.

    The old guard could only terminate, so the answer to "that didn't work" was
    always "give up". Repetition is the trigger for trying *differently*.
    """

    CLICK = '{"action": "cua_click", "params": {"element_id": 3, "target_name": "Send"}, "expect": "the message sends"}'

    def test_the_rungs_are_in_order(self):
        from grace.agent.loop import _rung

        assert [_rung(n, 3) for n in (1, 2, 3, 4, 5)] == [
            "normal", "reground", "stronger", "stop", "stop",
        ]

    def test_a_first_attempt_is_not_grounded(self):
        loop, _, _ = make_loop([self.CLICK, DONE])
        loop._ground = AsyncMock()
        freeze_screen(loop)
        asyncio.run(loop.run(user_goal="send the message"))
        assert loop._ground.await_count == 0

    def test_a_repeat_is_regrounded(self):
        # Rung 2: the element the plan named resolved and clicking it did
        # nothing, so stop trusting the tree and ask the pixels.
        loop, _, _ = make_loop([self.CLICK] * 6)
        loop._ground = AsyncMock()
        freeze_screen(loop)
        asyncio.run(loop.run(user_goal="send the message"))
        assert loop._ground.await_count >= 1
        assert loop._ground.await_args.kwargs.get("force") is True

    def test_regrounding_drops_the_element_id(self):
        # The executor resolves the graph first, so leaving element_id in place
        # would send the click straight back to the element that just failed.
        from grace.agent.grounder import GroundedPoint

        loop, _, _ = make_loop([])
        loop._grounder.locate = AsyncMock(return_value=GroundedPoint(x=400, y=300))
        step = PlannedStep(action="cua_click",
                           params={"element_id": 3, "target_name": "Send"})
        snap = snapshot()
        snap.png_bytes = b"not-really-a-png"

        asyncio.run(loop._ground(step, snap, force=True))
        assert step.params["x"] == 400
        assert "element_id" not in step.params

    def test_an_ordinary_grounding_keeps_the_element_id(self):
        from grace.agent.grounder import GroundedPoint

        loop, _, _ = make_loop([])
        loop._grounder.locate = AsyncMock(return_value=GroundedPoint(x=400, y=300))
        step = PlannedStep(action="cua_click", params={"element_id": 3})
        snap = snapshot()
        snap.png_bytes = b"not-really-a-png"

        asyncio.run(loop._ground(step, snap))
        assert step.params["element_id"] == 3

    def test_the_third_attempt_asks_a_stronger_model(self):
        loop, _, _ = make_loop([self.CLICK] * 8)
        freeze_screen(loop)
        asyncio.run(loop.run(user_goal="send the message"))

        models = [
            c.kwargs.get("model")
            for c in loop._planner._llm.generate_text.call_args_list
        ]
        assert loop._stronger_model in models, (
            f"no call escalated to the stronger planner: {models}"
        )
        assert models[0] is None, "the first call must use the configured model"

    def test_it_still_stops(self):
        # Escalation must not become a way of never giving up.
        loop, _, _ = make_loop([self.CLICK] * 20)
        freeze_screen(loop)
        res = asyncio.run(loop.run(user_goal="send the message"))
        assert res["status"] == "no_progress"

    def test_a_rate_limited_escalation_does_not_escape_run(self):
        # `_replan_stronger` deliberately re-raises RateLimitError and
        # PlannerBudgetExceeded for its caller to handle, but the caller had no
        # try/except around it, so either exception escaped `run()` entirely
        # and took the whole process down with it.
        from grace.llm.gemma_client import RateLimitError

        loop, _, _ = make_loop([self.CLICK] * 8)
        freeze_screen(loop)
        loop._replan_stronger = AsyncMock(side_effect=RateLimitError("429 quota"))
        res = asyncio.run(loop.run(user_goal="send the message"))
        # Escalation is optional; being refused it must not end the goal when
        # the originally planned step can still be tried.
        assert res["status"] != "rate_limited"

    def test_a_budget_exceeded_escalation_stops_with_an_honest_reason(self):
        from grace.agent.planner import PlannerBudgetExceeded

        loop, _, _ = make_loop([self.CLICK] * 8)
        freeze_screen(loop)
        loop._replan_stronger = AsyncMock(
            side_effect=PlannerBudgetExceeded("planner call budget exhausted")
        )
        res = asyncio.run(loop.run(user_goal="send the message"))
        assert res["status"] == "planner_budget_exceeded"
        assert res["final_response"]


class TestRollingScratchpad:
    """What was established and ruled out has to outlive the 3-step window."""

    def test_a_failed_step_is_ruled_out(self):
        memory = AgentMemory("g")
        loop, _, _ = make_loop([])
        step = PlannedStep(action="cua_click", params={"element_id": 3}, expect="it opens")
        loop._remember(step, {"status": "error", "error": "element not found"}, memory)

        assert any("element not found" in e for e in memory.scratchpad["ruled_out"])
        assert "established" not in memory.scratchpad

    def test_a_verified_step_establishes_its_expectation(self):
        memory = AgentMemory("g")
        loop, _, _ = make_loop([])
        step = PlannedStep(action="cua_click", params={}, expect="the chat list is open")
        loop._remember(step, {"status": "ok", "result": {"verified": True,
                                                         "evidence": "focus moved"}}, memory)
        assert memory.scratchpad["established"] == ["the chat list is open"]

    def test_unknown_outcomes_are_not_recorded_as_either(self):
        # A window that reports nothing produces one of these on every step.
        # Writing them down would crowd out the facts worth carrying and teach
        # the planner that nothing works.
        memory = AgentMemory("g")
        loop, _, _ = make_loop([])
        step = PlannedStep(action="cua_click", params={}, expect="something")
        loop._remember(step, {"status": "ok", "result": {"verified": None,
                                                         "evidence": "cannot tell"}}, memory)
        assert memory.scratchpad == {}

    def test_entries_are_deduplicated_and_bounded(self):
        from grace.agent.memory import MAX_REMEMBERED

        memory = AgentMemory("g")
        memory.establish("the same fact")
        memory.establish("the same fact")
        assert memory.scratchpad["established"] == ["the same fact"]

        for i in range(MAX_REMEMBERED + 5):
            memory.rule_out(f"approach {i}")
        assert len(memory.scratchpad["ruled_out"]) == MAX_REMEMBERED

    def test_it_reaches_the_planner_prompt(self):
        memory = AgentMemory("g")
        memory.rule_out("`cua_click` with {'element_id': 3}", "nothing changed")
        assert "cua_click" in memory.format_scratchpad_markdown()
        assert "ruled_out" in memory.format_scratchpad_markdown()


class TestWallClockBudget:
    """Time is the only budget a stuck loop cannot spend slowly."""

    def test_a_goal_stops_when_it_runs_out_of_time(self):
        click = '{"action": "cua_click", "params": {"element_id": 1}, "expect": "something"}'
        loop, _, _ = make_loop([click] * 50)
        # Already over budget when the first iteration is checked.
        loop._max_seconds = 1
        memory = AgentMemory("do a slow thing", max_seconds=1)
        memory._started -= 5

        res = asyncio.run(loop._continue(memory))
        assert res["status"] == "timed_out"
        assert "stopped" in res["final_response"]

    def test_the_clock_does_not_run_while_waiting_for_an_answer(self):
        # A safety question is answered by voice. Counting the user's thinking
        # time against their budget would mean the slower they speak, the less
        # of their task gets done.
        memory = AgentMemory("delete something", max_seconds=60)
        memory.pause_clock()
        memory._paused_at -= 30
        memory.resume_clock()
        assert memory.elapsed_seconds < 1

    def test_zero_disables_the_clock(self):
        memory = AgentMemory("g", max_seconds=0)
        memory._started -= 10_000
        assert memory.is_out_of_time is False

    def test_the_default_is_not_unlimited(self):
        # The twelve-minute run crossed no limit because there was none to
        # cross. If this ever reverts to 0 the whole of A4 is inert.
        from grace.agent.loop import DEFAULT_MAX_SECONDS

        assert DEFAULT_MAX_SECONDS > 0
        loop, _, _ = make_loop([])
        assert loop._max_seconds > 0


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

    def test_a_fast_path_intent_can_be_parked_and_confirmed(self):
        # The fast path has no goal, no memory and no plan behind it - one tool
        # call - so it parks as itself. Before the guard moved to the dispatch
        # boundary there was nothing to park, because nothing asked.
        from grace.intent.parser import Intent

        loop, _, dispatcher = make_loop([])
        loop.park_intent(Intent(tool="delete_file", params={"name": "x.txt"}), "Sure?")
        assert loop.has_pending_confirmation is True
        assert loop.pending_prompt == "Sure?"

        res = asyncio.run(loop.resume_pending(approved=True))
        assert res["status"] == "ok"
        call = dispatcher.execute.call_args
        assert call.args[0].tool == "delete_file"
        assert call.kwargs.get("confirmed") is True, (
            "the parked action must be dispatched as confirmed, or the guard "
            "refuses it a second time and 'yes' silently does nothing"
        )
        assert loop.has_pending_confirmation is False

    def test_a_declined_fast_path_intent_never_runs(self):
        from grace.intent.parser import Intent

        loop, _, dispatcher = make_loop([])
        loop.park_intent(Intent(tool="delete_file", params={"name": "x.txt"}), "Sure?")
        res = asyncio.run(loop.resume_pending(approved=False))
        assert dispatcher.execute.call_count == 0
        assert "won't" in res["final_response"]

    def test_cancel_clears_the_parked_step(self):
        loop, _, _ = make_loop(['{"action": "delete_file", "params": {"name": "x"}}'])
        asyncio.run(loop.run(user_goal="delete x"))
        loop.cancel_pending()
        assert loop.has_pending_confirmation is False

    def test_a_stale_confirmation_expires_and_cannot_be_resumed(self):
        # Without a TTL, a delete parked at the start of a goal stayed
        # approvable forever - an unrelated "yes" spoken hours later in a
        # different conversation would silently confirm it.
        from grace.intent.parser import Intent

        loop, _, dispatcher = make_loop([])
        loop.park_intent(Intent(tool="delete_file", params={"name": "x.txt"}), "Sure?")
        assert loop.has_pending_confirmation is True

        loop._pending["parked_at"] -= 31  # older than the 30s TTL

        assert loop.has_pending_confirmation is False
        res = asyncio.run(loop.resume_pending(approved=True))
        assert res["status"] == "error"
        assert dispatcher.execute.call_count == 0


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
        pin_snapshot(loop, with_image)

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


class TestCancellation:
    """R5: once a goal is running, a spoken "stop" must actually stop it.

    Before this, `AgentLoop.run` consulted no cancellation source at all - its
    own step/time/repeat budgets were the only way it ever ended early. These
    exercise `request_cancel()` at the loop's own two checkpoints, deterministic
    and without any real concurrency: a mock's side effect sets the flag at the
    exact moment being tested, rather than racing a background task against it.
    """

    CLICK = '{"action": "cua_click", "params": {"x": 1, "y": 2}, "expect": "something"}'

    def test_cancel_between_steps_stops_before_the_next_plan(self):
        # The flag is set from inside the first dispatch, which stands in for
        # a "stop" heard while that action was running. The top-of-loop check
        # must catch it before a second plan is even requested.
        loop, gemma, dispatcher = make_loop([self.CLICK, self.CLICK, DONE])

        async def dispatch_then_cancel(*args, **kwargs):
            loop.request_cancel()
            return {"status": "ok"}

        dispatcher.execute.side_effect = dispatch_then_cancel

        res = asyncio.run(loop.run(user_goal="click twice"))

        assert res["status"] == "cancelled"
        assert len(res["steps"]) == 1
        assert gemma.generate_text.call_count == 1

    def test_cancel_during_planning_stops_before_that_steps_dispatch(self):
        # The flag is set while the *second* step is being planned - after the
        # first step already dispatched, before the second one does. Only the
        # pre-dispatch checkpoint (not the top-of-loop one) can catch this.
        loop, gemma, dispatcher = make_loop([self.CLICK, self.CLICK, DONE])
        calls = {"n": 0}

        def plan_then_cancel_on_second_call(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                loop.request_cancel()
            return self.CLICK
        gemma.generate_text.side_effect = plan_then_cancel_on_second_call

        res = asyncio.run(loop.run(user_goal="click twice"))

        assert res["status"] == "cancelled"
        assert dispatcher.execute.await_count == 1

    def test_cancellation_ends_the_goal_like_a_timeout_does(self):
        # Same shape as `_timeout_result`: a spoken acknowledgement and the
        # steps taken so far, not an error - main.py's normal end-of-turn
        # handling needs no special case for a voice-cancelled goal.
        loop, _, dispatcher = make_loop([self.CLICK, DONE])

        async def dispatch_then_cancel(*args, **kwargs):
            loop.request_cancel()
            return {"status": "ok"}
        dispatcher.execute.side_effect = dispatch_then_cancel

        res = asyncio.run(loop.run(user_goal="click something"))

        assert res["status"] == "cancelled"
        assert "stopped" in res["final_response"].lower()
        assert res["steps"]

    def test_a_finished_cancellation_does_not_carry_into_the_next_goal(self):
        # `run()` must reset the flag for each new goal, or a cancelled task
        # would leave every goal after it silently dying at the first
        # checkpoint too.
        loop, gemma, dispatcher = make_loop([self.CLICK, DONE, self.CLICK, DONE])

        async def dispatch_then_cancel(*args, **kwargs):
            loop.request_cancel()
            return {"status": "ok"}
        dispatcher.execute.side_effect = dispatch_then_cancel

        first = asyncio.run(loop.run(user_goal="first goal"))
        assert first["status"] == "cancelled"

        dispatcher.execute.side_effect = None
        dispatcher.execute.return_value = {"status": "ok"}
        second = asyncio.run(loop.run(user_goal="second goal"))
        assert second["status"] == "ok"
