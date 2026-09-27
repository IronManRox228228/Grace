"""Agentic Control Loop (ReAct Engine) for Grace.

Observe -> plan -> (ground) -> act -> verify, until the goal is done.

The loop's job is orchestration only. Deciding *what* to do belongs to
`agent.planner` (structure, no pixels) and deciding *where* something is on
screen belongs to `agent.grounder` (pixels, no goals). Keeping those apart is
what stopped the model conflating a browser address bar with a page's own
search box.
"""

import asyncio
import logging
import re
import time
from types import SimpleNamespace
from typing import Any, Optional

from grace.agent.grounder import Grounder, describe_target
from grace.agent.memory import AgentMemory
from grace.agent.perception import PerceptionEngine, observe_for_planner
from grace.agent.planner import Planner, PlannedStep, PlannerBudgetExceeded, parse_planned_step
from grace.agent.safety import SafetyGuard
from grace.agent.ui_tars_parser import UITarsParser
from grace.intent.parser import Intent
from grace.llm.gemma_client import GemmaClient, RateLimitError
from grace.tools.dispatcher import Dispatcher
from grace.util.timing import stage

logger = logging.getLogger("grace.agent.loop")

# Tools whose execution constitutes actually doing something to the desktop.
INTERACTIVE_TOOLS = {
    "cua_click", "cua_type_text", "cua_press_key", "cua_scroll",
    "cua_drag", "cua_launch", "cua_activate", "cua_set_value",
    "cua_secondary_action",
    "open_app", "close_app", "open_file", "delete_file",
    "read_pdf", "summarize_pdf", "search_files",
    "adjust_volume", "lock_computer", "open_calculator",
}

# Phrasings that describe a question rather than a desktop change. Matched on
# word boundaries: the old substring check treated "hi" as conversational, so
# "which", "this" and "white" all skipped verification entirely.
_CONVERSATIONAL_PATTERNS = (
    r"\bwhat\s+is\b", r"\bwhat's\b", r"\bwho\s+is\b", r"\bwhere\s+is\b",
    r"\bwhy\b", r"\btell\s+me\b", r"\bexplain\b", r"\bhow\s+(?:are|do|does)\b",
    r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bthanks?\b", r"\bcalculate\b",
)
_CONVERSATIONAL_RE = re.compile("|".join(_CONVERSATIONAL_PATTERNS), re.IGNORECASE)

# 0 means no step limit. A cap here does not make the agent smarter, it just
# makes it stop halfway through a real task; set AGENT_MAX_ITERATIONS to a
# positive number to re-impose one.
DEFAULT_MAX_ITERATIONS = 0

# How many plans in a row may fail to parse before the loop gives up.
#
# `AGENT_MAX_ITERATIONS=0` means unlimited, which is the right default for real
# goals - some legitimately need many steps. But "unlimited" also applied to
# *failing* steps, so anything that made the planner return nothing on every
# call (LLM unreachable, rejected API key, an outage mid-goal) turned into an
# infinite retry at a few hundred milliseconds a lap, with the overlay stuck
# mid-task and no way to interrupt it. For a user who navigates by voice and
# cannot reach a keyboard, that is not a degraded response, it is a lockout.
#
# This bounds *consecutive* failures only. A goal that keeps making progress is
# still unlimited; the counter resets the moment one plan parses.
DEFAULT_MAX_CONSECUTIVE_PLAN_FAILURES = 3

# How many times the loop may issue the *same action with the same parameters*
# while the screen it can see stays identical, before concluding it is stuck.
#
# The failure this exists for is not a planner that breaks, but one that works:
# against an app whose accessibility tree never changes, the planner reasons
# soundly about a stale observation, acts, sees the same thing, and reasons
# again. One session ran 130 steps and nearly twelve minutes that way, cycling
# Ctrl+F / type / Enter roughly ten times over, and stopped only because the LLM
# provider began returning 429s.
#
# Note carefully what is counted, because the obvious measure is wrong. "The
# observation did not change" is *not* evidence of failure: WhatsApp reports
# four elements - its window frame - and reports the same four whatever happens
# inside it. Judging by that alone killed a run that had correctly opened the
# app, searched for a group and opened it, because none of that success was
# visible through UIA. What actually distinguishes stuck from working is
# repetition: a loop that is getting somewhere issues *different* actions, even
# when it cannot see the results.
#
# So the repeat counter resets the moment the observation changes, and only
# accumulates for an identical action retried against an identical view.
#
# What the counter now drives is the ladder below, not an immediate stop. A
# guard that can only terminate turns every unfamiliar situation into a refusal;
# repetition is a signal to try *differently*, and only the last rung stops.
DEFAULT_MAX_REPEATED_ACTIONS = 3

# The escalation ladder, indexed by how many times this exact action has been
# tried against this exact view.
#
# Each rung is a genuinely different strategy rather than a retry, which is what
# makes spending another attempt worth anything:
#
#   normal    - as planned. Nearly every goal lives here and never leaves.
#   reground  - the element the plan named is not working. Stop trusting the
#               tree and ask the visual grounder (UI-TARS) where the thing
#               actually is, in pixels. This path has never once fired in
#               production: `needs_grounding` requires a *named* target with no
#               coordinates, and until Stage 3 a blind planner invented
#               coordinates instead, which disqualified it.
#   stronger  - re-plan this step with a more capable model, carrying the
#               history and the failure evidence. Flash Lite is fast and does
#               not model UI state; paying for capability on the one step that
#               needs it is the whole point of a tiered design.
#   stop      - name what was tried, and say so.
ESCALATION_LADDER = ("normal", "reground", "stronger", "stop")

# The model the third rung asks. Deliberately a different, stronger model than
# the configured planner; if it is unset, that rung degrades to a plain retry
# with the failure evidence attached, which is still worth one attempt.
DEFAULT_STRONGER_PLANNER_MODEL = "gemini-3.1-flash"

# How long a parked safety confirmation waits for an answer before it is
# treated as abandoned. Without this, a delete parked at the start of a goal
# stayed approvable indefinitely - an unrelated "yes" spoken in a much later,
# unrelated conversation would silently execute it.
PENDING_CONFIRMATION_TTL_SECONDS = 30

# The wall-clock ceiling for one goal, in seconds.
#
# This is the bound the twelve-minute run needed and did not have. Every other
# limit here counts something the loop does - steps, plans, repeats - and a loop
# that is failing usefully-differently each time passes all of them while the
# user sits and waits. Time is the one budget that cannot be spent slowly.
#
# Three minutes is chosen to be longer than any goal observed to succeed
# (the working chemistry-group run took about 40 seconds) and far shorter than
# the failure. Set AGENT_MAX_SECONDS=0 to disable, at the cost of restoring the
# state where nothing structurally prevents an unbounded run.
DEFAULT_MAX_SECONDS = 180


class AgentLoop:
    """Autonomous ReAct Execution Engine for complex multi-step tasks."""

    def __init__(
        self,
        gemma: GemmaClient,
        dispatcher: Dispatcher,
        perception: Optional[PerceptionEngine] = None,
        ws_server: Optional[Any] = None,
        vision_llm: Optional[GemmaClient] = None,
        planner: Optional[Planner] = None,
        grounder: Optional[Grounder] = None,
        max_iterations: Optional[int] = None,
        start_grounding_backend=None,
    ):
        self._gemma = gemma
        self._vision_llm = vision_llm or gemma
        self._dispatcher = dispatcher
        self._perception = perception or PerceptionEngine()
        self._ws_server = ws_server
        self._planner = planner or Planner(gemma, max_calls=_config_int("planner_max_calls_per_goal", 0))
        self._grounder = grounder or Grounder(self._vision_llm, on_demand_start=start_grounding_backend)
        # `is None`, not `or`: 0 is a meaningful value here (unlimited).
        self._max_iterations = (
            max_iterations if max_iterations is not None
            else _config_int("agent_max_iterations", DEFAULT_MAX_ITERATIONS)
        )
        self._max_consecutive_failures = _config_int(
            "agent_max_consecutive_plan_failures", DEFAULT_MAX_CONSECUTIVE_PLAN_FAILURES
        )
        self._max_repeated_actions = _config_int(
            "agent_max_repeated_actions", DEFAULT_MAX_REPEATED_ACTIONS
        )
        self._max_seconds = _config_int("agent_max_seconds", DEFAULT_MAX_SECONDS)
        self._stronger_model = _config_str(
            "stronger_planner_model", DEFAULT_STRONGER_PLANNER_MODEL
        )
        self._pending: Optional[dict[str, Any]] = None
        # Set by `request_cancel()`, checked between steps and again just
        # before a step is dispatched (R5). This is the only way a goal already
        # running can be stopped - the loop's own budgets all end it on their
        # own terms, none of them on the user's.
        self._cancel_requested = False

    # -- safety resumption -------------------------------------------------

    @property
    def has_pending_confirmation(self) -> bool:
        """True when a step is parked waiting for the user to say yes or no.

        SafetyGuard used to set memory.safety_pending and return, with nothing
        anywhere able to resume it - so answering "yes" started an unrelated new
        request and the confirmed action never ran.
        """
        self._expire_pending()
        return self._pending is not None

    @property
    def pending_prompt(self) -> Optional[str]:
        self._expire_pending()
        return self._pending.get("prompt") if self._pending else None

    def cancel_pending(self) -> None:
        self._pending = None

    def request_cancel(self) -> None:
        """Ask a running goal to stop at the next checkpoint.

        Meant to be called from outside the loop - the caller wires this to a
        spoken "stop"/"cancel" heard while a goal is in flight (R5), since
        nothing else can reach a running goal at all. It only sets a flag:
        the loop itself decides when it is safe to act on it, between steps
        and again immediately before dispatching one, so a step already
        announced to the user is never abandoned mid-way.
        """
        self._cancel_requested = True

    def _expire_pending(self) -> None:
        """Drop a parked confirmation once it has waited too long to still apply.

        Without a TTL a pending delete or app-close stays approvable forever, so
        an unrelated "yes" spoken in a much later exchange would silently confirm
        it. The clock is `time.monotonic()`, the same one `AgentMemory` uses for
        the goal's own wall-clock budget.
        """
        if self._pending is None:
            return
        parked_at = self._pending.get("parked_at", 0.0)
        if time.monotonic() - parked_at > PENDING_CONFIRMATION_TTL_SECONDS:
            logger.warning("Pending confirmation expired unanswered; cancelling it.")
            self._pending = None

    def park_intent(self, intent: Intent, prompt: str) -> None:
        """Hold a fast-path tool call until the user answers the question.

        A fast-path intent has no goal, no memory and no plan behind it - it is
        one tool call - so it parks as itself. It lives here rather than in
        `main.py` because this is where `has_pending_confirmation` is already
        consulted, and two places tracking "is something waiting for a yes"
        is how one of them comes to be missed.
        """
        self._pending = {"intent": intent, "prompt": prompt, "parked_at": time.monotonic()}

    async def resume_pending(self, approved: bool) -> dict[str, Any]:
        """Continue whatever stopped for a safety confirmation."""
        self._expire_pending()
        if not self._pending:
            return {"status": "error", "error": "Nothing is waiting for confirmation."}

        pending = self._pending
        self._pending = None

        if "intent" in pending:
            return await self._resume_intent(pending["intent"], approved)

        memory: AgentMemory = pending["memory"]
        step: PlannedStep = pending["step"]
        memory.resume_clock()

        if not approved:
            memory.safety_pending = None
            memory.is_completed = True
            memory.final_response = "Alright, I won't do that."
            memory.add_step(step.thought, "converse", {}, {"status": "cancelled"}, "Cancelled")
            return self._result(memory)

        logger.info(f"Resuming confirmed action '{step.action}'")
        memory.safety_pending = None
        exec_result = await self._dispatch(step, memory, confirmed=True)
        memory.add_step(step.thought, step.action, step.params, exec_result, step.user_update)
        return await self._continue(memory, last_step=step, last_result=exec_result)

    async def _resume_intent(self, intent: Intent, approved: bool) -> dict[str, Any]:
        """Run, or drop, a parked fast-path tool call."""
        if not approved:
            return {"status": "ok", "final_response": "Alright, I won't do that.", "steps": []}

        logger.info(f"Resuming confirmed fast-path action '{intent.tool}'")
        result = await self._dispatcher.execute(intent, confirmed=True)
        return {
            "status": result.get("status", "ok"),
            "final_response": result.get("text") or "Done.",
            "steps": [],
        }

    # -- main entry point --------------------------------------------------

    async def run(self, user_goal: str, max_iterations: Optional[int] = None) -> dict[str, Any]:
        """Run the autonomous Observe-Plan-Act loop for a given user goal."""
        limit = max_iterations if max_iterations is not None else self._max_iterations
        memory = AgentMemory(user_goal=user_goal, max_iterations=limit,
                             max_seconds=self._max_seconds)
        self._planner.reset()
        self._pending = None
        self._cancel_requested = False
        logger.info(
            f"AgentLoop started for goal: '{user_goal}' "
            f"({'unlimited' if not limit or limit <= 0 else f'max {limit}'} steps, "
            f"{f'{self._max_seconds}s' if self._max_seconds > 0 else 'no time limit'})"
        )
        return await self._continue(memory)

    async def _continue(
        self,
        memory: AgentMemory,
        last_step: Optional[PlannedStep] = None,
        last_result: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Drive the loop until completion, the step cap, or a confirmation."""
        consecutive_failures = 0
        last_observation: Optional[str] = None
        attempts: dict[str, int] = {}

        while not memory.is_completed and not memory.is_exceeded:
            if self._cancel_requested:
                logger.info("AgentLoop: cancelled by voice stop request.")
                return self._cancel_result(memory)

            if memory.is_out_of_time:
                logger.error(
                    f"AgentLoop: out of time after {memory.elapsed_seconds:.0f}s "
                    f"and {memory.current_iteration} steps."
                )
                return self._timeout_result(memory)

            step_no = memory.current_iteration + 1

            async with stage(f"observe#{step_no}"):
                snapshot = await self._observe()

            # Built once and reused: the mode decides both what the planner is
            # shown and what counts as "the view changed", and deriving those
            # from two separate renderings is how they drift apart.
            view = observe_for_planner(snapshot)

            # A changed view means the loop has learned something, so nothing it
            # tried before is a repeat any more.
            if view.markdown != last_observation:
                last_observation = view.markdown
                attempts.clear()

            expectation_note = self._expectation_note(last_step, last_result, snapshot)

            try:
                async with stage(f"plan#{step_no}") as plan_stage:
                    step = await self._planner.plan(
                        goal=memory.user_goal,
                        elements_prompt=view.markdown,
                        history=memory.format_history_markdown(),
                        scratchpad=memory.format_scratchpad_markdown(),
                        expectation_note=expectation_note,
                        window_title=_window_title(snapshot),
                        image_b64=view.image_b64,
                    )
                    plan_stage.detail(
                        f"{step.action if step else 'unparsed'} "
                        f"(call {self._planner.calls_made}, {view.mode}"
                        f"{f', {view.marks} marks' if view.marks else ''})"
                    )
            except PlannerBudgetExceeded as e:
                logger.warning(str(e))
                return self._budget_result(memory)
            except RateLimitError as e:
                logger.error(f"Planner rate limited: {e}")
                return {
                    "status": "rate_limited",
                    "final_response": "I've hit my request limit for now. Please try again shortly.",
                    "steps": [s.to_dict() for s in memory.steps_taken],
                }

            if step is None:
                # An unparseable plan is a wasted step, not a fatal error, but
                # it must consume budget or the loop can spin forever.
                memory.set_scratchpad("last_error", "The previous plan could not be parsed.")
                memory.add_step("", "converse", {}, {"status": "error", "error": "unparseable plan"}, "Rethinking…")
                last_step, last_result = None, None

                # Consuming budget is only a brake when the budget is finite,
                # and the shipped default is unlimited. Without this, a planner
                # that fails every call never stops being retried.
                consecutive_failures += 1
                if consecutive_failures >= self._max_consecutive_failures:
                    logger.error(
                        f"AgentLoop: giving up after {consecutive_failures} consecutive "
                        f"plans that could not be used."
                    )
                    return self._plan_failure_result(memory)
                continue

            consecutive_failures = 0

            # Retrying the identical action against the identical view. Doing
            # it once more is reasonable - a window may not have been ready -
            # but past that the loop is just spending time and quota.
            signature = _step_signature(step)
            attempts[signature] = attempts.get(signature, 0) + 1
            rung = _rung(attempts[signature], self._max_repeated_actions)

            if rung == "stop":
                logger.error(
                    f"AgentLoop: giving up - tried {step.action} with the same "
                    f"parameters {attempts[signature]} times, through every "
                    f"strategy available, and nothing changed."
                )
                memory.rule_out(f"{step.action} on this screen", "tried at every tier; nothing changed")
                return self._no_progress_result(memory, snapshot)

            if rung == "stronger":
                logger.warning(
                    f"AgentLoop: escalating - re-planning '{step.action}' with "
                    f"{self._stronger_model or 'the same model'}."
                )
                try:
                    stronger = await self._replan_stronger(memory, view, snapshot, expectation_note)
                except PlannerBudgetExceeded as e:
                    # The overall per-goal budget is spent, not just this one
                    # optional attempt - the same terminal condition the
                    # ordinary planning call above hits, so it gets the same
                    # honest, spoken reason instead of taking Grace down.
                    logger.warning(str(e))
                    return self._budget_result(memory)
                except RateLimitError as e:
                    # Escalation is optional (see _replan_stronger); a rate
                    # limit on this extra call must not end the goal when the
                    # step can still be tried as originally planned.
                    logger.warning(f"Escalated re-plan rate limited, continuing without it: {e}")
                    stronger = None
                if stronger is not None:
                    step = stronger
                    signature = _step_signature(step)
                    attempts[signature] = attempts.get(signature, 0) + 1

            logger.info(f"AgentLoop step {step_no}: [{step.action}] {step.thought}")

            is_safe, confirm_prompt = SafetyGuard.evaluate(step.action, step.params)
            if not is_safe:
                logger.warning(f"AgentLoop: safety confirmation required for '{step.action}'")
                memory.safety_pending = {
                    "action": step.action,
                    "params": step.params,
                    "prompt": confirm_prompt,
                }
                memory.pause_clock()
                self._pending = {
                    "memory": memory, "step": step, "prompt": confirm_prompt,
                    "parked_at": time.monotonic(),
                }
                return {
                    "status": "safety_confirmation_required",
                    "confirmation_prompt": confirm_prompt,
                    "pending_step": step.to_dict(),
                    "steps": [s.to_dict() for s in memory.steps_taken],
                }

            if step.needs_grounding or rung == "reground":
                async with stage(f"ground#{step_no}"):
                    await self._ground(step, snapshot, force=(rung == "reground"))

            # Checked again here, not just at the top of the loop: planning
            # (and grounding, above) can take long enough that a "stop" heard
            # while this step was being worked out must still be honoured
            # before the step it produced is ever sent to the desktop.
            if self._cancel_requested:
                logger.info("AgentLoop: cancelled by voice stop request.")
                return self._cancel_result(memory)

            await self._emit({
                "type": "ToolExecutionStarted",
                "label": step.user_update,
                "tool": step.action,
                "step": step_no,
            })

            # Executed before the completion check so a final step that also
            # performs an action is never skipped.
            exec_result = await self._dispatch(step, memory)

            if step.is_completed or step.action == "converse":
                verified, hint = self._verify_goal_completion(
                    memory.user_goal, snapshot, memory, claiming_step=step, claiming_result=exec_result
                )
                if verified:
                    memory.is_completed = True
                    memory.final_response = self._final_response(step, memory)
                    memory.add_step(step.thought, step.action, step.params, exec_result, step.user_update)
                    break
                logger.warning(f"AgentLoop: completion rejected by verification guard: {hint}")
                memory.set_scratchpad("verification_hint", hint)
                step.is_completed = False

            self._remember(step, exec_result, memory)
            self._harvest(exec_result, memory)
            memory.add_step(step.thought, step.action, step.params, exec_result, step.user_update)
            last_step, last_result = step, exec_result

            await self._emit({
                "type": "ToolExecutionFinished",
                "tool": step.action,
                "status": exec_result.get("status", "ok"),
            })

        if memory.is_exceeded and not memory.is_completed:
            return {
                "status": "max_iterations_reached",
                "final_response": "I reached the step limit before fully completing the goal.",
                "steps": [s.to_dict() for s in memory.steps_taken],
            }

        return self._result(memory)

    # -- pieces ------------------------------------------------------------

    async def _observe(self):
        snap_res = self._perception.capture_snapshot_async()
        if asyncio.iscoroutine(snap_res):
            return await snap_res
        return self._perception.capture_snapshot()

    async def _dispatch(self, step: PlannedStep, memory: AgentMemory,
                        confirmed: bool = False) -> dict[str, Any]:
        if step.action == "converse":
            return {"status": "ok"}
        intent = Intent(tool=step.action, params=step.params)
        async with stage(f"dispatch:{step.action}"):
            return await self._dispatcher.execute(intent, confirmed=confirmed)

    async def _replan_stronger(
        self, memory: AgentMemory, view, snapshot, expectation_note: str
    ) -> Optional[PlannedStep]:
        """Ask a more capable model for a different approach to the same goal.

        Given the same prompt it would likely produce the same plan, so the
        scratchpad is what makes this rung worth its cost: it carries what has
        already been established and ruled out, which the three-step verbatim
        history has usually forgotten by the time this fires.
        """
        try:
            return await self._planner.plan(
                goal=memory.user_goal,
                elements_prompt=view.markdown,
                history=memory.format_history_markdown(),
                scratchpad=memory.format_scratchpad_markdown(),
                expectation_note=expectation_note,
                window_title=_window_title(snapshot),
                image_b64=view.image_b64,
                model=self._stronger_model or None,
            )
        except (PlannerBudgetExceeded, RateLimitError):
            # Both are handled by the caller's own planning path on the next
            # lap. Escalating is optional; failing the goal because the
            # optional attempt was refused is not.
            raise
        except Exception as e:
            logger.warning(f"Escalated re-plan failed, keeping the original step: {e}")
            return None

    async def _ground(self, step: PlannedStep, snapshot, force: bool = False) -> None:
        """Fill in x/y for a target the element graph could not resolve.

        `force` is the ladder's second rung: the element the plan named resolved
        fine and clicking it changed nothing, so the tree is to be distrusted
        and the pixels asked instead. That means dropping `element_id` on
        success - the executor resolves the graph first, so leaving it in would
        send the click straight back to the element that already failed.
        """
        description = describe_target(step.params)
        png = getattr(snapshot, "png_bytes", None)
        if not png:
            logger.debug(f"No screenshot available to ground '{description}'")
            return

        image_size = (
            getattr(snapshot, "image_width", 0) or snapshot.width,
            getattr(snapshot, "image_height", 0) or snapshot.height,
        )
        point = await self._grounder.locate(
            description=description,
            png_bytes=png,
            image_size=image_size,
            screen_size=(snapshot.width, snapshot.height),
        )
        if point is not None:
            step.params["x"] = point.x
            step.params["y"] = point.y
            if force:
                step.params.pop("element_id", None)
                step.params.pop("element_index", None)
            logger.info(f"Grounded '{description}' to ({point.x}, {point.y})")

    def _expectation_note(
        self,
        last_step: Optional[PlannedStep],
        last_result: Optional[dict[str, Any]],
        snapshot,
    ) -> str:
        """Report the previous step's stated expectation against what happened.

        This is the verification signal. Rather than guessing semantically
        whether `expect` came true, it hands the planner the facts - status,
        focus, window title - alongside its own stated expectation, inside the
        call it was going to make anyway. No extra model call, and a failed
        step becomes an explicit correction instead of a blind retry.
        """
        if last_step is None or not last_result:
            return ""

        lines = [f"You ran `{last_step.action}` and expected: {last_step.expect or '(nothing stated)'}"]

        status = last_result.get("status", "ok")
        inner = last_result.get("result") if isinstance(last_result.get("result"), dict) else {}
        reason = _failure_reason(last_result)

        if reason:
            lines.append(f"IT FAILED: {reason}")
            lines.append("Do not repeat this action unchanged. Try a different element or a different approach.")
        else:
            lines.append(f"The tool reported: {inner.get('message') or status}")

        # The A2 verdict. `sent` and `verified` are separate claims, and the
        # third value is the one that matters: "I could not tell" is a different
        # instruction to the planner than "it did not work", and before this
        # they were the same string.
        verified = inner.get("verified")
        evidence = inner.get("evidence") or ""
        if verified is True:
            lines.append(f"EFFECT CONFIRMED: {evidence}")
        elif verified is False and not reason:
            lines.append(f"NO EFFECT: {evidence}")
            lines.append("The input was sent but changed nothing. Try a different target.")
        elif verified is None and inner.get("sent"):
            lines.append(f"EFFECT UNKNOWN: {evidence}")
            lines.append("Do not treat that as failure. Look at the screen description "
                         "below and judge for yourself.")

        title = _window_title(snapshot)
        if title:
            lines.append(f"The active window is now: {title}")

        graph = getattr(snapshot, "graph", None)
        focused = graph.focused() if graph is not None else None
        if focused is not None:
            lines.append(
                f"Keyboard focus is on [{focused.id}] {focused.role} "
                f"'{focused.name or focused.placeholder}' (frame={focused.frame})"
            )
        elif last_step.action == "cua_click":
            lines.append("Nothing currently has keyboard focus.")

        lines.append("Judge from the screen description above whether your expectation actually came true.")
        return "\n".join(lines)

    @staticmethod
    def _remember(step: PlannedStep, exec_result: dict[str, Any], memory: AgentMemory) -> None:
        """Keep what this step settled, in a form that outlives the history window.

        Only definite outcomes are recorded. `verified is None` is deliberately
        not written down as either: an unreadable window produces one of those
        on every single step, and filling the scratchpad with "could not tell"
        would crowd out the facts that are worth carrying and teach the planner
        that nothing works.
        """
        reason = _failure_reason(exec_result)
        if reason:
            memory.rule_out(f"`{step.action}` with {_short_params(step.params)}", reason)
            return

        inner = exec_result.get("result") if isinstance(exec_result.get("result"), dict) else {}
        verified = inner.get("verified")
        if verified is True and step.expect:
            memory.establish(step.expect)
        elif verified is False:
            memory.rule_out(
                f"`{step.action}` with {_short_params(step.params)}",
                inner.get("evidence") or "it changed nothing",
            )

    def _harvest(self, exec_result: dict[str, Any], memory: AgentMemory) -> None:
        """Move useful tool output into the scratchpad."""
        if exec_result.get("status") != "ok":
            return
        res = exec_result.get("result", exec_result)
        if isinstance(res, dict):
            if "windows" in res:
                memory.set_scratchpad(
                    "open_windows",
                    [w.get("title", "") for w in res["windows"] if w.get("title")],
                )
            if "apps" in res:
                memory.set_scratchpad(
                    "open_apps",
                    [a.get("name", "") for a in res["apps"] if a.get("name")],
                )
        for key, target in (("text", "latest_extracted_text"),
                            ("summary", "latest_summary"),
                            ("files", "found_files")):
            if key in exec_result:
                memory.set_scratchpad(target, exec_result[key])

    def _final_response(self, step: PlannedStep, memory: AgentMemory) -> str:
        final = step.final_response
        if final and final != step.thought:
            return final
        if "open_windows" in memory.scratchpad:
            wins = memory.scratchpad["open_windows"]
            return f"The open windows are: {', '.join(wins[:5])}."
        if "open_apps" in memory.scratchpad:
            apps = memory.scratchpad["open_apps"]
            return f"The open applications are: {', '.join(apps[:5])}."
        return "Goal completed."

    def _result(self, memory: AgentMemory) -> dict[str, Any]:
        return {
            "status": "ok",
            "final_response": memory.final_response,
            "steps": [s.to_dict() for s in memory.steps_taken],
        }

    def _budget_result(self, memory: AgentMemory) -> dict[str, Any]:
        """Finish with what is known rather than continuing to spend quota."""
        response = memory.final_response
        if not response:
            if "latest_extracted_text" in memory.scratchpad:
                response = str(memory.scratchpad["latest_extracted_text"])[:400]
            elif memory.steps_taken:
                response = "I got part of the way through that, but I've used up my planning budget for this request."
            else:
                response = "I wasn't able to work out how to do that."
        return {
            "status": "planner_budget_exceeded",
            "final_response": response,
            "steps": [s.to_dict() for s in memory.steps_taken],
        }

    def _no_progress_result(self, memory: AgentMemory, snapshot) -> dict[str, Any]:
        """Stop, and name the reason the screen was not changing.

        The two causes need different answers from the user, so they get
        different sentences. An app that reports no elements at all cannot be
        driven by the accessibility path however long the loop runs, and saying
        "I couldn't do that" invites them to simply ask again; naming the app
        tells them to try a different route.
        """
        app = _window_title(snapshot) or "that window"

        if not _has_elements(snapshot):
            response = (
                f"I can't read anything inside {app} - it doesn't report its "
                f"contents to Windows, so I can't see what to click. I've "
                f"stopped rather than keep guessing."
            )
        else:
            response = (
                f"I've tried the same thing several times in {app} without "
                f"getting anywhere, so I've stopped."
            )

        return {
            "status": "no_progress",
            "final_response": memory.final_response or response,
            "steps": [s.to_dict() for s in memory.steps_taken],
        }

    def _timeout_result(self, memory: AgentMemory) -> dict[str, Any]:
        """Stop on the clock, and say what was reached rather than what failed.

        Named terminal states are the point of A4: "I ran out of time" with no
        account of where it got to is barely better than the loop never stopping,
        because the user still has to go and look at the screen to find out what
        state their desktop was left in - and this user navigates by voice.
        """
        last = memory.steps_taken[-1] if memory.steps_taken else None
        where = f" The last thing I did was `{last.action}`." if last is not None else ""
        return {
            "status": "timed_out",
            "final_response": memory.final_response or (
                f"I spent {memory.elapsed_seconds:.0f} seconds on that without "
                f"finishing, so I've stopped.{where}"
            ),
            "steps": [s.to_dict() for s in memory.steps_taken],
        }

    def _cancel_result(self, memory: AgentMemory) -> dict[str, Any]:
        """Stop because the user said so, not because a budget ran out.

        Same envelope `_timeout_result` returns: a spoken acknowledgement plus
        whatever steps already happened, so main.py's ordinary end-of-turn
        handling - speak the response, then ConversationFinished/Idle - needs
        no special case for a voice-cancelled goal.
        """
        last = memory.steps_taken[-1] if memory.steps_taken else None
        where = f" The last thing I did was `{last.action}`." if last is not None else ""
        return {
            "status": "cancelled",
            "final_response": memory.final_response or f"Okay, I've stopped.{where}",
            "steps": [s.to_dict() for s in memory.steps_taken],
        }

    def _plan_failure_result(self, memory: AgentMemory) -> dict[str, Any]:
        """Stop and say so, rather than retrying a planner that is not answering.

        Deliberately says the trouble is on Grace's side: the usual cause is the
        LLM being unreachable or refusing the API key, and "I couldn't
        understand you" would send the user off rewording a request that was
        never the problem.
        """
        response = memory.final_response or (
            "I'm having trouble planning that right now - I can't reach my "
            "language model. Please check the connection and try again."
        )
        return {
            "status": "planner_failed",
            "final_response": response,
            "steps": [s.to_dict() for s in memory.steps_taken],
        }

    async def _emit(self, event: dict[str, Any]) -> None:
        if not self._ws_server:
            return
        try:
            await self._ws_server.emit(event)
        except Exception as e:
            logger.debug(f"Failed to emit UI event: {e}")

    # -- verification ------------------------------------------------------

    def _verify_goal_completion(
        self,
        user_goal: str,
        snapshot,
        memory: AgentMemory,
        claiming_step: Optional[PlannedStep] = None,
        claiming_result: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, Optional[str]]:
        """Block a claimed completion that no action could have produced.

        Deliberately a narrow check. It cannot tell whether the goal was
        *achieved* - that needs to understand the goal - only whether the claim
        is compatible with what happened. Two things are incompatible with it:

        * nothing was ever done to the desktop;
        * the last thing done to the desktop reported that it failed, and the
          screen is readable enough that the planner had no excuse for reading
          the failure as success.

        The second is the case the loop actually hit: a step returned
        ``wrong_focus``, the planner announced the goal was complete anyway, and
        nothing disagreed. ``snapshot`` is consulted rather than ignored,
        because on a window Grace cannot read a claim of success is not
        contradicted by anything - and blocking it there would be inventing
        evidence, which is the mistake that stopped a working run.

        ``claiming_step``/``claiming_result`` are the step making the claim and
        its own dispatch result. This runs before ``memory.add_step`` records
        that step, so without them the check judges a completing action only by
        whatever came *before* it - a successful first-and-last action had no
        history to pass on, and a failing one was judged by an earlier, possibly
        successful, step instead of its own failure.
        """
        if _CONVERSATIONAL_RE.search(user_goal or ""):
            return True, None

        interactive = [s for s in memory.steps_taken if s.action in INTERACTIVE_TOOLS]
        if claiming_step is not None and claiming_step.action in INTERACTIVE_TOOLS:
            interactive = interactive + [SimpleNamespace(
                action=claiming_step.action, result=claiming_result or {},
            )]

        if not interactive:
            return False, (
                "Goal observation check: no desktop interaction has been performed yet. "
                "Inspect what is on screen and execute the next action."
            )

        if getattr(snapshot, "is_blind", False):
            # No channel to check against. Trust the claim rather than
            # manufacture a verdict from an absence of evidence.
            return True, None

        last = interactive[-1]
        failure = _failure_reason(last.result)
        if failure:
            return False, (
                f"Goal observation check: your last action (`{last.action}`) reported "
                f"failure - {failure} - so the goal cannot be complete. Address that "
                f"before finishing."
            )

        return True, None

    # -- retained for compatibility ---------------------------------------

    def _parse_llm_step(self, text: str) -> Optional[dict[str, Any]]:
        """Parse either planner JSON or native UI-TARS Thought/Action text."""
        step = parse_planned_step(text)
        if step is not None:
            return step.to_dict()
        return UITarsParser.parse_response(text)


def _window_title(snapshot) -> str:
    window = getattr(snapshot, "active_window", None)
    return getattr(window, "title", "") if window is not None else ""


def _failure_reason(result: Optional[dict[str, Any]]) -> Optional[str]:
    """Why a dispatch failed, or None if it did not.

    One definition of "this failed", because there are two shapes of failure -
    the dispatcher's ``{"status": "error"}`` envelope and the handler's
    ``{"ok": False}`` body - and a caller that checks only one of them treats
    half of all failures as successes.
    """
    if not result:
        return None

    inner = result.get("result") if isinstance(result.get("result"), dict) else {}
    if result.get("status") != "error" and inner.get("ok") is not False:
        return None

    return (
        result.get("error")
        or inner.get("error")
        or inner.get("message")
        or "no reason given"
    )


def _step_signature(step: PlannedStep) -> str:
    """Identifies "the same action again".

    The window the step names is excluded: its handle and title wobble between
    snapshots of the same application, and treating that as a different action
    would let an identical retry loop slip through the repeat check.
    """
    params = {k: v for k, v in sorted(step.params.items()) if k != "window"}
    return f"{step.action}:{params}"


def _short_params(params: dict[str, Any]) -> str:
    """Enough of a step's parameters to recognise it again, and no more."""
    interesting = {k: v for k, v in params.items() if k != "window"}
    text = str(interesting)
    return text if len(text) <= 90 else text[:87] + "…"


def _has_elements(snapshot) -> bool:
    """Whether the snapshot describes anything the planner could act on.

    Mirrors the choice `ScreenSnapshot.to_markdown` makes: the element graph if
    it has anything in it, the legacy UIA list otherwise. False means the window
    reported no contents at all, which is the signature of an app that does not
    implement UI Automation rather than one that is merely busy.
    """
    graph = getattr(snapshot, "graph", None)
    if graph is not None and len(graph):
        return True
    return bool(getattr(snapshot, "ui_elements", None))


def _rung(attempt: int, before_stop: int) -> str:
    """Which strategy this attempt gets.

    `before_stop` is how many strategies are tried before giving up, so with the
    default of 3 an action is attempted normally, then re-grounded, then
    re-planned by a stronger model, and only then abandoned. It used to be the
    number of identical retries before stopping - the same budget, spent on
    repetition rather than on different approaches.
    """
    if attempt >= max(1, before_stop) + 1:
        return "stop"
    index = min(attempt, len(ESCALATION_LADDER)) - 1
    return ESCALATION_LADDER[max(0, index)]


def _config_str(field: str, default: str) -> str:
    try:
        from grace.config import Config

        value = getattr(Config(), field, default)
        return value if value is not None else default
    except Exception:
        return default


def _config_int(field: str, default: int) -> int:
    try:
        from grace.config import Config

        return int(getattr(Config(), field, default) or default)
    except Exception:
        return default
