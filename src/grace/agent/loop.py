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
from typing import Any, Optional

from grace.agent.grounder import Grounder, describe_target
from grace.agent.memory import AgentMemory
from grace.agent.perception import PerceptionEngine
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

DEFAULT_MAX_ITERATIONS = 12


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
        self._planner = planner or Planner(gemma, max_calls=_config_int("planner_max_calls_per_goal", 8))
        self._grounder = grounder or Grounder(self._vision_llm, on_demand_start=start_grounding_backend)
        self._max_iterations = max_iterations or _config_int("agent_max_iterations", DEFAULT_MAX_ITERATIONS)
        self._pending: Optional[dict[str, Any]] = None

    # -- safety resumption -------------------------------------------------

    @property
    def has_pending_confirmation(self) -> bool:
        """True when a step is parked waiting for the user to say yes or no.

        SafetyGuard used to set memory.safety_pending and return, with nothing
        anywhere able to resume it - so answering "yes" started an unrelated new
        request and the confirmed action never ran.
        """
        return self._pending is not None

    @property
    def pending_prompt(self) -> Optional[str]:
        return self._pending.get("prompt") if self._pending else None

    def cancel_pending(self) -> None:
        self._pending = None

    async def resume_pending(self, approved: bool) -> dict[str, Any]:
        """Continue a goal that stopped for a safety confirmation."""
        if not self._pending:
            return {"status": "error", "error": "Nothing is waiting for confirmation."}

        pending = self._pending
        self._pending = None
        memory: AgentMemory = pending["memory"]
        step: PlannedStep = pending["step"]

        if not approved:
            memory.safety_pending = None
            memory.is_completed = True
            memory.final_response = "Alright, I won't do that."
            memory.add_step(step.thought, "converse", {}, {"status": "cancelled"}, "Cancelled")
            return self._result(memory)

        logger.info(f"Resuming confirmed action '{step.action}'")
        memory.safety_pending = None
        exec_result = await self._dispatch(step, memory)
        memory.add_step(step.thought, step.action, step.params, exec_result, step.user_update)
        return await self._continue(memory, last_step=step, last_result=exec_result)

    # -- main entry point --------------------------------------------------

    async def run(self, user_goal: str, max_iterations: Optional[int] = None) -> dict[str, Any]:
        """Run the autonomous Observe-Plan-Act loop for a given user goal."""
        limit = max_iterations or self._max_iterations
        memory = AgentMemory(user_goal=user_goal, max_iterations=limit)
        self._planner.reset()
        self._pending = None
        logger.info(f"AgentLoop started for goal: '{user_goal}' (max {limit} steps)")
        return await self._continue(memory)

    async def _continue(
        self,
        memory: AgentMemory,
        last_step: Optional[PlannedStep] = None,
        last_result: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Drive the loop until completion, the step cap, or a confirmation."""
        while not memory.is_completed and not memory.is_exceeded:
            step_no = memory.current_iteration + 1

            async with stage(f"observe#{step_no}"):
                snapshot = await self._observe()

            expectation_note = self._expectation_note(last_step, last_result, snapshot)

            try:
                async with stage(f"plan#{step_no}") as plan_stage:
                    step = await self._planner.plan(
                        goal=memory.user_goal,
                        elements_prompt=snapshot.to_markdown(),
                        history=memory.format_history_markdown(),
                        scratchpad=memory.format_scratchpad_markdown(),
                        expectation_note=expectation_note,
                        window_title=_window_title(snapshot),
                    )
                    plan_stage.detail(
                        f"{step.action if step else 'unparsed'} "
                        f"(call {self._planner.calls_made})"
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
                continue

            logger.info(f"AgentLoop step {step_no}: [{step.action}] {step.thought}")

            is_safe, confirm_prompt = SafetyGuard.evaluate(step.action, step.params)
            if not is_safe:
                logger.warning(f"AgentLoop: safety confirmation required for '{step.action}'")
                memory.safety_pending = {
                    "action": step.action,
                    "params": step.params,
                    "prompt": confirm_prompt,
                }
                self._pending = {"memory": memory, "step": step, "prompt": confirm_prompt}
                return {
                    "status": "safety_confirmation_required",
                    "confirmation_prompt": confirm_prompt,
                    "pending_step": step.to_dict(),
                    "steps": [s.to_dict() for s in memory.steps_taken],
                }

            if step.needs_grounding:
                async with stage(f"ground#{step_no}"):
                    await self._ground(step, snapshot)

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
                verified, hint = self._verify_goal_completion(memory.user_goal, snapshot, memory)
                if verified:
                    memory.is_completed = True
                    memory.final_response = self._final_response(step, memory)
                    memory.add_step(step.thought, step.action, step.params, exec_result, step.user_update)
                    break
                logger.warning(f"AgentLoop: completion rejected by verification guard: {hint}")
                memory.set_scratchpad("verification_hint", hint)
                step.is_completed = False

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

    async def _dispatch(self, step: PlannedStep, memory: AgentMemory) -> dict[str, Any]:
        if step.action == "converse":
            return {"status": "ok"}
        intent = Intent(tool=step.action, params=step.params)
        async with stage(f"dispatch:{step.action}"):
            return await self._dispatcher.execute(intent)

    async def _ground(self, step: PlannedStep, snapshot) -> None:
        """Fill in x/y for a target the element graph could not resolve."""
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
        failed = status == "error" or inner.get("ok") is False

        if failed:
            reason = last_result.get("error") or inner.get("error") or inner.get("message") or "no reason given"
            lines.append(f"IT FAILED: {reason}")
            lines.append("Do not repeat this action unchanged. Try a different element or a different approach.")
        else:
            lines.append(f"The tool reported: {inner.get('message') or status}")

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

        lines.append("Judge from the elements list above whether your expectation actually came true.")
        return "\n".join(lines)

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

    async def _emit(self, event: dict[str, Any]) -> None:
        if not self._ws_server:
            return
        try:
            await self._ws_server.emit(event)
        except Exception as e:
            logger.debug(f"Failed to emit UI event: {e}")

    # -- verification ------------------------------------------------------

    def _verify_goal_completion(self, user_goal: str, snapshot, memory: AgentMemory) -> tuple[bool, Optional[str]]:
        """Block a claimed completion that no action could have produced.

        Questions do not need desktop state to change, so they pass straight
        through. Anything else must have actually done something.
        """
        if _CONVERSATIONAL_RE.search(user_goal or ""):
            return True, None

        if not any(s.action in INTERACTIVE_TOOLS for s in memory.steps_taken):
            return False, (
                "Goal observation check: no desktop interaction has been performed yet. "
                "Inspect the elements list and execute the next action."
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


def _config_int(field: str, default: int) -> int:
    try:
        from grace.config import Config

        return int(getattr(Config(), field, default) or default)
    except Exception:
        return default
