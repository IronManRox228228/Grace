"""Agentic Control Loop (ReAct Engine) for Grace.

Executes an Observe-Plan-Act iteration loop to accomplish high-level user goals.
Integrates PerceptionEngine, GemmaClient, Dispatcher, SafetyGuard, and AgentMemory.
"""

import asyncio
import base64
import json
import logging
from typing import Any, Optional

from grace.agent.memory import AgentMemory
from grace.agent.perception import PerceptionEngine
from grace.agent.safety import SafetyGuard
from grace.agent.ui_tars_parser import UITarsParser
from grace.intent.agent_prompt import get_agentic_system_prompt
from grace.intent.parser import Intent, clean_json_fence
from grace.llm.gemma_client import GemmaClient
from grace.tools.dispatcher import Dispatcher

logger = logging.getLogger("grace.agent.loop")


class AgentLoop:
    """Autonomous ReAct Execution Engine for complex multi-step tasks."""

    def __init__(
        self,
        gemma: GemmaClient,
        dispatcher: Dispatcher,
        perception: Optional[PerceptionEngine] = None,
        ws_server: Optional[Any] = None,
        vision_llm: Optional[GemmaClient] = None,
    ):
        self._gemma = gemma
        self._vision_llm = vision_llm or gemma
        self._dispatcher = dispatcher
        self._perception = perception or PerceptionEngine()
        self._ws_server = ws_server
        self._system_prompt = get_agentic_system_prompt()

    async def run(self, user_goal: str, max_iterations: int = 100) -> dict[str, Any]:
        """Run the autonomous Observe-Plan-Act loop for a given user goal.

        Returns result dict with status, final_response, and steps history.
        """
        memory = AgentMemory(user_goal=user_goal, max_iterations=max_iterations)
        logger.info(f"AgentLoop started for goal: '{user_goal}'")

        while not memory.is_completed and not memory.is_exceeded:
            # 1. Observe screen state
            snap_res = self._perception.capture_snapshot_async()
            if asyncio.iscoroutine(snap_res):
                snapshot = await snap_res
            else:
                snapshot = self._perception.capture_snapshot()
            perception_md = snapshot.to_markdown()

            prompt_text = (
                f"### User Goal:\n{memory.user_goal}\n\n"
                f"### Current Desktop Perception Snapshot:\n{perception_md}\n\n"
                f"### Action History:\n{memory.format_history_markdown()}\n\n"
                f"### Scratchpad Data:\n{memory.format_scratchpad_markdown()}\n\n"
                "Decide the next step. Output ONLY valid JSON inside ```json ... ``` code fences."
            )

            image_b64 = None
            if snapshot.png_bytes:
                image_b64 = base64.b64encode(snapshot.png_bytes).decode("utf-8")

            # 3. Request LLM step decision via Vision LLM (UI-TARS local or fallback)
            llm_output = await self._vision_llm.generate_text(
                prompt=prompt_text,
                system_prompt=self._system_prompt,
                temperature=0.2,
                max_tokens=1024,
                image_b64=image_b64,
            )

            if not llm_output:
                logger.error("AgentLoop: LLM returned empty response")
                return {
                    "status": "error",
                    "error": "LLM failed to return a valid step plan.",
                    "steps": [s.to_dict() for s in memory.steps_taken],
                }

            # 4. Parse step decision (with retry loop for format robustness)
            step_data = self._parse_llm_step(llm_output)
            json_retries = 0
            while not step_data and json_retries < 2:
                json_retries += 1
                logger.warning(f"AgentLoop: Failed to parse LLM action (attempt {json_retries}), retrying...")
                prev_snippet = (llm_output or "")[:200]
                retry_prompt = (
                    f"{prompt_text}\n\n"
                    f"### Error:\nYour previous response could not be parsed:\n{prev_snippet}\n"
                    "Output your decision using the exact Thought: ...\\nAction: ... format."
                )
                llm_output = await self._vision_llm.generate_text(
                    prompt=retry_prompt,
                    system_prompt=self._system_prompt,
                    temperature=0.1,
                    max_tokens=1024,
                    image_b64=image_b64,
                )
                step_data = self._parse_llm_step(llm_output) if llm_output else None

            if not step_data:
                logger.error(f"AgentLoop: Failed to parse LLM step after retries: {(llm_output or 'None')[:100]}")
                memory.set_scratchpad("last_error", "Invalid output from LLM")
                continue  # Skip to next iteration instead of aborting whole loop

            thought = step_data.get("thought", "")
            action = step_data.get("action", "converse")
            params = step_data.get("params", {})
            user_update = step_data.get("user_update", f"Executing {action}...")
            is_completed = step_data.get("is_completed", False)
            final_resp = step_data.get("final_response") or params.get("response")

            logger.info(f"AgentLoop Step {memory.current_iteration + 1}: [{action}] '{thought}'")

            # 5. Check Safety Guard
            is_safe, confirm_prompt = SafetyGuard.evaluate(action, params)
            if not is_safe:
                logger.warning(f"AgentLoop: Safety confirmation required for '{action}'")
                memory.safety_pending = {
                    "action": action,
                    "params": params,
                    "prompt": confirm_prompt,
                }
                return {
                    "status": "safety_confirmation_required",
                    "confirmation_prompt": confirm_prompt,
                    "pending_step": step_data,
                    "steps": [s.to_dict() for s in memory.steps_taken],
                }

            # 6. Broadcast progress update to UI overlay
            if self._ws_server:
                try:
                    await self._ws_server.emit(
                        {
                            "type": "ToolExecutionStarted",
                            "label": user_update,
                            "tool": action,
                            "step": memory.current_iteration + 1,
                        }
                    )
                except Exception as e:
                    logger.debug(f"Failed to emit UI event: {e}")

            # 7. Execute action via Dispatcher (BEFORE checking completion so final step action is never skipped)
            exec_result = {"status": "ok"}
            if action != "converse":
                intent = Intent(tool=action, params=params)
                exec_result = await self._dispatcher.execute(intent)

            # 8. Check if task completed (with Empirical Verification Guard)
            if is_completed or action == "converse":
                verified, hint = self._verify_goal_completion(user_goal, snapshot, memory)
                if not verified:
                    logger.warning(f"AgentLoop: Goal completion rejected by verification guard: {hint}")
                    memory.set_scratchpad("verification_hint", hint)
                    is_completed = False
                else:
                    memory.is_completed = True
                    if not final_resp or final_resp == thought:
                        if "open_windows" in memory.scratchpad:
                            wins = memory.scratchpad["open_windows"]
                            final_resp = f"The open applications are: {', '.join(wins[:5])}."
                        elif "open_apps" in memory.scratchpad:
                            apps = memory.scratchpad["open_apps"]
                            final_resp = f"The open applications are: {', '.join(apps[:5])}."
                        else:
                            final_resp = "Goal completed."
                    memory.final_response = final_resp
                    memory.add_step(thought, action, params, exec_result, user_update)
                    break

            # 9. Automatic Data Flow to Scratchpad for document / search outputs
            if exec_result.get("status") == "ok":
                res = exec_result.get("result", exec_result)
                if isinstance(res, dict):
                    if "windows" in res:
                        titles = [w.get("title", "") for w in res["windows"] if w.get("title")]
                        memory.set_scratchpad("open_windows", titles)
                    if "apps" in res:
                        app_names = [a.get("name", "") for a in res["apps"] if a.get("name")]
                        memory.set_scratchpad("open_apps", app_names)
                if "text" in exec_result:
                    memory.set_scratchpad("latest_extracted_text", exec_result["text"])
                if "summary" in exec_result:
                    memory.set_scratchpad("latest_summary", exec_result["summary"])
                if "files" in exec_result:
                    memory.set_scratchpad("found_files", exec_result["files"])

            # 10. Record step in memory
            memory.add_step(thought, action, params, exec_result, user_update)

            # 11. Broadcast step completed to UI
            if self._ws_server:
                try:
                    await self._ws_server.emit(
                        {
                            "type": "ToolExecutionFinished",
                            "tool": action,
                            "status": exec_result.get("status", "ok"),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Failed to emit UI event: {e}")

        if memory.is_exceeded and not memory.is_completed:
            return {
                "status": "max_iterations_reached",
                "final_response": "I reached the step limit before fully completing the goal.",
                "steps": [s.to_dict() for s in memory.steps_taken],
            }

        return {
            "status": "ok",
            "final_response": memory.final_response,
            "steps": [s.to_dict() for s in memory.steps_taken],
        }

    def _parse_llm_step(self, text: str) -> Optional[dict[str, Any]]:
        """Extract and parse JSON or UI-TARS step object from LLM output text."""
        json_str = clean_json_fence(text)
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "action" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Robust regex fallback for JSON
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict) and "action" in data:
                    return data
        except Exception:
            pass

        # Native UI-TARS Thought/Action format parser fallback
        ui_tars_parsed = UITarsParser.parse_response(text)
        if ui_tars_parsed:
            return ui_tars_parsed

        return None

    def _verify_goal_completion(self, user_goal: str, snapshot, memory: AgentMemory) -> tuple[bool, Optional[str]]:
        """General Intent Convergence Check (Hermes Agent Methodology).

        Verifies that the observed Accessibility Tree & Window State explicitly reflects
        the outcome requested by the user, without any hardcoded app domain hacks.
        """
        q = user_goal.lower().strip()

        # Conversational / knowledge queries do not require desktop state convergence
        if any(c in q for c in ("what is", "who is", "tell me", "explain", "hello", "hi", "how are", "calculate")):
            return True, None

        # Block completion if no interactive desktop tool has ever been executed
        INTERACTIVE_TOOLS = {
            "cua_click", "cua_type_text", "cua_press_key", "cua_scroll",
            "cua_drag", "cua_launch", "cua_activate", "cua_set_value",
            "open_app", "close_app", "open_file"
        }
        has_executed_action = any(s.action in INTERACTIVE_TOOLS for s in memory.steps_taken)
        if not has_executed_action:
            return False, "Goal observation check: No desktop interaction performed yet. Inspect Accessibility Tree controls and execute the next action."

        return True, None

