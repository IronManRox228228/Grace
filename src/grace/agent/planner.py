"""The planner: decides *what* to do next, from structure rather than pixels.

Grace used to make one model call per step that had to decompose the goal,
choose from a 26-tool schema, *and* work out screen coordinates from an image.
That is three different jobs, and the model did all three badly - most visibly
by typing a YouTube search into the browser's address bar.

This module does only the first two, and it does them from the element graph:
role, name, placeholder, container, and `frame` ("chrome" vs "page") are enough
to name a target unambiguously in text. No screenshot is sent. Pixel work is
delegated to the grounder, and only when a target cannot be named.

Every step also declares `expect` - what should be true afterwards. The next
observation is checked against it, which is what makes the loop able to notice
it went wrong instead of blindly retrying.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from grace.intent.parser import clean_json_fence
from grace.intent.tools import format_tools_for_prompt
from grace.llm.gemma_client import RateLimitError

logger = logging.getLogger("grace.agent.planner")

# The planner is the only thing that talks to the cloud during a goal, so the
# cap here is what actually bounds quota use per request.
DEFAULT_MAX_CALLS = 8


class PlannerBudgetExceeded(RuntimeError):
    """The per-goal call cap was hit. The loop finishes with what it knows."""


@dataclass
class PlannedStep:
    """One decision from the planner."""

    thought: str = ""
    action: str = "converse"
    params: dict[str, Any] = field(default_factory=dict)
    expect: str = ""
    is_completed: bool = False
    final_response: str = ""
    user_update: str = ""

    @property
    def needs_grounding(self) -> bool:
        """True when the planner named a target it could not pin to an element.

        This is the only condition under which UI-TARS is invoked.
        """
        if self.action not in ("cua_click", "cua_secondary_action", "cua_set_value"):
            return False
        if self.params.get("element_id") is not None:
            return False
        if self.params.get("x") is not None and self.params.get("y") is not None:
            return False
        return bool(self.params.get("target_name") or self.params.get("describe"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "thought": self.thought,
            "action": self.action,
            "params": self.params,
            "expect": self.expect,
            "is_completed": self.is_completed,
            "final_response": self.final_response,
            "user_update": self.user_update,
        }


def get_planner_system_prompt() -> str:
    """System prompt for the planner. JSON only - no Thought/Action syntax here.

    The old single prompt told the model "Do NOT output JSON" while the user
    message demanded "ONLY valid JSON inside fences". Each prompt in the split
    is now internally consistent: planner speaks JSON, grounder speaks
    UI-TARS's native Thought/Action.
    """
    return f"""You are the planner for Grace, a voice assistant that operates Windows 11 for a user who cannot use a keyboard or mouse.

You are given the user's goal, a JSON list of the interactive elements currently on screen, what you have already done, and whether the last step did what you expected. You decide the single next step.

Tools:
{format_tools_for_prompt()}

How to choose a target, in order of preference:
1. `element_id` - an `id` from the elements list. Always prefer this. It is exact.
2. `target_name` plus `frame` - when you can name the control but it is not in the list yet.
3. `x`/`y` - only for something you can see in no other way.

About `frame`:
- `"chrome"` is the browser's own UI: address bar, tabs, bookmarks, back button.
- `"page"` is the content of the website itself.
- `"app"` is a normal desktop application.
A website's own search box is ALWAYS `frame: "page"`. The browser address bar is ALWAYS `frame: "chrome"`. Typing a site's search query into the address bar is a mistake - it searches the web instead of the site.

Rules:
- One step per response. Do not plan several actions at once.
- Before typing, make sure the field you want is focused - click it first.
- `expect` must describe something you will be able to *see* in the next elements list, e.g. "the YouTube search box is focused" or "video result links are listed".
- Set `is_completed: true` only when the elements list or window title shows the goal is actually achieved. Put the spoken answer in `final_response`, in plain sentences with no JSON or markdown.
- If the last step reports it did not do what you expected, do something different. Do not repeat the same failing action.

Respond with ONLY one JSON object, no code fences, no commentary:
{{"thought": "why this step", "action": "<tool name>", "params": {{...}}, "expect": "what should be true next", "is_completed": false, "user_update": "short phrase shown to the user", "final_response": ""}}"""


class Planner:
    """Wraps the cloud model with a per-goal call budget and JSON parsing."""

    def __init__(self, llm, max_calls: int = DEFAULT_MAX_CALLS):
        self._llm = llm
        self._max_calls = max_calls
        self._calls = 0
        self._system_prompt = get_planner_system_prompt()

    @property
    def calls_made(self) -> int:
        return self._calls

    @property
    def calls_remaining(self) -> int:
        return max(0, self._max_calls - self._calls)

    def reset(self) -> None:
        """Start a new goal with a fresh budget."""
        self._calls = 0

    def build_prompt(
        self,
        goal: str,
        elements_prompt: str,
        history: str = "",
        scratchpad: str = "",
        expectation_note: str = "",
        window_title: str = "",
    ) -> str:
        sections = [f"### Goal\n{goal}"]
        if window_title:
            sections.append(f"### Active window\n{window_title}")
        sections.append(f"### Interactive elements on screen\n{elements_prompt}")
        if history:
            sections.append(f"### What you have already done\n{history}")
        if scratchpad:
            sections.append(f"### Data collected so far\n{scratchpad}")
        if expectation_note:
            # Placed last so it is the freshest thing in context.
            sections.append(f"### Result of your last step\n{expectation_note}")
        sections.append("Decide the single next step. Respond with one JSON object only.")
        return "\n\n".join(sections)

    async def plan(
        self,
        goal: str,
        elements_prompt: str,
        history: str = "",
        scratchpad: str = "",
        expectation_note: str = "",
        window_title: str = "",
    ) -> Optional[PlannedStep]:
        """Ask for the next step. Returns None if the model gave nothing usable.

        Raises PlannerBudgetExceeded when the per-goal cap is spent, and
        RateLimitError straight through so the loop can tell the user.
        """
        if self._calls >= self._max_calls:
            raise PlannerBudgetExceeded(
                f"Planner reached its {self._max_calls}-call limit for this goal"
            )

        prompt = self.build_prompt(
            goal=goal,
            elements_prompt=elements_prompt,
            history=history,
            scratchpad=scratchpad,
            expectation_note=expectation_note,
            window_title=window_title,
        )

        self._calls += 1
        # No image: this model reasons over the element JSON. Sending a
        # screenshot as well cost tokens and latency for nothing.
        raw = await self._llm.generate_text(
            prompt=prompt,
            system_prompt=self._system_prompt,
            temperature=0.1,
            max_tokens=512,
        )
        if not raw:
            logger.warning("Planner returned an empty response")
            return None

        step = parse_planned_step(raw)
        if step is None:
            logger.warning(f"Planner output was not parseable: {raw[:200]}")
        return step


def parse_planned_step(text: str) -> Optional[PlannedStep]:
    """Parse the planner's JSON, tolerating fences and surrounding prose."""
    if not text or not isinstance(text, str):
        return None

    data = _load_json_object(clean_json_fence(text))
    if data is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            data = _load_json_object(text[start : end + 1])
    if data is None:
        return None

    action = data.get("action")
    if not isinstance(action, str) or not action.strip():
        # A response with no action is not a step, even if it parsed.
        return None

    params = data.get("params")
    if not isinstance(params, dict):
        params = {}

    final_response = data.get("final_response") or ""
    if not isinstance(final_response, str):
        final_response = ""
    if not final_response and action == "converse":
        response = params.get("response")
        if isinstance(response, str):
            final_response = response

    return PlannedStep(
        thought=str(data.get("thought") or ""),
        action=action.strip(),
        params=params,
        expect=str(data.get("expect") or ""),
        is_completed=bool(data.get("is_completed")),
        final_response=final_response,
        user_update=str(data.get("user_update") or f"Running {action}…"),
    )


def _load_json_object(candidate: str) -> Optional[dict]:
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None
