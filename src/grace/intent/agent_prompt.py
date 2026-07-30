"""Agentic system prompt.

Kept as a thin wrapper over the planner's prompt so there is one description of
the agent's job rather than two contradictory ones: this module used to end
with "Do NOT output JSON" while `agent/loop.py` appended "Output ONLY valid
JSON inside ```json fences" to the very same request.

The planner speaks JSON; the grounder speaks UI-TARS's native Thought/Action.
Each prompt is now internally consistent, and neither is asked to do the
other's job.
"""

from grace.agent.planner import get_planner_system_prompt


def get_agentic_system_prompt() -> str:
    """Return the planner's system prompt."""
    return get_planner_system_prompt()
