"""Safety Guard for Grace Agentic Loop.

Enforces Project Grace Safety & Privacy Spec by intercepting destructive
or irreversible actions (file deletion, app closure with unsaved work, system locking)
and requiring explicit voice confirmation before execution.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("grace.agent.safety")


class SafetyGuard:
    """Evaluates proposed agent actions and checks if user voice confirmation is required."""

    # Actions that ALWAYS require user confirmation before execution
    CONFIRMATION_REQUIRED_TOOLS = {
        "delete_file",
        "close_app",
        "lock_computer",
    }

    @classmethod
    def evaluate(cls, action: str, params: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Evaluate action.

        Returns:
            (is_safe: bool, confirmation_prompt: Optional[str])
        """
        if action in cls.CONFIRMATION_REQUIRED_TOOLS:
            prompt = cls._build_confirmation_prompt(action, params)
            logger.warning(f"Safety guard intercepted action '{action}': {prompt}")
            return False, prompt

        # Parameter checks for generic tools
        if action == "cua_press_key" and params.get("key") in ("alt+f4", "ctrl+w"):
            prompt = "Closing windows can cause loss of unsaved work. Should I proceed?"
            return False, prompt

        return True, None

    @classmethod
    def _build_confirmation_prompt(cls, action: str, params: dict[str, Any]) -> str:
        """Construct human-understandable confirmation question for voice output."""
        if action == "delete_file":
            filename = params.get("name") or params.get("path") or "this file"
            return f"Are you sure you want me to delete {filename}?"
        elif action == "close_app":
            app_name = params.get("name") or "this application"
            return f"Should I close {app_name}? Any unsaved changes may be lost."
        elif action == "lock_computer":
            return "Should I lock your computer now?"
        return f"Confirm executing action '{action}'?"
