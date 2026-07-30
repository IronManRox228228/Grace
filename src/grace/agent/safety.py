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

        # Parameter checks for generic tools. Normalised because the model
        # writes "Alt+F4" and "Control_L+w" at least as often as "alt+f4",
        # and the old exact-string comparison let those straight through.
        if action == "cua_press_key":
            key = cls._normalise_key(params.get("key"))
            if key in cls.CONFIRMATION_REQUIRED_KEYS:
                return False, "Closing windows can cause loss of unsaved work. Should I proceed?"

        return True, None

    # Hotkeys that close or discard work, in normalised form.
    CONFIRMATION_REQUIRED_KEYS = {
        "alt+f4",
        "ctrl+w",
        "ctrl+shift+w",
        "ctrl+q",
    }

    # X11-style keysyms the CUA layer accepts, mapped to their plain names.
    _KEY_ALIASES = {
        "control_l": "ctrl", "control_r": "ctrl", "control": "ctrl",
        "shift_l": "shift", "shift_r": "shift",
        "alt_l": "alt", "alt_r": "alt",
        "super_l": "win", "super_r": "win",
    }

    @classmethod
    def _normalise_key(cls, raw: Any) -> str:
        """Lower-case, de-alias, and sort modifiers so orderings compare equal."""
        if not raw or not isinstance(raw, str):
            return ""
        parts = [cls._KEY_ALIASES.get(p, p) for p in
                 (piece.strip().lower() for piece in raw.split("+")) if p]
        if not parts:
            return ""
        modifiers = sorted(p for p in parts[:-1])
        return "+".join(modifiers + [parts[-1]])

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
