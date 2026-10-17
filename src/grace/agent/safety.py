"""Safety Guard for Grace Agentic Loop.

Enforces Project Grace Safety & Privacy Spec by intercepting destructive
or irreversible actions (file deletion, app closure with unsaved work, system locking)
and requiring explicit voice confirmation before execution.
"""

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger("grace.agent.safety")


class SafetyGuard:
    """Evaluates proposed agent actions and checks if user voice confirmation is required.

    Uses an explicit allowlist of read-only and harmless actions. Everything else
    requires explicit user confirmation.
    """

    # Actions that ALWAYS require user confirmation before execution
    CONFIRMATION_REQUIRED_TOOLS = {
        "delete_file",
        "close_app",
        "lock_computer",
    }

    # Explicit allowlist of read-only or harmless actions that run without confirmation.
    # Note: read_pdf and summarize_pdf are checked separately to enforce safe root paths (R26).
    SAFE_READ_ONLY_TOOLS = {
        "converse",
        "search_files",
        "adjust_volume",
        "open_calculator",
        "undo",
        "describe_screen",
        "set_speech_rate",
        "voice_training_status",
        "rollback_voice_model",
        "cua_list_windows",
        "cua_get_window",
        "cua_screenshot",
        "cua_get_cursor_position",
        "cua_scroll",
        "cua_hover",
        "cua_move_cursor",
        "cua_drag",
        "cua_set_value",
        "cua_activate",
        "cua_launch",
        "cua_secondary_action",
    }

    # Target words that always require confirmation (delete, send, submit, pay/purchase, taskkill, overwrite)
    DESTRUCTIVE_KEYWORDS = {
        "delete", "remove", "send", "submit", "pay", "purchase", "buy",
        "order", "checkout", "format", "erase", "kill", "taskkill", "overwrite",
    }

    # Extensions that can run code or a script just by being "opened" - an
    # `open_file` naming one of these is refused unless the user has actually
    # said yes (R2). `open_file` is otherwise a fast-path tool with no
    # confirmation gate at all, so this is the only thing standing between a
    # misheard/hallucinated/injected name and silent code execution.
    DANGEROUS_OPEN_EXTENSIONS = {
        ".exe", ".bat", ".cmd", ".com", ".ps1", ".vbs", ".js", ".jse", ".wsf",
        ".msi", ".lnk", ".scr", ".pif", ".hta", ".reg", ".cpl",
    }

    @classmethod
    def evaluate(cls, action: str, params: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Evaluate action.

        Returns:
            (is_safe: bool, confirmation_prompt: Optional[str])
        """
        # 1. Explicit always-confirm tools
        if action in cls.CONFIRMATION_REQUIRED_TOOLS:
            prompt = cls._build_confirmation_prompt(action, params)
            logger.warning(f"Safety guard intercepted action '{action}': {prompt}")
            return False, prompt

        # 2. open_file with dangerous extension
        if action == "open_file":
            name = params.get("name") or params.get("path") or ""
            if cls.has_dangerous_open_extension(name):
                prompt = f"{name} can run code just by being opened. Should I open it anyway?"
                logger.warning(f"Safety guard intercepted action '{action}': {prompt}")
                return False, prompt
            return True, None

        # 3. open_app is harmless app launch
        if action == "open_app":
            return True, None

        # 4. cua_press_key: safe unless a dangerous hotkey
        if action == "cua_press_key":
            key = cls._normalise_key(params.get("key"))
            if key in cls.CONFIRMATION_REQUIRED_KEYS:
                return False, "Closing windows can cause loss of unsaved work. Should I proceed?"
            return True, None

        # 5. cua_click: safe ONLY if target does NOT have sensitive/destructive keywords
        if action == "cua_click":
            target = (params.get("target_name") or params.get("describe") or "").lower()
            for kw in cls.DESTRUCTIVE_KEYWORDS:
                if re.search(r"\b" + re.escape(kw) + r"\b", target):
                    prompt = cls._build_confirmation_prompt("cua_click", params, keyword=kw)
                    logger.warning(f"Safety guard intercepted click on destructive target '{target}': {prompt}")
                    return False, prompt
            return True, None

        # 6. cua_type_text: safe unless overwriting a file
        if action == "cua_type_text":
            if params.get("replace_file") or params.get("overwrite"):
                return False, "This action may overwrite an existing file. Should I proceed?"
            return True, None

        # 7. read_pdf and summarize_pdf: safe ONLY if within safe roots (R26)
        if action in ("read_pdf", "summarize_pdf"):
            path = params.get("path") or ""
            if not cls.is_within_safe_pdf_roots(path):
                prompt = f"Reading '{os.path.basename(path) or path}' from outside your documents folders requires confirmation. Should I proceed?"
                logger.warning(f"Safety guard intercepted {action} outside safe roots: {prompt}")
                return False, prompt
            return True, None

        # 8. Explicit read-only harmless tools on the allowlist
        if action in cls.SAFE_READ_ONLY_TOOLS:
            return True, None

        # 9. Everything else (unallowed / unknown / destructive) requires confirmation
        prompt = cls._build_confirmation_prompt(action, params)
        logger.warning(f"Safety guard intercepted unallowed action '{action}': {prompt}")
        return False, prompt

    @classmethod
    def is_within_safe_pdf_roots(cls, path_str: str) -> bool:
        """True when path resolves inside Documents, Desktop, Downloads, or temp dir (R26)."""
        if not path_str:
            return False
        import tempfile
        user_profile = os.environ.get("USERPROFILE", os.path.expanduser("~"))
        safe_dirs = [
            os.path.realpath(os.path.join(user_profile, "Documents")),
            os.path.realpath(os.path.join(user_profile, "Desktop")),
            os.path.realpath(os.path.join(user_profile, "Downloads")),
            os.path.realpath(tempfile.gettempdir()),
        ]
        real = os.path.realpath(path_str)
        return any(real == root or real.startswith(root + os.sep) for root in safe_dirs)

    @classmethod
    def has_dangerous_open_extension(cls, name_or_path: str) -> bool:
        """True when a name/path ends in an extension that can execute code.

        A free function of just the extension, not full path resolution, so
        `dispatcher._open_file` can re-run this same check on the path it
        actually resolves to (which may reveal an extension the raw model
        text did not show) without a circular import back into dispatcher.
        """
        return os.path.splitext(name_or_path or "")[1].lower() in cls.DANGEROUS_OPEN_EXTENSIONS

    # Hotkeys and keys that close, delete, or discard work, in normalised form.
    CONFIRMATION_REQUIRED_KEYS = {
        "alt+f4",
        "ctrl+w",
        "ctrl+shift+w",
        "ctrl+q",
        "ctrl+f4",
        "delete+shift",
        "l+win",
        "delete",
    }

    # X11-style keysyms and pyautogui's own native names, both mapped to their
    # plain canonical form. The planner (and pyautogui itself) uses either
    # depending on the path a key press took, so only aliasing one family let
    # requests like "altleft+f4" or "ctrlleft+w" slip past this guard unseen.
    _KEY_ALIASES = {
        "control_l": "ctrl", "control_r": "ctrl", "control": "ctrl",
        "shift_l": "shift", "shift_r": "shift",
        "alt_l": "alt", "alt_r": "alt",
        "super_l": "win", "super_r": "win",
        "ctrlleft": "ctrl", "ctrlright": "ctrl",
        "altleft": "alt", "altright": "alt",
        "shiftleft": "shift", "shiftright": "shift",
        "winleft": "win", "winright": "win", "super": "win",
    }

    @classmethod
    def _normalise_key(cls, raw: Any) -> str:
        """Lower-case, de-alias, and sort all keys so orderings compare equal (R38)."""
        if not raw or not isinstance(raw, str):
            return ""
        parts = [cls._KEY_ALIASES.get(p, p) for p in
                 (piece.strip().lower() for piece in raw.split("+")) if p]
        if not parts:
            return ""
        parts.sort()
        return "+".join(parts)

    @classmethod
    def _build_confirmation_prompt(cls, action: str, params: dict[str, Any], keyword: str = "") -> str:
        """Construct human-understandable confirmation question for voice output."""
        if action == "delete_file":
            filename = params.get("name") or params.get("path") or "this file"
            return f"Are you sure you want me to delete {filename}?"
        elif action == "close_app":
            app_name = params.get("name") or "this application"
            return f"Should I close {app_name}? Any unsaved changes may be lost."
        elif action == "lock_computer":
            return "Should I lock your computer now?"
        elif action == "cua_click":
            target = params.get("target_name") or params.get("describe") or "this element"
            if keyword in ("pay", "purchase", "buy", "order", "checkout"):
                return f"Clicking '{target}' will proceed with a purchase or payment. Should I proceed?"
            if keyword in ("send", "submit"):
                return f"Should I submit or send by clicking '{target}'?"
            if keyword in ("delete", "remove", "erase", "format", "kill"):
                return f"Are you sure you want me to delete or remove by clicking '{target}'?"
            return f"Should I click '{target}'?"
        elif "delete" in action or "remove" in action:
            return f"Are you sure you want to execute {action} to delete items?"
        elif "send" in action or "submit" in action:
            return f"Should I proceed with submitting or sending?"
        elif "pay" in action or "purchase" in action:
            return "This will proceed with a purchase or payment. Should I proceed?"
        return f"Confirm executing action '{action}'?"
