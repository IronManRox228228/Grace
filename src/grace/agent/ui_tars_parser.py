"""UI-TARS Action Parser for Grace Desktop Automation.

Parses native UI-TARS 7B model text output (Thought / Action format)
into structured Grace step intents (action, params, is_completed, thought).
"""

import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("grace.agent.ui_tars_parser")


class UITarsParser:
    """Parses UI-TARS model responses into Grace step data dicts."""

    @classmethod
    def parse_response(cls, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse raw text from UI-TARS model output into Grace step plan format."""
        if not response_text or not isinstance(response_text, str):
            return None

        clean_text = response_text.strip()
        thought = ""
        action_str = clean_text

        # Separate Thought and Action sections if present
        if "Thought:" in clean_text:
            parts = clean_text.split("Action:", 1)
            thought = parts[0].replace("Thought:", "").strip()
            action_str = "Action:" + parts[1] if len(parts) > 1 else parts[0]

        if "Action:" in action_str:
            action_str = action_str.split("Action:", 1)[1].strip()

        logger.debug(f"UITarsParser: Parsed thought='{thought}', action_str='{action_str}'")

        # 1. Parse click action: click(start_box='(...)') or left_double_click(start_box='(...)')
        click_match = re.search(r"(?:left_double_click|double_click|right_click|click)\s*\(\s*(?:start_box|point|location)?\s*=\s*['\"]?\(?\s*(\d+)\s*,\s*(\d+)\s*\)?['\"]?\s*\)", action_str, re.IGNORECASE)

        if click_match:
            x, y = int(click_match.group(1)), int(click_match.group(2))
            return {
                "thought": thought or f"Clicking target at ({x}, {y})",
                "action": "cua_click",
                "params": {"x": x, "y": y},
                "user_update": f"Clicking at ({x}, {y})...",
                "is_completed": False,
            }

        # 2. Parse element click: click(target='...') or click(name='...')
        click_target_match = re.search(r"click\s*\(\s*(?:target|name|label|text)\s*=\s*['\"]([^'\"]+)['\"]\s*\)", action_str, re.IGNORECASE)
        if click_target_match:
            target_name = click_target_match.group(1)
            return {
                "thought": thought or f"Clicking '{target_name}'",
                "action": "cua_click",
                "params": {"target_name": target_name},
                "user_update": f"Clicking '{target_name}'...",
                "is_completed": False,
            }

        # 3. Parse type action: type(text='...')
        type_match = re.search(r"type\s*\(\s*(?:text|input|value)\s*=\s*['\"]([^'\"]+)['\"]\s*\)", action_str, re.IGNORECASE)
        if type_match:
            text = type_match.group(1)
            return {
                "thought": thought or f"Typing text '{text}'",
                "action": "cua_type_text",
                "params": {"text": text},
                "user_update": f"Typing '{text}'...",
                "is_completed": False,
            }

        # 4. Parse keypress / hotkey action: hotkey(key='...') or press_key(key='...')
        key_match = re.search(r"(?:hotkey|press_key|key)\s*\(\s*(?:key|name)\s*=\s*['\"]([^'\"]+)['\"]\s*\)", action_str, re.IGNORECASE)
        if key_match:
            key_name = key_match.group(1)
            return {
                "thought": thought or f"Pressing key '{key_name}'",
                "action": "cua_press_key",
                "params": {"key": key_name},
                "user_update": f"Pressing '{key_name}'...",
                "is_completed": False,
            }

        # 5. Parse finished / complete action: finished(response='...') or stop()
        finish_match = re.search(r"(?:finished|complete|stop)\s*\(\s*(?:response|message)?\s*=\s*['\"]?([^'\"]*)['\"]?\s*\)", action_str, re.IGNORECASE)
        if finish_match or "finished()" in action_str.lower() or "completed" in action_str.lower():
            response_msg = finish_match.group(1) if finish_match and finish_match.group(1) else "Goal completed."
            return {
                "thought": thought or "Task execution complete",
                "action": "converse",
                "params": {"response": response_msg},
                "final_response": response_msg,
                "user_update": "Task completed.",
                "is_completed": True,
            }

        # Fallback: Unrecognized output -> return None to force a retry
        logger.warning(f"UITarsParser: No recognized action pattern in: '{clean_text[:120]}'")
        return None

