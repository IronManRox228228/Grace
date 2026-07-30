"""UI-TARS Action Parser for Grace Desktop Automation.

Parses native UI-TARS 7B model text output (Thought / Action format)
into structured Grace step intents (action, params, is_completed, thought).

The model's own coordinate syntax uses box tokens:

    click(start_box='<|box_start|>(345,678)<|box_end|>')

which the previous regex did not match, so every native-format click fell
through to "unrecognized" and burned a retry turn. Actions the model emits
often - scroll, drag, wait, positional hotkey - were unparsed for the same
reason, and left_double_click / right_click both silently collapsed into a
plain left click.
"""

import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("grace.agent.ui_tars_parser")

# A coordinate pair, with or without the box tokens and quoting the model uses.
_BOX = r"(?:<\|box_start\|>)?\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?\s*(?:<\|box_end\|>)?"
_ARG = r"['\"]?"


def _coord_pattern(keywords: str, arg_names: str = "start_box|point|location|coordinate") -> str:
    return (
        rf"(?:{keywords})\s*\(\s*(?:(?:{arg_names})\s*=\s*)?{_ARG}{_BOX}{_ARG}\s*\)"
    )


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

        for handler in (
            cls._parse_finished,
            cls._parse_drag,
            cls._parse_scroll,
            cls._parse_coordinate_click,
            cls._parse_named_click,
            cls._parse_type,
            cls._parse_hotkey,
            cls._parse_wait,
        ):
            result = handler(thought, action_str)
            if result is not None:
                return result

        # Fallback: Unrecognized output -> return None to force a retry
        logger.warning(f"UITarsParser: No recognized action pattern in: '{clean_text[:120]}'")
        return None

    # -- individual actions ------------------------------------------------

    @classmethod
    def _parse_coordinate_click(cls, thought: str, action_str: str) -> Optional[Dict[str, Any]]:
        """click / left_double_click / right_click / middle_click at a point.

        The click *kind* is preserved. It used to be discarded, so asking for a
        right click opened nothing and asking for a double click opened nothing
        twice.
        """
        match = re.search(
            _coord_pattern(r"left_double_click|double_click|left_click|right_click|middle_click|click|hover|mouse_move"),
            action_str,
            re.IGNORECASE,
        )
        if not match:
            return None

        x, y = int(match.group(1)), int(match.group(2))
        verb = match.group(0).split("(")[0].strip().lower()

        params: Dict[str, Any] = {"x": x, "y": y}
        if verb in ("left_double_click", "double_click"):
            params["click_count"] = 2
            label = "Double-clicking"
        elif verb == "right_click":
            params["button"] = "right"
            label = "Right-clicking"
            return {
                "thought": thought or f"Right-clicking at ({x}, {y})",
                "action": "cua_secondary_action",
                "params": params,
                "user_update": f"Right-clicking at ({x}, {y})...",
                "is_completed": False,
            }
        elif verb in ("hover", "mouse_move"):
            params["click_count"] = 1
            label = "Moving to"
        else:
            params["click_count"] = 1
            label = "Clicking"

        return {
            "thought": thought or f"{label} target at ({x}, {y})",
            "action": "cua_click",
            "params": params,
            "user_update": f"{label} at ({x}, {y})...",
            "is_completed": False,
        }

    @classmethod
    def _parse_named_click(cls, thought: str, action_str: str) -> Optional[Dict[str, Any]]:
        """click(target='...') - a named rather than located target."""
        match = re.search(
            r"(?:left_double_click|double_click|right_click|click)\s*\(\s*(?:target|name|label|text|element)\s*=\s*['\"]([^'\"]+)['\"]\s*\)",
            action_str,
            re.IGNORECASE,
        )
        if not match:
            return None

        target_name = match.group(1)
        verb = match.group(0).split("(")[0].strip().lower()
        params: Dict[str, Any] = {"target_name": target_name}
        if verb in ("left_double_click", "double_click"):
            params["click_count"] = 2
        if verb == "right_click":
            return {
                "thought": thought or f"Right-clicking '{target_name}'",
                "action": "cua_secondary_action",
                "params": params,
                "user_update": f"Right-clicking '{target_name}'...",
                "is_completed": False,
            }

        return {
            "thought": thought or f"Clicking '{target_name}'",
            "action": "cua_click",
            "params": params,
            "user_update": f"Clicking '{target_name}'...",
            "is_completed": False,
        }

    @classmethod
    def _parse_type(cls, thought: str, action_str: str) -> Optional[Dict[str, Any]]:
        """type(content='...') - `content` is UI-TARS's own argument name."""
        match = re.search(
            r"type\s*\(\s*(?:content|text|input|value)?\s*=?\s*['\"]([^'\"]*)['\"]\s*\)",
            action_str,
            re.IGNORECASE,
        )
        if not match:
            return None

        text = match.group(1)
        return {
            "thought": thought or f"Typing text '{text}'",
            "action": "cua_type_text",
            "params": {"text": text},
            "user_update": f"Typing '{text[:40]}'...",
            "is_completed": False,
        }

    @classmethod
    def _parse_hotkey(cls, thought: str, action_str: str) -> Optional[Dict[str, Any]]:
        """hotkey('ctrl a') and press_key(key='Return').

        UI-TARS emits hotkeys positionally and space-separated; only the
        keyword form was previously handled.
        """
        match = re.search(
            r"(?:hotkey|press_key|press|key)\s*\(\s*(?:(?:key|name|content)\s*=\s*)?['\"]([^'\"]+)['\"]\s*\)",
            action_str,
            re.IGNORECASE,
        )
        if not match:
            return None

        key_name = match.group(1).strip()
        # "ctrl a" -> "ctrl+a"; the executor splits on '+'.
        if " " in key_name and "+" not in key_name:
            key_name = "+".join(part for part in key_name.split() if part)

        return {
            "thought": thought or f"Pressing key '{key_name}'",
            "action": "cua_press_key",
            "params": {"key": key_name},
            "user_update": f"Pressing '{key_name}'...",
            "is_completed": False,
        }

    @classmethod
    def _parse_scroll(cls, thought: str, action_str: str) -> Optional[Dict[str, Any]]:
        """scroll(start_box='(x,y)', direction='down')."""
        if not re.search(r"\bscroll\s*\(", action_str, re.IGNORECASE):
            return None

        point = re.search(_BOX, action_str)
        x = int(point.group(1)) if point else 0
        y = int(point.group(2)) if point else 0

        direction_match = re.search(
            r"direction\s*=\s*['\"]?(up|down|left|right)['\"]?", action_str, re.IGNORECASE
        )
        direction = (direction_match.group(1) if direction_match else "down").lower()

        amounts = {"down": (0, 500), "up": (0, -500), "right": (500, 0), "left": (-500, 0)}
        scroll_x, scroll_y = amounts[direction]

        return {
            "thought": thought or f"Scrolling {direction}",
            "action": "cua_scroll",
            "params": {"x": x, "y": y, "scrollX": scroll_x, "scrollY": scroll_y},
            "user_update": f"Scrolling {direction}...",
            "is_completed": False,
        }

    @classmethod
    def _parse_drag(cls, thought: str, action_str: str) -> Optional[Dict[str, Any]]:
        """drag(start_box='(x1,y1)', end_box='(x2,y2)')."""
        if not re.search(r"\b(?:drag|select)\s*\(", action_str, re.IGNORECASE):
            return None

        points = re.findall(_BOX, action_str)
        if len(points) < 2:
            return None

        (x1, y1), (x2, y2) = points[0], points[1]
        return {
            "thought": thought or "Dragging between two points",
            "action": "cua_drag",
            "params": {
                "from_x": int(x1), "from_y": int(y1),
                "to_x": int(x2), "to_y": int(y2),
            },
            "user_update": "Dragging...",
            "is_completed": False,
        }

    @classmethod
    def _parse_wait(cls, thought: str, action_str: str) -> Optional[Dict[str, Any]]:
        """wait() - the model asking for the screen to settle.

        Mapped to a no-op observation rather than an interaction, so the loop
        takes a fresh snapshot without touching anything.
        """
        if not re.search(r"\b(?:wait|sleep)\s*\(", action_str, re.IGNORECASE):
            return None

        return {
            "thought": thought or "Waiting for the screen to update",
            "action": "cua_screenshot",
            "params": {},
            "user_update": "Waiting...",
            "is_completed": False,
        }

    @classmethod
    def _parse_finished(cls, thought: str, action_str: str) -> Optional[Dict[str, Any]]:
        """finished(content='...') / stop()."""
        match = re.search(
            r"(?:finished|complete|stop)\s*\(\s*(?:(?:response|message|content)\s*=\s*)?['\"]?([^'\"]*)['\"]?\s*\)",
            action_str,
            re.IGNORECASE,
        )
        if not match and "finished()" not in action_str.lower() and "completed" not in action_str.lower():
            return None

        response_msg = (match.group(1).strip() if match and match.group(1) else "") or "Goal completed."
        return {
            "thought": thought or "Task execution complete",
            "action": "converse",
            "params": {"response": response_msg},
            "final_response": response_msg,
            "user_update": "Task completed.",
            "is_completed": True,
        }
