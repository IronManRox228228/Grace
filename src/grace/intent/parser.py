"""Intent parser - validates and parses JSON output from Gemma.

Extracts tool name, parameters, and optional response from
the structured JSON output produced by the intent engine.
"""

import json
import logging
from typing import Optional, Any

logger = logging.getLogger("grace.intent")

from grace.intent.tools import ALL_TOOLS

# Derived from tools.py rather than hand-maintained. There were three separate
# copies of this list (here, tools.py, and prompt.py) and they had already
# drifted apart, so a tool could be advertised to the model and then rejected
# by the parser - or vice versa.
VALID_TOOLS = {tool.name for tool in ALL_TOOLS}


class IntentParseError(Exception):
    """Raised when intent JSON cannot be parsed or validated."""
    pass


class Intent:
    """Parsed and validated intent."""

    def __init__(
        self,
        tool: str,
        params: dict[str, Any],
        response: Optional[str] = None,
    ):
        self.tool = tool
        self.params = params
        self.response = response

    @property
    def needs_cua(self) -> bool:
        """Whether this intent requires the CUA bridge."""
        return self.tool.startswith("cua_")

    @property
    def is_conversation(self) -> bool:
        return self.tool == "converse"

    def __repr__(self) -> str:
        return f"Intent(tool={self.tool!r}, params={self.params!r}, response={self.response!r})"


def clean_json_fence(raw_json: str) -> str:
    """Strip markdown code fences and whitespace from raw JSON strings."""
    cleaned = raw_json.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        opened = False
        content_lines = []
        for line in lines:
            stripped = line.strip()
            if not opened:
                if stripped.startswith("```"):
                    opened = True
                continue
            if stripped.startswith("```"):
                break
            content_lines.append(line)
        cleaned = "\n".join(content_lines)
    return cleaned.strip()


class IntentParser:
    """Parses and validates intent JSON from Gemma."""

    def __init__(self):
        self._last_intent: Optional[Intent] = None

    @property
    def last_intent(self) -> Optional[Intent]:
        return self._last_intent

    def parse(self, raw_json: str) -> Intent:
        """Parse and validate a raw JSON string from Gemma.

        Args:
            raw_json: Raw JSON string produced by the LLM

        Returns:
            Validated Intent object

        Raises:
            IntentParseError: If the JSON is invalid or missing required fields
        """
        cleaned = clean_json_fence(raw_json)

        # Try to parse JSON
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise IntentParseError(f"Failed to parse JSON: {e}") from e

        # Must be a dict
        if not isinstance(data, dict):
            raise IntentParseError(f"Expected JSON object, got {type(data).__name__}")

        # Extract tool
        tool = data.get("tool")
        if not tool:
            raise IntentParseError("Missing required field: 'tool'")

        if tool not in VALID_TOOLS:
            raise IntentParseError(
                f"Unknown tool '{tool}'. Valid tools: {sorted(VALID_TOOLS)}"
            )

        # Extract params
        params = data.get("params", {})
        if not isinstance(params, dict):
            raise IntentParseError(f"'params' must be a dict, got {type(params).__name__}")

        # Extract optional response (can be top-level or in params)
        response = data.get("response") or params.get("response")

        intent = Intent(tool=tool, params=params, response=response)
        self._last_intent = intent

        logger.debug(f"Parsed intent: {intent}")
        return intent

    def extract_response_text(self, intent: Intent) -> str:
        """Extract the verbal response text from an intent.

        For converse tools, returns intent.response or params["response"].
        For other tools, returns an empty string (response generated separately).
        """
        if intent.is_conversation and intent.response:
            return intent.response
        if intent.is_conversation and "response" in intent.params:
            return intent.params["response"]
        return ""
