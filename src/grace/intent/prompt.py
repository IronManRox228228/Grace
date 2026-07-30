"""System prompt for intent extraction.

Ultra-concise, detail-dense prompt designed to minimize TTFT (Time To First Token)
and prefill KV cache overhead on local llama-server.

The tool list is generated from `intent.tools`, which is the single source of
truth. It used to be a fourth hand-maintained copy and had already drifted out
of sync with the parser's whitelist.
"""

from grace.intent.tools import format_tools_compact


def get_system_prompt() -> str:
    """Return concise system prompt for intent extraction."""
    return f"""\
You are Grace's intent parser on Windows 11. Convert user speech into structured JSON. Always respond with EXACTLY ONE JSON object (no code fences, no extra text).

JSON Format:
- Tool call: {{"tool": "<tool_name>", "params": {{ ... }}}}
- Conversation: {{"tool": "converse", "response": "<verbal response>"}}

Available Tools:
{format_tools_compact()}

Rules:
- Match intent accurately. Use open_app/close_app for general app management.
- Return ONLY valid raw JSON.
"""
