"""Tests verifying no tool description teaches a literal numeric window id (real-desktop-3.md Item 7)."""

import re
from grace.intent.tools import ALL_TOOLS


def test_no_tool_description_contains_literal_numeric_window_id():
    """No tool description or parameter description should contain a literal numeric window id (e.g. 'id': 1234)."""
    pattern = re.compile(r"['\"]id['\"]\s*:\s*\d+")

    for tool in ALL_TOOLS:
        assert not pattern.search(tool.description), (
            f"Tool {tool.name} description contains literal numeric window id: {tool.description}"
        )
        for param in tool.params:
            assert not pattern.search(param.description), (
                f"Tool {tool.name} param '{param.name}' description contains literal numeric window id: {param.description}"
            )
