"""Read-only tools the planner is offered must never park a confirmation."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.agent.safety import SafetyGuard


@pytest.mark.parametrize("tool", ["cua_text", "cua_list_apps", "cua_list_windows", "cua_screenshot", "describe_screen"])
def test_read_only_tools_are_safe(tool):
    assert SafetyGuard.evaluate(tool, {}) == (True, None)


def test_unknown_tool_still_needs_confirmation():
    safe, prompt = SafetyGuard.evaluate("cua_frobnicate", {})
    assert safe is False and prompt
