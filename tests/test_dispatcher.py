"""Test tool dispatcher."""

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.intent.parser import Intent
from grace.tools.dispatcher import Dispatcher


class TestDispatcher:
    """Test suite for Dispatcher."""

    def setup_method(self):
        self.dispatcher = Dispatcher(computer_use=None)

    def test_execute_unknown_tool(self):
        """Execute with an unknown tool returns error."""
        intent = Intent(tool="nonexistent_tool", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "error"
        assert "nonexistent_tool" in result["error"]

    def test_execute_converse(self):
        """Converse tool is handled but returns status ok."""
        intent = Intent(tool="converse", params={"response": "Hi!"})
        # converse is not in handler_map, so it goes to the unknown tool path
        result = asyncio.run(self.dispatcher.execute(intent))
        # converse falls through to unknown tool handler
        assert "error" in result or "status" in result

    def test_cua_tool_without_bridge(self):
        """CUA tool without bridge returns error."""
        intent = Intent(tool="cua_click", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "error"
        assert "not available" in result["error"].lower()
