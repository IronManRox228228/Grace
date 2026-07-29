"""Unit and Integration Tests for Grace Agentic Supermind Components."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from grace.agent.memory import AgentMemory, StepRecord
from grace.agent.perception import PerceptionEngine, ScreenSnapshot, WindowInfo, OcrLine
from grace.agent.safety import SafetyGuard
from grace.intent.router import CapabilityRouter, TaskComplexity
from grace.intent.parser import Intent
from grace.agent.loop import AgentLoop


class TestAgentMemory:
    def test_add_step_and_scratchpad(self):
        mem = AgentMemory(user_goal="Summarize notes in Notepad", max_iterations=5)
        assert mem.user_goal == "Summarize notes in Notepad"
        assert not mem.is_exceeded

        mem.set_scratchpad("extracted_text", "Sample chemistry notes text")
        assert mem.get_scratchpad("extracted_text") == "Sample chemistry notes text"

        step = mem.add_step(
            thought="Open Notepad application",
            action="open_app",
            params={"name": "Notepad"},
            result={"status": "ok"},
            user_update="Opening Notepad...",
        )
        assert step.step_number == 1
        assert len(mem.steps_taken) == 1
        assert "Step 1" in mem.format_history_markdown()
        assert "extracted_text" in mem.format_scratchpad_markdown()


class TestPerceptionEngine:
    def test_screen_snapshot_markdown_formatting(self):
        win_info = WindowInfo(hwnd=1234, title="Untitled - Notepad", class_name="Notepad", rect=(0, 0, 800, 600))
        ocr_lines = [OcrLine(text="Hello World", bounding_box=(10, 10, 100, 20))]
        snapshot = ScreenSnapshot(active_window=win_info, ocr_lines=ocr_lines, width=1920, height=1080)

        md = snapshot.to_markdown()
        assert "Untitled - Notepad" in md
        assert "Hello World" in md
        assert "1920x1080" in md

    def test_capture_snapshot_no_crash(self):
        engine = PerceptionEngine()
        snapshot = engine.capture_snapshot()
        assert isinstance(snapshot, ScreenSnapshot)
        assert snapshot.width > 0
        assert snapshot.height > 0


class TestSafetyGuard:
    def test_safe_actions_pass(self):
        is_safe, prompt = SafetyGuard.evaluate("open_app", {"name": "Notepad"})
        assert is_safe is True
        assert prompt is None

        is_safe, prompt = SafetyGuard.evaluate("cua_type_text", {"text": "hello"})
        assert is_safe is True

    def test_confirmation_required_actions(self):
        is_safe, prompt = SafetyGuard.evaluate("delete_file", {"name": "notes.txt"})
        assert is_safe is False
        assert "delete notes.txt" in prompt.lower()

        is_safe, prompt = SafetyGuard.evaluate("close_app", {"name": "Word"})
        assert is_safe is False
        assert "word" in prompt.lower()


class TestCapabilityRouter:
    def test_fast_path_routing(self):
        intent = Intent(tool="adjust_volume", params={"level": 50})
        route = CapabilityRouter.classify("Turn volume to 50", intent)
        assert route == TaskComplexity.FAST_PATH

    def test_agentic_goal_routing(self):
        route = CapabilityRouter.classify("Summarize my chemistry notes in Notepad", None)
        assert route == TaskComplexity.AGENTIC_GOAL

        intent = Intent(tool="read_pdf", params={"path": "notes.pdf"})
        route = CapabilityRouter.classify("Read notes.pdf", intent)
        assert route == TaskComplexity.AGENTIC_GOAL


class TestAgentLoop:
    def test_loop_completion(self):
        gemma = AsyncMock()
        dispatcher = AsyncMock()
        perception = MagicMock()

        perception.capture_snapshot.return_value = ScreenSnapshot(
            active_window=None, ocr_lines=[], width=1920, height=1080
        )

        gemma.generate_text.side_effect = [
            '```json\n{"thought": "Open Notepad", "action": "open_app", "params": {"name": "Notepad"}, "is_completed": false, "user_update": "Opening Notepad..."}\n```',
            '```json\n{"thought": "Notes summarized", "action": "converse", "params": {}, "is_completed": true, "final_response": "Summary created successfully.", "user_update": "Done"}\n```',
        ]

        dispatcher.execute.return_value = {"status": "ok"}

        loop = AgentLoop(gemma=gemma, dispatcher=dispatcher, perception=perception)
        res = asyncio.run(loop.run(user_goal="Summarize notes in Notepad"))

        assert res["status"] == "ok"
        assert res["final_response"] == "Summary created successfully."
        assert len(res["steps"]) == 2
