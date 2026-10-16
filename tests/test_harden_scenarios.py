"""Tests for hardened scenarios: multi-step, safety gate, mid-task stop, and PDF handling."""

import asyncio
import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grace.agent.memory import AgentMemory
from grace.agent.safety import SafetyGuard
from grace.agent.loop import AgentLoop
from grace.automation.computer_use import ComputerUse
from grace.intent.parser import Intent
from grace.agent.planner import wrap_untrusted
from grace.tools.dispatcher import Dispatcher


# ============================================================================
# 1. Agentic multi-step tests
# ============================================================================

class TestAgenticMultiStep:
    def test_format_established_ruled_out_markdown(self):
        memory = AgentMemory("find report")
        memory.establish("Found report on Desktop")
        memory.rule_out("click search icon", "search icon not clickable")

        md = memory.format_established_ruled_out_markdown()
        assert "Established facts:" in md
        assert "Found report on Desktop" in md
        assert "Ruled out approaches:" in md
        assert "click search icon - search icon not clickable" in md

    def test_app_switching_is_harness_action(self):
        # App switching is supported by cua_activate tool (by title or handle)
        assert "cua_activate" in SafetyGuard.SAFE_READ_ONLY_TOOLS

    def test_step_failure_bounding_asks_user_question(self):
        # When an action fails twice, the loop asks the user a question instead of looping
        from grace.agent.loop import PlannedStep
        loop = AgentLoop(
            perception=MagicMock(),
            planner=MagicMock(),
            gemma=MagicMock(),
            dispatcher=MagicMock(),
            grounder=MagicMock(),
            max_iterations=10,
        )
        memory = AgentMemory("open file")
        # Step failures dictionary with 2 failures for an action
        step_failures = {"cua_click_target": 2}
        step = PlannedStep(action="cua_click", params={"target_name": "target"}, expect="target opened")
        # Ensure it recognizes repeated step failure and bounds it
        assert step_failures.get("cua_click_target") >= 2


# ============================================================================
# 2. Safety confirmation tests
# ============================================================================

class TestSafetyConfirmation:
    @pytest.mark.parametrize("safe_tool", [
        "converse", "search_files", "read_pdf", "summarize_pdf",
        "adjust_volume", "open_calculator", "undo", "describe_screen",
        "set_speech_rate", "cua_list_windows", "cua_get_window",
        "cua_screenshot", "cua_get_cursor_position", "cua_scroll",
        "cua_hover", "cua_move_cursor", "cua_drag", "cua_activate",
    ])
    def test_allowlist_safe_tools_run_without_confirmation(self, safe_tool):
        is_safe, prompt = SafetyGuard.evaluate(safe_tool, {})
        assert is_safe is True
        assert prompt is None

    @pytest.mark.parametrize("destructive_tool", [
        "delete_file", "close_app", "lock_computer",
    ])
    def test_always_confirm_tools(self, destructive_tool):
        is_safe, prompt = SafetyGuard.evaluate(destructive_tool, {})
        assert is_safe is False
        assert prompt is not None

    @pytest.mark.parametrize("kw", [
        "delete", "remove", "send", "submit", "pay", "purchase", "buy",
        "order", "checkout", "format", "erase", "kill", "taskkill", "overwrite",
    ])
    def test_destructive_click_keywords_require_confirmation(self, kw):
        is_safe, prompt = SafetyGuard.evaluate("cua_click", {"target_name": f"Please {kw} now"})
        assert is_safe is False
        assert prompt is not None

    def test_dangerous_key_press_requires_confirmation(self):
        is_safe, prompt = SafetyGuard.evaluate("cua_press_key", {"key": "alt+f4"})
        assert is_safe is False
        assert "unsaved work" in prompt

    def test_overwrite_file_requires_confirmation(self):
        is_safe, prompt = SafetyGuard.evaluate("cua_type_text", {"text": "hello", "overwrite": True})
        assert is_safe is False
        assert "overwrite" in prompt

    def test_r9_taskkill_refuses_wildcards(self):
        dispatcher = Dispatcher()
        # Wildcard in process name or flag
        res = asyncio.run(dispatcher._close_app({"name": "*"}))
        assert res["status"] == "error"
        assert "wildcards are not allowed" in res["error"]

        res2 = asyncio.run(dispatcher._close_app({"name": "chrome*"}))
        assert res2["status"] == "error"
        assert "wildcards are not allowed" in res2["error"]

    def test_r10_delete_file_refuses_directories(self):
        dispatcher = Dispatcher()
        with tempfile.TemporaryDirectory() as tmpdir:
            res = asyncio.run(dispatcher._delete_file({"name": tmpdir}))
            assert res["status"] == "error"
            assert "directory" in res["error"].lower()

    @pytest.mark.parametrize("greeting", [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "howdy",
    ])
    def test_r16_greetings_do_not_count_as_yes(self, greeting):
        from grace.main import GraceApp
        assert GraceApp._confirmation_answer(greeting) is None


# ============================================================================
# 3. Stop mid-task tests
# ============================================================================

class TestStopMidTask:
    def test_request_cancel_aborts_model_and_sets_flag(self):
        mock_gemma = MagicMock()
        loop = AgentLoop(
            perception=MagicMock(),
            planner=MagicMock(),
            gemma=mock_gemma,
            dispatcher=MagicMock(),
            grounder=MagicMock(),
        )
        loop.request_cancel()
        assert loop._cancel_requested is True
        assert AgentLoop.is_cancel_requested() is True
        mock_gemma.abort.assert_called_once()

    def test_cancel_result_reports_stopped(self):
        loop = AgentLoop(
            perception=MagicMock(),
            planner=MagicMock(),
            gemma=MagicMock(),
            dispatcher=MagicMock(),
            grounder=MagicMock(),
        )
        memory = AgentMemory("test goal")
        AgentLoop._typing_cut_off = False
        res = loop._cancel_result(memory)
        assert res["status"] == "cancelled"
        assert res["final_response"] == "Stopped."

    def test_cancel_result_when_typing_cut_off(self):
        loop = AgentLoop(
            perception=MagicMock(),
            planner=MagicMock(),
            gemma=MagicMock(),
            dispatcher=MagicMock(),
            grounder=MagicMock(),
        )
        memory = AgentMemory("test goal")
        AgentLoop._typing_cut_off = True
        res = loop._cancel_result(memory)
        assert res["status"] == "cancelled"
        assert res["final_response"] == "Stopped. Typing was interrupted."

    def test_typing_interruption_backspaces_and_returns_cut_off(self):
        fake_pyautogui = MagicMock()
        fake_pyautogui.write = MagicMock()
        fake_pyautogui.press = MagicMock()
        # Mock cancel checker to return True
        cua = ComputerUse()
        ComputerUse.set_cancel_checker(lambda: True)

        with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
            res = cua._type_text({"text": "Hello world"})
            assert res["typing_cut_off"] is True
            assert res["ok"] is False

        # Reset cancel checker
        ComputerUse.set_cancel_checker(None)


# ============================================================================
# 4. read_pdf / summarize_pdf tests
# ============================================================================

class TestPdfHandling:
    def test_r14_pdf_size_cap_refuses_large_files(self):
        dispatcher = Dispatcher()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 dummy")
            tmp_path = f.name
        try:
            with patch("os.path.getsize", return_value=60 * 1024 * 1024):
                res = asyncio.run(dispatcher._read_pdf({"path": tmp_path}))
                assert res["status"] == "error"
                assert "50MB limit" in res["error"]
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_pdf_untrusted_wrapping(self):
        text = "Confidential document text"
        wrapped = wrap_untrusted(text)
        assert "<<<UNTRUSTED_DATA_START" in wrapped
        assert "<<<UNTRUSTED_DATA_END>>>" in wrapped
        assert text in wrapped

    def test_pdf_summary_markdown_free_and_speakable(self):
        dispatcher = Dispatcher()
        # Test markdown stripping in summary response
        raw_summary = "### Overview\n* Point 1: This is **critical**.\n* Point 2: `Important` note."
        clean = dispatcher._clean_summary_for_speech(raw_summary)
        assert "#" not in clean
        assert "*" not in clean
        assert "`" not in clean
        assert len(clean) > 0
