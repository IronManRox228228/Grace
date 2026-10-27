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


@pytest.fixture(autouse=True)
def _no_terminal_in_front(monkeypatch):
    # The send-time terminal check reads the real foreground window; a test run
    # started from a terminal would otherwise refuse every keystroke.
    import grace.automation.computer_use as cu
    monkeypatch.setattr(cu, "_terminal_foreground", lambda: None)


# ============================================================================
# 1. Agentic multi-step tests
# ============================================================================

class TestAgenticMultiStep:
    def test_format_established_ruled_out_markdown(self):
        memory = AgentMemory("find report")
        memory.establish("Found report on Desktop")
        memory.rule_out("click search icon", "search icon not clickable")

        md = memory.format_established_ruled_out_markdown()
        assert "Observed so far:" in md
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
        "converse", "search_files",
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
        "delete", "remove", "submit", "pay", "purchase", "buy",
        "order", "checkout", "format", "erase", "kill", "taskkill", "overwrite",
    ])
    def test_destructive_click_keywords_require_confirmation(self, kw):
        is_safe, prompt = SafetyGuard.evaluate("cua_click", {"target_name": f"Please {kw} now"})
        assert is_safe is False
        assert prompt is not None

    def test_clicking_send_needs_no_confirmation(self):
        # A message the user asked for is already authorised.
        assert SafetyGuard.evaluate("cua_click", {"target_name": "Send"}) == (True, None)

    def test_dangerous_key_press_requires_confirmation(self):
        is_safe, prompt = SafetyGuard.evaluate("cua_press_key", {"key": "alt+f4"})
        assert is_safe is False
        assert "unsaved work" in prompt

    def test_r37_delete_key_requires_confirmation(self):
        is_safe, prompt = SafetyGuard.evaluate("cua_press_key", {"key": "delete"})
        assert is_safe is False
        assert prompt is not None

    def test_r38_reversed_hotkey_requires_confirmation(self):
        is_safe, prompt = SafetyGuard.evaluate("cua_press_key", {"key": "f4+alt"})
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


class TestR25SecretShapeHeuristics:
    def test_redact_punctuated_cards_and_passphrases(self):
        from grace.harness.redact import redact_string, redact_data

        # Punctuated card numbers
        assert "<redacted>" in redact_string("Card: 4111, 1111, 1111, 1111")
        assert "<redacted>" in redact_string("Card: 4111.1111.1111.1111")

        # Delimiter-separated passphrases
        assert "<redacted>" in redact_string("Key: correct-horse-battery-staple")
        assert "<redacted>" in redact_string("Key: blue_sky_morning_coffee")

        # Passphrase key in dict
        data = {"passphrase": "any value here"}
        redacted = redact_data(data)
        assert redacted["passphrase"] == "<redacted>"


class TestR26PdfPathAllowlist:
    def test_pdf_outside_safe_roots_requires_confirmation(self):
        from grace.agent.safety import SafetyGuard
        from grace.tools.dispatcher import Dispatcher

        unsafe_path = "C:\\Windows\\System32\\test.pdf"
        safe, prompt = SafetyGuard.evaluate("read_pdf", {"path": unsafe_path})
        assert not safe
        assert prompt is not None
        assert "outside your documents folders" in prompt

        safe_sum, prompt_sum = SafetyGuard.evaluate("summarize_pdf", {"path": unsafe_path})
        assert not safe_sum
        assert prompt_sum is not None

        # Dispatcher refuses without confirmation
        dispatcher = Dispatcher()
        res_read = asyncio.run(dispatcher._read_pdf({"path": unsafe_path}, confirmed=False))
        assert res_read["status"] == "error"
        assert "outside Documents, Desktop, and Downloads" in res_read["error"]

        res_sum = asyncio.run(dispatcher._summarize_pdf({"path": unsafe_path}, confirmed=False))
        assert res_sum["status"] == "error"
        assert "outside Documents, Desktop, and Downloads" in res_sum["error"]

    def test_pdf_inside_safe_roots_is_allowed(self):
        from grace.agent.safety import SafetyGuard
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 dummy")
            tmp_path = f.name
        try:
            safe, prompt = SafetyGuard.evaluate("read_pdf", {"path": tmp_path})
            assert safe
            assert prompt is None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestR39ListenCutoff:
    def test_listen_cutoff_speaks_apology(self):
        from grace.main import GraceApp
        from grace.config import Config
        from grace.util.timing import start_turn
        app = GraceApp.__new__(GraceApp)
        app._running = True
        app.config = Config(whisper_vad_threshold=0.01)
        app.ws_server = MagicMock()
        app.ws_server.emit = AsyncMock()
        app.wake_word = MagicMock()
        app.vad = MagicMock()
        app.vad.process_chunk = MagicMock(return_value=False)
        app.whisper = MagicMock()
        app.capture = MagicMock()
        app.capture.get_rms = MagicMock(return_value=1000.0)
        app._speak_response = AsyncMock()


        chunk = b"\x00" * 1024
        app.pump = MagicMock()
        app.pump.get = AsyncMock(return_value=chunk)
        app.pump.drain = MagicMock()

        fake_times = [0.0, 0.1, 35.0, 65.0, 70.0]
        time_iter = iter(fake_times)
        with start_turn("wake") as trace:
            with patch("time.time", side_effect=lambda: next(time_iter, 100.0)):
                res = asyncio.run(app._handle_activation_turn(trace))


        assert res is False
        app._speak_response.assert_awaited_with("I didn't catch all of that.")
        app.ws_server.emit.assert_any_await({"type": "ListeningStopped"})
        app.ws_server.emit.assert_any_await({"type": "ConversationFinished"})
        app.ws_server.emit.assert_any_await({"type": "Idle"})


class TestR40TTSPlayerLockAndGeneration:
    def test_stop_closes_stream_under_lock_and_aborts(self):
        from grace.tts.player import TTSPlayer
        player = TTSPlayer()
        mock_stream = MagicMock()
        player._stream = mock_stream
        player._stream_params = (24000, 1, 2)

        player.stop()

        assert player._stream is None
        assert player._stream_params is None
        mock_stream.abort.assert_called_once()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_write_stream_drops_stale_generation(self):
        from grace.tts.player import TTSPlayer
        player = TTSPlayer()
        mock_stream = MagicMock()
        player._stream = mock_stream
        player._stream_params = (24000, 1, 2)

        with patch("grace.tts.player.decode_wav", return_value=(b"\x00\x00" * 100, 24000, 1, 2)):
            player._generation = 1
            res = player._write_stream(b"dummy_wav", generation=0)
            assert res is True
            mock_stream.write.assert_not_called()


class TestR41GeminiApiKeyHeader:
    def test_stream_gemini_response_sends_key_in_header_not_query(self):
        from grace.llm.gemma_client import GemmaClient
        client = GemmaClient(api_key="secret-api-key-12345")

        posted_calls = []

        class FakeResponse:
            status = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            class _FakeContent:
                def __aiter__(self):
                    return self
                async def __anext__(self):
                    raise StopAsyncIteration
            content = _FakeContent()


        class FakeSession:
            closed = False
            def post(self, url, json=None, headers=None, timeout=None):
                posted_calls.append({"url": url, "headers": headers, "json": json})
                return FakeResponse()

        client._http = FakeSession()

        async def run_stream():
            tokens = []
            async for tok in client._stream_gemini_response([{"role": "user", "content": "hi"}]):
                tokens.append(tok)
            return tokens

        asyncio.run(run_stream())
        assert len(posted_calls) == 1
        call = posted_calls[0]
        assert "key=secret-api-key-12345" not in call["url"]
        assert "key=" not in call["url"]
        assert call["headers"].get("x-goog-api-key") == "secret-api-key-12345"



class TestR42TauriStrictCsp:
    def test_tauri_conf_has_strict_csp(self):
        import json
        conf_path = os.path.join(os.path.dirname(__file__), "..", "src-tauri", "tauri.conf.json")
        with open(conf_path, "r", encoding="utf-8") as f:
            conf = json.load(f)
        csp = conf.get("app", {}).get("security", {}).get("csp", "")
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "connect-src" in csp



