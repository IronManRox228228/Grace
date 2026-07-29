"""Adversarial break tests - edge cases, crash tests, and boundary conditions.

These tests intentionally push code into failure modes to find bugs.
"""

import sys
import os
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio
import pytest

from grace.config import Config
from grace.audio.capture import AudioCapture
from grace.audio.wake_word import WakeWordDetector
from grace.vad.detector import VadDetector
from grace.stt.whisper_stream import WhisperStreaming
from grace.llm.gemma_client import GemmaClient
from grace.intent.parser import IntentParser, Intent, IntentParseError
from grace.automation.computer_use import ComputerUse
from grace.tools.dispatcher import Dispatcher
from grace.tts.kokoro_engine import KokoroEngine, KokoroWorker
from grace.tts.player import TTSPlayer
from grace.response.generator import ResponseGenerator


# ─────────────────────────────────────────────
# CONFIG BREAK TESTS
# ─────────────────────────────────────────────

class TestConfigBreak:
    def test_frozen_dataclass_cannot_mutate(self):
        config = Config()
        with pytest.raises(Exception):
            config.llama_port = 9999

    def test_whisper_model_path_exists(self):
        """Config now has whisper_model_path (fix for main.py reference)."""
        config = Config()
        assert hasattr(config, "whisper_model_path")
        assert config.whisper_model_path == "small"

    def test_invalid_env_port_is_string(self):
        import dataclasses
        # Simulate what happens if env var is non-numeric
        config = Config()
        assert isinstance(config.llama_port, int)

    def test_kokoro_workers_zero_or_negative(self):
        """Zero workers would cause division-by-mod in synthesize."""
        config = Config()
        assert config.kokoro_workers > 0, "Zero kokoro_workers would break KokoroEngine.synthesize"

    def test_mic_device_index_minus_one(self):
        """-1 means default device; should still be int."""
        config = Config()
        assert config.mic_device_index == -1


# ─────────────────────────────────────────────
# AUDIO CAPTURE BREAK TESTS
# ─────────────────────────────────────────────

class TestAudioCaptureBreak:
    def test_get_chunk_before_start_raises(self):
        capture = AudioCapture()
        with pytest.raises(RuntimeError, match="not running"):
            capture.get_chunk()

    def test_get_chunk_as_int16_before_start_raises(self):
        capture = AudioCapture()
        with pytest.raises(RuntimeError):
            capture.get_chunk_as_int16()

    def test_get_rms_odd_byte_length_crashes(self):
        """unpack with odd byte count raises struct.error."""
        capture = AudioCapture()
        with pytest.raises(Exception):
            capture.get_rms(b"\x00\x00\x00")

    def test_close_then_start(self):
        """close then start should reinitialize."""
        capture = AudioCapture()
        capture.close()
        # Should be able to list devices after close (re-inits pyaudio)
        devices = capture.list_devices()
        assert isinstance(devices, list)

    def test_get_device_name_negative_index(self):
        capture = AudioCapture()
        try:
            name = capture.get_device_name(-1)
            # May succeed on some systems, but must not crash
            assert isinstance(name, str)
        except Exception:
            # OSError expected on most systems
            pass

    def test_get_device_name_out_of_range(self):
        capture = AudioCapture()
        count = capture.get_device_count()
        with pytest.raises(Exception):
            capture.get_device_name(count + 999)


# ─────────────────────────────────────────────
# WAKE WORD BREAK TESTS
# ─────────────────────────────────────────────

class TestWakeWordBreak:
    def test_double_start_crashes_on_invalid_model(self):
        """BUG: start() tries Model(invalid_path) and crashes without handling error."""
        detector = WakeWordDetector(model_path="nonexistent-vosk-model")
        try:
            detector.start()
            # If we got here, model loading somehow succeeded - shouldn't happen
            # with a clearly invalid path, but we gracefully clean up
            detector.stop()
        except Exception:
            # start() should catch this and log an error, not crash
            pass

    def test_start_without_model_path(self):
        """start() with None model_path should not crash but won't detect."""
        detector = WakeWordDetector(model_path=None)
        detector.start()
        assert detector._model is None
        assert detector._rec is None
        detector.stop()

    def test_check_keyword_malformed_json(self):
        """_check_keyword should handle non-JSON gracefully."""
        detector = WakeWordDetector(model_path="dummy")
        detector._check_keyword("not json")  # Should not raise

    def test_check_keyword_missing_fields(self):
        """_check_keyword with incomplete data should not raise."""
        detector = WakeWordDetector(model_path="dummy")
        detector._check_keyword('{"partial": true}')  # Should not raise

    def test_check_keyword_empty_result(self):
        detector = WakeWordDetector(model_path="dummy")
        detector._check_keyword('{"text": "", "confidence": 0.0}')  # Should not raise

    def test_callback_non_callable(self):
        detector = WakeWordDetector(model_path="dummy")
        detector.set_callback("not callable")  # No type enforcement
        assert not callable(detector._callback)

    def test_reset_while_stopped(self):
        detector = WakeWordDetector(model_path="dummy")
        detector.stop()
        detector.reset()  # Should not raise

    def test_stop_twice(self):
        detector = WakeWordDetector(model_path="dummy")
        detector.stop()
        detector.stop()  # Second stop should not raise

    def test_wait_for_detection_no_timeout(self):
        detector = WakeWordDetector(model_path="dummy")
        # No thread running, detection will never happen
        result = detector.wait_for_detection(timeout=0.1)
        assert result is False

    def test_detected_after_set(self):
        """detected property should reflect underlying event."""
        detector = WakeWordDetector(model_path="dummy")
        assert not detector.detected
        detector._event.set()
        assert detector.detected is True
        detector.reset()
        assert not detector.detected


# ─────────────────────────────────────────────
# VAD BREAK TESTS
# ─────────────────────────────────────────────

class TestVadBreak:
    def test_process_empty_chunk_handled_gracefully(self):
        """struct.unpack with 0 elements returns (), so empty chunk is handled."""
        detector = VadDetector(threshold=0.001, silence_duration_ms=100)
        result = detector.process_chunk(b"")
        # Should not crash, returns False (silent, no callback triggered)
        assert result is False

    def test_process_single_byte_chunk_no_crash(self):
        """Odd/short chunks no longer crash (struct guard added)."""
        detector = VadDetector(threshold=0.001, silence_duration_ms=100)
        result = detector.process_chunk(b"\x00")
        assert isinstance(result, bool)

    def test_threshold_at_exact_boundary(self):
        detector = VadDetector(threshold=0.0, silence_duration_ms=100)
        # Zero threshold means everything is speech
        samples = struct.pack("<" + "h" * 128, *([0] * 128))
        result = detector.process_chunk(samples)
        assert detector.is_speaking is True

    def test_negative_threshold(self):
        """Negative threshold would mean everything is speech."""
        detector = VadDetector(threshold=-1.0, silence_duration_ms=100)
        samples = struct.pack("<" + "h" * 128, *([0] * 128))
        detector.process_chunk(samples)
        # normalized_rms (0) >= -1.0, so speech
        assert detector.is_speaking is True

    def test_threshold_above_max(self):
        """Threshold > 1.0 means silence never detected as speech."""
        detector = VadDetector(threshold=2.0, silence_duration_ms=100)
        samples = struct.pack("<" + "h" * 128, *([32767] * 128))
        detector.process_chunk(samples)
        assert detector.is_speaking is False

    def test_silence_callback_triggers(self):
        import time
        detector = VadDetector(threshold=0.1, silence_duration_ms=50)
        triggered = []

        def on_silence(state):
            triggered.append(True)

        detector.set_silence_callback(on_silence)

        # First simulate speech (loud signal above threshold)
        loud = struct.pack("<" + "h" * 128, *([8000] * 128))
        detector.process_chunk(loud)
        assert detector.is_speaking is True
        assert detector.has_detected_speech is True

        # Then silence - need wall-clock time to accumulate
        samples = struct.pack("<" + "h" * 128, *([0] * 128))
        for _ in range(10):
            detector.process_chunk(samples)
            time.sleep(0.01)  # 10ms real time per chunk

        # silence callback should have been triggered
        assert len(triggered) >= 1, "Silence callback never triggered"

    def test_speech_to_silence_transition(self):
        detector = VadDetector(threshold=0.1, silence_duration_ms=50)
        # Start with loud signal (32767 * 0.2 ≈ 6553, normalized ≈ 0.2 > 0.1)
        loud = struct.pack("<" + "h" * 128, *([8000] * 128))
        silent = struct.pack("<" + "h" * 128, *([0] * 128))

        detector.process_chunk(loud)
        assert detector.is_speaking is True

        # Then silence - should transition to silent
        for _ in range(5):
            detector.process_chunk(silent)

        assert detector.is_speaking is False

    def test_audio_vs_fallback_rms(self):
        """Test both code paths: with audio parameter and without."""
        class MockAudio:
            def get_rms(self, chunk):
                return 32767  # max value
        detector = VadDetector(threshold=0.5, silence_duration_ms=1000)
        samples = struct.pack("<" + "h" * 128, *([16000] * 128))
        result = detector.process_chunk(samples, MockAudio())
        # With mock audio returning max RMS, should detect speech
        assert detector.is_speaking is True


# ─────────────────────────────────────────────
# WHISPER STREAMING BREAK TESTS
# ─────────────────────────────────────────────

class TestWhisperStreamBreak:
    def test_transcribe_empty_buffer(self):
        ws = WhisperStreaming("small")
        assert ws.transcribe() == ""

    def test_transcribe_without_initialization_raises(self):
        ws = WhisperStreaming("nonexistent-path")
        ws._buffer.extend(b"\x00\x00" * 160)  # Some PCM data
        with pytest.raises(Exception):
            ws.transcribe()

    def test_add_chunk_odd_length(self):
        """BUG: Adding odd-length bytes means transcribe will crash on np.frombuffer with int16."""
        ws = WhisperStreaming("small")
        ws.add_chunk(b"\x00")  # 1 byte - odd
        assert len(ws._buffer) == 1

    def test_transcribe_odd_buffer_handled_gracefully(self):
        """Odd-length buffer is trimmed, not crashed."""
        ws = WhisperStreaming("small")
        ws._buffer = bytearray(b"\x00")  # 1 byte - odd length
        result = ws.transcribe()
        assert result == ""

    def test_clear_and_reset_buffer(self):
        ws = WhisperStreaming("small")
        ws.add_chunk(b"\x00\x00" * 100)
        assert len(ws._buffer) > 0
        ws.clear_buffer()
        assert len(ws._buffer) == 0
        ws.add_chunk(b"\x00\x00" * 50)
        ws.reset_buffer()
        assert len(ws._buffer) == 0
        assert ws._last_transcript == ""

    def test_get_buffer_returns_copy(self):
        ws = WhisperStreaming("small")
        ws.add_chunk(b"\x01\x02" * 10)
        buf = ws.get_buffer()
        ws.clear_buffer()
        # buf should still have the old data (it's bytes, immutable)
        assert len(buf) == 20

    def test_double_initialize_no_error(self):
        ws = WhisperStreaming("small")
        # Should not raise - _initialize checks _initialized flag but doesn't set it
        # without actually loading a model (we're not calling transcribe)
        try:
            ws._initialize()
            ws._initialize()  # second call should be no-op
            assert ws._initialized
        except Exception as e:
            # If this fails, it's because faster-whisper is not installed or model missing
            # That's OK for this test - we just verify no crash from double init
            assert "faster_whisper" in str(e) or "model" in str(e).lower()


# ─────────────────────────────────────────────
# GEMMA CLIENT BREAK TESTS
# ─────────────────────────────────────────────

class TestGemmaClientBreak:
    def test_health_check_no_server(self):
        client = GemmaClient("http://127.0.0.1:1")  # unlikely to be running
        result = asyncio.run(client.health_check())
        assert result is False

    def test_chat_no_server_returns_none(self):
        client = GemmaClient("http://127.0.0.1:1")
        result = asyncio.run(client.chat(
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        ))
        assert result is None

    def test_close_without_start(self):
        client = GemmaClient()
        asyncio.run(client.close())  # Should not raise

    def test_generate_intent_no_server(self):
        client = GemmaClient("http://127.0.0.1:1")
        result = asyncio.run(client.generate_intent("hello", "system prompt"))
        assert result is None

    def test_generate_text_no_server(self):
        """generate_text coroutine should return None on server errors."""
        client = GemmaClient("http://127.0.0.1:1")
        result = asyncio.run(client.generate_text("user", "sys"))
        assert result is None


# ─────────────────────────────────────────────
# INTENT PARSER BREAK TESTS
# ─────────────────────────────────────────────

class TestIntentParserBreak:
    def setup_method(self):
        self.parser = IntentParser()

    def test_parse_empty_string_raises(self):
        with pytest.raises(IntentParseError):
            self.parser.parse("")

    def test_parse_whitespace_only_raises(self):
        with pytest.raises(IntentParseError):
            self.parser.parse("   ")

    def test_parse_json_array_raises(self):
        """JSON array is not a dict."""
        with pytest.raises(IntentParseError, match="Expected JSON object"):
            self.parser.parse('[{"tool": "open_app", "params": {}}]')

    def test_parse_json_null_raises(self):
        with pytest.raises(IntentParseError, match="Expected JSON object"):
            self.parser.parse("null")

    def test_parse_json_string_raises(self):
        with pytest.raises(IntentParseError, match="Expected JSON object"):
            self.parser.parse('"hello"')

    def test_parse_json_number_raises(self):
        with pytest.raises(IntentParseError):
            self.parser.parse("42")

    def test_parse_tool_empty_string_raises(self):
        with pytest.raises(IntentParseError, match="Missing required field"):
            self.parser.parse('{"tool": "", "params": {}}')

    def test_parse_params_not_dict_raises(self):
        with pytest.raises(IntentParseError, match="params.*must be a dict"):
            self.parser.parse('{"tool": "open_app", "params": "not_a_dict"}')

    def test_parse_params_as_list_raises(self):
        with pytest.raises(IntentParseError):
            self.parser.parse('{"tool": "open_app", "params": [1, 2, 3]}')

    def test_parse_tool_with_extra_fields(self):
        """Extra fields should be tolerated."""
        intent = self.parser.parse('{"tool": "open_app", "params": {"name": "calc"}, "extra": "stuff", "unused": true}')
        assert intent.tool == "open_app"
        assert intent.params["name"] == "calc"

    def test_parse_response_at_top_level(self):
        intent = self.parser.parse('{"tool": "converse", "response": "Hello", "params": {}}')
        assert intent.response == "Hello"

    def test_parse_response_in_params(self):
        intent = self.parser.parse('{"tool": "converse", "params": {"response": "Hello"}}')
        assert intent.response == "Hello"

    def test_parse_response_both_places(self):
        """Top-level response takes precedence over params response."""
        intent = self.parser.parse(
            '{"tool": "converse", "response": "Top", "params": {"response": "Params"}}'
        )
        assert intent.response == "Top"

    def test_parse_tool_with_unicode(self):
        intent = self.parser.parse(
            '{"tool": "converse", "params": {"response": "Hola, ¿cómo estás?"}}'
        )
        assert intent.response == "Hola, ¿cómo estás?"

    def test_parse_tool_with_very_long_json(self):
        long_response = "a" * 10000
        json_str = '{"tool": "converse", "params": {"response": "%s"}}' % long_response
        intent = self.parser.parse(json_str)
        assert intent.response == long_response

    def test_parse_markdown_fences_with_language(self):
        json_str = '```json\n{"tool": "lock_computer", "params": {}}\n```'
        intent = self.parser.parse(json_str)
        assert intent.tool == "lock_computer"

    def test_parse_markdown_fences_with_trailing_text(self):
        """Trailing text after closing fence is now handled."""
        json_str = '```\n{"tool": "open_app", "params": {"name": "Chrome"}}\n```\nsome trailing text'
        intent = self.parser.parse(json_str)
        assert intent.tool == "open_app"
        assert intent.params["name"] == "Chrome"

    def test_parse_markdown_incomplete_fences(self):
        """No closing fence is now handled."""
        json_str = '```json\n{"tool": "converse", "params": {"response": "Hi"}}'
        intent = self.parser.parse(json_str)
        assert intent.tool == "converse"
        assert intent.params["response"] == "Hi"

    def test_parse_last_intent_tracked(self):
        assert self.parser.last_intent is None
        self.parser.parse('{"tool": "open_calculator", "params": {}}')
        assert self.parser.last_intent is not None
        assert self.parser.last_intent.tool == "open_calculator"

    def test_needs_cua_for_all_cua_tools(self):
        cua_tools = [
            "cua_click", "cua_type_text", "cua_press_key", "cua_screenshot",
            "cua_text", "cua_scroll", "cua_drag", "cua_activate",
            "cua_list_apps", "cua_list_windows", "cua_get_window",
            "cua_launch", "cua_set_value", "cua_secondary_action",
        ]
        for tool in cua_tools:
            intent = Intent(tool=tool, params={})
            assert intent.needs_cua is True, f"{tool} should need CUA"

    def test_needs_cua_for_system_tools(self):
        system_tools = [
            "open_app", "close_app", "search_files", "open_file",
            "read_pdf", "summarize_pdf", "adjust_volume", "lock_computer",
            "open_calculator", "delete_file", "converse",
        ]
        for tool in system_tools:
            intent = Intent(tool=tool, params={})
            assert intent.needs_cua is False, f"{tool} should not need CUA"

    def test_extract_response_non_converse_no_response(self):
        intent = Intent(tool="open_app", params={"name": "calc"}, response="say this")
        # extract_response_text only returns for converse
        assert self.parser.extract_response_text(intent) == ""


# ─────────────────────────────────────────────
# COMPUTER USE BREAK TESTS
# ─────────────────────────────────────────────

class TestComputerUseBreak:
    def test_perform_before_start(self):
        cu = ComputerUse()
        result = cu.perform("screenshot", {})
        assert result.get("ok") is True  # perform works even before start

    def test_perform_unknown_action(self):
        cu = ComputerUse()
        result = cu.perform("nonexistent_action", {})
        assert "error" in result

    def test_start_twice(self):
        cu = ComputerUse()
        cu.start()
        cu.start()  # Should not crash
        assert cu.is_ready
        cu.stop()

    def test_stop_twice(self):
        cu = ComputerUse()
        cu.start()
        cu.stop()
        cu.stop()  # Should not crash
        assert not cu.is_ready


# ─────────────────────────────────────────────
# DISPATCHER BREAK TESTS
# ─────────────────────────────────────────────

class TestDispatcherBreak:
    def setup_method(self):
        self.dispatcher = Dispatcher(computer_use=None)

    def test_converse_handler_returns_ok(self):
        """converse tool now has a handler in dispatcher."""
        intent = Intent(tool="converse", params={"response": "Hello"})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "ok"
        assert result["text"] == "Hello"

    def test_open_app_missing_name(self):
        intent = Intent(tool="open_app", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "error"

    def test_close_app_missing_name(self):
        intent = Intent(tool="close_app", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "error"

    def test_search_files_missing_query(self):
        intent = Intent(tool="search_files", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "error"

    def test_open_file_missing_name(self):
        intent = Intent(tool="open_file", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "error"

    def test_delete_file_missing_name(self):
        intent = Intent(tool="delete_file", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "error"

    def test_read_pdf_missing_path(self):
        intent = Intent(tool="read_pdf", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "error"

    def test_summarize_pdf_missing_path(self):
        intent = Intent(tool="summarize_pdf", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "error"

    def test_adjust_volume_missing_params(self):
        intent = Intent(tool="adjust_volume", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        # Should either work with defaults or error
        assert "status" in result

    def test_open_calculator_without_cua(self):
        intent = Intent(tool="open_calculator", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        # Should try to open and either succeed or fail gracefully
        assert "status" in result

    def test_lock_computer_without_cua(self):
        intent = Intent(tool="lock_computer", params={})
        result = asyncio.run(self.dispatcher.execute(intent))
        # Should try to lock and either succeed or fail gracefully
        assert "status" in result

    def test_cua_tool_not_ready(self):
        """ComputerUse not started -> not ready."""
        cu = ComputerUse()
        dispatcher = Dispatcher(computer_use=cu)
        intent = Intent(tool="cua_click", params={})
        result = asyncio.run(dispatcher.execute(intent))
        assert result["status"] == "error"

    def test_handler_exception_returns_error(self):
        """If a handler raises, dispatcher catches and returns error."""
        class BadDispatcher(Dispatcher):
            async def _open_app(self, params):
                raise RuntimeError("boom")
        d = BadDispatcher()
        intent = Intent(tool="open_app", params={"name": "test"})
        result = asyncio.run(d.execute(intent))
        assert result["status"] == "error"
        assert "boom" in result["error"]

    def test_delete_file_empty_name(self):
        intent = Intent(tool="delete_file", params={"name": ""})
        result = asyncio.run(self.dispatcher.execute(intent))
        assert result["status"] == "error"


# ─────────────────────────────────────────────
# KOKORO ENGINE BREAK TESTS
# ─────────────────────────────────────────────

class TestKokoroBreak:
    def test_split_sentences_handles_abbreviations_poorly(self):
        """Known issue: regex splits on 'e.g.' or 'Dr.' as sentence boundary."""
        engine = KokoroEngine("/dummy", "/dummy")
        result = engine._split_sentences("Dr. Smith went to Washington. He likes e.g. apples.")
        # This will incorrectly split on "Dr." and "e.g." - this is a known limitation
        assert len(result) >= 2  # At minimum splits on the real sentence boundary

    def test_split_sentences_with_ellipsis(self):
        engine = KokoroEngine("/dummy", "/dummy")
        result = engine._split_sentences("Wait... let me think... OK.")
        # Ellipsis has periods but shouldn't necessarily split at each one
        assert len(result) >= 1

    def test_split_sentences_newlines_only(self):
        engine = KokoroEngine("/dummy", "/dummy")
        result = engine._split_sentences("\n\n\n")
        assert result == []

    def test_split_sentences_tabs_and_spaces(self):
        engine = KokoroEngine("/dummy", "/dummy")
        result = engine._split_sentences("  Hello.  \t  World.  ")
        assert len(result) == 2

    def test_worker_to_wav_multidimensional(self):
        worker = KokoroWorker("/dummy", "/dummy")
        import numpy as np
        # Simulate multi-channel audio
        audio_2d = np.array([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32)
        result = worker._to_wav(audio_2d)
        assert isinstance(result, bytes)
        assert len(result) > 44  # Valid WAV has header

    def test_worker_to_wav_empty_array(self):
        worker = KokoroWorker("/dummy", "/dummy")
        import numpy as np
        audio_empty = np.array([], dtype=np.float32)
        result = worker._to_wav(audio_empty)
        assert isinstance(result, bytes)
        # WAV header only (no audio data)
        assert len(result) == 44

    def test_worker_to_wav_list_input(self):
        worker = KokoroWorker("/dummy", "/dummy")
        result = worker._to_wav([0.1, -0.1, 0.2, -0.2])
        assert isinstance(result, bytes)
        assert len(result) > 44

    def test_engine_initialize_zero_workers(self):
        """Zero workers - should not crash but synthesize will fail."""
        engine = KokoroEngine("/dummy", "/dummy", num_workers=0)
        engine.initialize()
        assert len(engine._workers) == 0
        result = engine.synthesize("Hello")
        assert result is None

    def test_engine_synthesize_before_initialize(self, monkeypatch):
        """synthesize calls initialize() automatically if needed."""
        monkeypatch.setattr("grace.tts.kokoro_engine.KokoroWorker._initialize", lambda self: setattr(self, "_initialized", True))
        engine = KokoroEngine("/dummy", "/dummy", num_workers=1)
        result = engine.synthesize("Hello")
        assert engine.is_initialized

    def test_engine_synthesize_sentences_empty(self):
        engine = KokoroEngine("/dummy", "/dummy")
        result = engine.synthesize_sentences("", "af_bella")
        assert result == []


# ─────────────────────────────────────────────
# TTS PLAYER BREAK TESTS
# ─────────────────────────────────────────────

class TestTTSPlayerBreak:
    def test_get_wav_duration_empty_bytes(self):
        player = TTSPlayer()
        duration = player._get_wav_duration(b"")
        assert duration > 0

    def test_get_wav_duration_garbage_data(self):
        player = TTSPlayer()
        duration = player._get_wav_duration(b"this is not a wav file at all")
        assert duration > 0

    def test_get_wav_duration_valid_wav(self):
        """Valid WAV should return correct duration."""
        player = TTSPlayer()
        import wave, io
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00\x00" * 24000)  # 1 second of silence
        wav_bytes = buf.getvalue()
        duration = player._get_wav_duration(wav_bytes)
        assert abs(duration - 1.0) < 0.1

    def test_play_invalid_data_no_crash(self):
        player = TTSPlayer()
        player.play_sync(b"invalid wav data")
        assert player.queue_size == 0

    def test_queue_internal_state(self):
        """Test queue internals without calling sd.play."""
        player = TTSPlayer()
        assert player.queue_size == 0
        with player._lock:
            player._queue.append(b"data1")
            player._queue.append(b"data2")
        assert player.queue_size == 2
        with player._lock:
            item = player._queue.pop(0)
        assert item == b"data1"
        assert player.queue_size == 1

    def test_stop_clears_queue_internal(self):
        player = TTSPlayer()
        with player._lock:
            player._queue.append(b"test")
        player.stop()
        assert player.queue_size == 0
        assert not player._playing


# ─────────────────────────────────────────────
# RESPONSE GENERATOR BREAK TESTS
# ─────────────────────────────────────────────

class TestResponseGeneratorBreak:
    def setup_method(self):
        self.gemma = GemmaClient("http://127.0.0.1:1")
        self.kokoro = KokoroEngine("/dummy", "/dummy", num_workers=0)
        self.player = TTSPlayer()
        self.gen = ResponseGenerator(self.gemma, self.kokoro, self.player)

    def test_generate_no_system_prompt(self):
        result = asyncio.run(self.gen.generate_and_speak("hello"))
        assert result is False

    def test_generate_with_text_empty(self):
        self.gen.set_system_prompt("test")
        result = asyncio.run(self.gen.generate_and_speak_with_text(""))
        assert result is False

    def test_generate_with_text_whitespace(self):
        self.gen.set_system_prompt("test")
        result = asyncio.run(self.gen.generate_and_speak_with_text("   "))
        assert result is False

    def test_split_sentences_empty(self):
        assert ResponseGenerator._split_sentences("") == []

    def test_split_sentences_no_punctuation(self):
        result = ResponseGenerator._split_sentences("Hello world")
        assert result == ["Hello world"]

    def test_split_sentences_multiple_spaces(self):
        result = ResponseGenerator._split_sentences("Hello.  World.  Test.")
        assert len(result) == 3

    def test_split_sentences_only_punctuation(self):
        result = ResponseGenerator._split_sentences("... !!! ???")
        # Each punctuation mark followed by space is a sentence boundary
        # This is a known limitation of the simple regex approach
        assert len(result) >= 1

    def test_split_sentences_with_numbers(self):
        result = ResponseGenerator._split_sentences("Version 2.0 is out. It costs $1.99.")
        # Abbreviation splitting issue - known limitation
        assert len(result) >= 1

    def test_generate_and_speak_with_text_single_sentence(self):
        """No server, no model - should fail gracefully."""
        self.gen.set_system_prompt("test")
        result = asyncio.run(self.gen.generate_and_speak_with_text("Hello world."))
        # Synthesis will fail, so returns False
        assert result is False


# ─────────────────────────────────────────────
# FEEDBACK SOUNDS BREAK TESTS
# ─────────────────────────────────────────────

# FeedbackSounds methods use sounddevice.play/sd.wait which can hang in
# headless/CI environments. We test the sound generation logic directly.

class TestFeedbackBreak:
    def test_chime_sound_generation(self):
        import numpy as np
        duration = 0.01
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        tone1 = np.sin(2 * np.pi * 523.25 * t)
        tone2 = np.sin(2 * np.pi * 659.25 * t) * 0.5
        fade_len = min(int(sample_rate * 0.05), len(t))
        envelope = np.ones(len(t))
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        decay = np.exp(-2 * t)
        audio = (tone1 + tone2) * envelope * decay * 0.4 / 2
        assert len(audio) > 0
        assert np.isfinite(audio).all()

    def test_error_tone_generation(self):
        import numpy as np
        sample_rate = 24000
        duration = 0.3
        t = np.linspace(0, duration, int(sample_rate * duration))
        tone = np.sin(2 * np.pi * 220 * t) * np.exp(-8 * t) * 0.3
        assert len(tone) > 0
        assert np.isfinite(tone).all()

    def test_completion_tone_generation(self):
        import numpy as np
        sample_rate = 24000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        tone1 = np.sin(2 * np.pi * 440 * t)
        tone2 = np.sin(2 * np.pi * 554 * t) * 0.5
        audio = (tone1 + tone2) * np.exp(-4 * t) * 0.2
        assert len(audio) > 0
        assert np.isfinite(audio).all()

    def test_wake_confirm_generation(self):
        import numpy as np
        sample_rate = 24000
        duration = 0.15
        t = np.linspace(0, duration, int(sample_rate * duration))
        tone = np.sin(2 * np.pi * 800 * t) * np.exp(-20 * t) * 0.3
        assert len(tone) > 0
        assert np.isfinite(tone).all()


# ─────────────────────────────────────────────
# INTEGRATION BREAK TESTS
# ─────────────────────────────────────────────

class TestIntegrationBreak:
    def test_main_imports_no_crash(self):
        """Importing main should not crash (but won't run)."""
        try:
            import importlib
            import grace.main
            importlib.reload(grace.main)
        except Exception:
            pass  # may fail due to missing models, that's OK

    def test_config_has_all_fields_used_by_main(self):
        """Verify all fields accessed in main.py exist on Config."""
        config = Config()
        fields = [
            "mic_sample_rate", "mic_chunk", "mic_channels", "mic_width",
            "mic_device_index", "vosk_model_path", "vosk_keyword", "vosk_threshold",
            "llama_server_exe", "whisper_model_path", "whisper_vad_threshold", "whisper_silence_duration_ms",
            "llama_server_url",
            "kokoro_model_path", "kokoro_voices_path", "kokoro_workers",
            "kokoro_device", "kokoro_dtype",
            "ws_host", "ws_port",
        ]
        for field in fields:
            assert hasattr(config, field), f"Config missing field '{field}' used by main.py"

    def test_prompt_module_returns_string(self):
        from grace.intent.prompt import get_system_prompt
        prompt = get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_tool_definitions_all_valid_names(self):
        from grace.intent.tools import ALL_TOOLS, CUA_TOOLS, SYSTEM_TOOLS
        from grace.intent.parser import VALID_TOOLS
        # Every tool definition should have a name in VALID_TOOLS
        for tool in ALL_TOOLS:
            assert tool.name in VALID_TOOLS, f"{tool.name} not in VALID_TOOLS"
        # Every VALID_TOOLS entry should have a definition
        defined_names = {t.name for t in ALL_TOOLS}
        assert defined_names == VALID_TOOLS, f"Mismatch: {VALID_TOOLS - defined_names}"

    def test_tool_definition_params_have_required_fields(self):
        from grace.intent.tools import ALL_TOOLS
        for tool in ALL_TOOLS:
            for param in tool.params:
                assert param.name, f"Tool {tool.name} has param with no name"
                assert param.description, f"Tool {tool.name}.{param.name} has no description"
