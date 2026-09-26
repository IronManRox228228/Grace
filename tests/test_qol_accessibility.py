"""Comprehensive tests for Accessibility & Quality of Life (QoL) features."""

import asyncio
import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.response.feedback import FeedbackSounds
from grace.intent.parser import Intent
from grace.intent.router import CapabilityRouter, TaskComplexity
from grace.tools.dispatcher import Dispatcher
from grace.tts.kokoro_engine import KokoroWorker, KokoroEngine
from grace.response.generator import ResponseGenerator
from grace.main import GraceApp


class TestSyntheticEarcons:
    """Verify synthetic earcons execute safely without throwing exceptions."""

    def test_play_chime_runs_safely(self):
        # Sound playback should complete or degrade gracefully without uncaught exceptions
        FeedbackSounds.play_chime()

    def test_play_success_runs_safely(self):
        FeedbackSounds.play_success()

    def test_play_cancel_runs_safely(self):
        FeedbackSounds.play_cancel()

    def test_play_error_runs_safely(self):
        FeedbackSounds.play_error()

    def test_play_listening_runs_safely(self):
        FeedbackSounds.play_listening()

    @patch("sounddevice.play", side_effect=Exception("Audio device busy"))
    def test_sounddevice_error_degrades_gracefully(self, mock_play):
        # Even if sounddevice fails, earcons should catch the error and not crash the caller
        FeedbackSounds.play_success()
        FeedbackSounds.play_cancel()
        FeedbackSounds.play_error()
        FeedbackSounds.play_listening()


class TestKokoroSpeedAndCache:
    """Verify TTS speech speed control and cache handling."""

    def test_worker_speed_clamping(self):
        worker = KokoroWorker(model_path="/dummy", voices_path="/dummy", device="cpu", speed=1.0)
        assert worker.speed == 1.0

        worker.set_speed(3.5)
        assert worker.speed == 2.5

        worker.set_speed(0.1)
        assert worker.speed == 0.5

        worker.set_speed(1.2)
        assert worker.speed == 1.2
        worker.shutdown()

    def test_engine_speed_propagation_and_cache_clear(self):
        engine = KokoroEngine(model_path="/dummy", voices_path="/dummy", num_workers=2, warmup=False, speed=1.0)
        assert engine.speed == 1.0

        # Pre-fill cache
        engine._cache_put(("Test phrase", "af_bella"), b"fake_wav_1")
        assert engine._cache_get(("Test phrase", "af_bella")) == b"fake_wav_1"

        # Updating speed clears old cache
        engine.set_speed(1.5)
        assert engine.speed == 1.5
        assert engine._cache_get(("Test phrase", "af_bella", 1.0)) is None

        engine.shutdown()


class TestResponseGeneratorStopFlag:
    """Verify generator stops synthesis immediately upon barge-in."""

    @pytest.mark.asyncio
    async def test_generator_stop_speaking_sets_flag(self):
        player_mock = MagicMock()
        player_mock.is_playing = False
        generator = ResponseGenerator(
            gemma_client=MagicMock(),
            kokoro_engine=MagicMock(),
            tts_player=player_mock,
        )
        assert not generator._stopped
        assert not generator.is_speaking

        generator.stop_speaking()
        assert generator._stopped
        player_mock.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_and_play_aborts_on_stopped(self):
        player_mock = MagicMock()
        kokoro_mock = MagicMock()
        del kokoro_mock.submit
        del kokoro_mock.resolve
        kokoro_mock.synthesize.return_value = b"wav_data"

        generator = ResponseGenerator(
            gemma_client=MagicMock(),
            kokoro_engine=kokoro_mock,
            tts_player=player_mock,
        )

        def side_effect_synth(*args, **kwargs):
            generator.stop_speaking()
            return b"wav_data"

        kokoro_mock.synthesize.side_effect = side_effect_synth

        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        success = await generator._synthesize_and_play(sentences, voice="af_bella")

        # After the first synthesis triggers stop_speaking, subsequent sentences must NOT be played
        assert generator._stopped
        assert kokoro_mock.synthesize.call_count <= 2


class TestCapabilityRouterAccessibilityFastPath:
    """Verify atomic accessibility & navigation tools take the 20ms fast path."""

    def test_cua_press_key_atomic_routes_fast_path(self):
        intent = Intent(tool="cua_press_key", params={"key": "Return"})
        assert CapabilityRouter.classify("press enter", intent) == TaskComplexity.FAST_PATH

    def test_cua_scroll_atomic_routes_fast_path(self):
        intent = Intent(tool="cua_scroll", params={"scrollY": 500})
        assert CapabilityRouter.classify("scroll down", intent) == TaskComplexity.FAST_PATH

    def test_cua_activate_atomic_routes_fast_path(self):
        intent = Intent(tool="cua_activate", params={"window": {"title": "Chrome"}})
        assert CapabilityRouter.classify("switch to chrome", intent) == TaskComplexity.FAST_PATH

    def test_undo_routes_fast_path(self):
        intent = Intent(tool="undo", params={})
        assert CapabilityRouter.classify("undo", intent) == TaskComplexity.FAST_PATH

    def test_describe_screen_routes_fast_path(self):
        intent = Intent(tool="describe_screen", params={})
        assert CapabilityRouter.classify("what's on my screen", intent) == TaskComplexity.FAST_PATH

    def test_set_speech_rate_routes_fast_path(self):
        intent = Intent(tool="set_speech_rate", params={"rate": "0.8"})
        assert CapabilityRouter.classify("speak slower", intent) == TaskComplexity.FAST_PATH

    def test_chained_request_with_press_key_routes_agentic(self):
        intent = Intent(tool="cua_press_key", params={"key": "Return"})
        assert CapabilityRouter.classify("open notepad and press enter", intent) == TaskComplexity.AGENTIC_GOAL


class TestDispatcherQoLTools:
    """Verify Dispatcher handlers for undo, describe_screen, and set_speech_rate."""

    @pytest.mark.asyncio
    @patch("pyautogui.hotkey")
    async def test_undo_handler(self, mock_hotkey):
        dispatcher = Dispatcher(computer_use=None)
        intent = Intent(tool="undo", params={})
        result = await dispatcher.execute(intent)
        assert result["status"] == "ok"
        assert result["text"] == "Undone."
        mock_hotkey.assert_called_once_with("ctrl", "z")

    @pytest.mark.asyncio
    async def test_describe_screen_handler(self):
        dispatcher = Dispatcher(computer_use=None)
        intent = Intent(tool="describe_screen", params={})
        result = await dispatcher.execute(intent)
        assert result["status"] == "ok"
        assert "text" in result
        assert "window_title" in result

    @pytest.mark.asyncio
    async def test_set_speech_rate_handler(self):
        kokoro_mock = MagicMock()
        dispatcher = Dispatcher(computer_use=None, kokoro_engine=kokoro_mock)

        # Numeric string
        intent = Intent(tool="set_speech_rate", params={"rate": "1.25"})
        result = await dispatcher.execute(intent)
        assert result["status"] == "ok"
        assert result["speed"] == 1.25
        kokoro_mock.set_speed.assert_called_with(1.25)

        # Keyword 'slower'
        intent_slow = Intent(tool="set_speech_rate", params={"rate": "slower"})
        result_slow = await dispatcher.execute(intent_slow)
        assert result_slow["status"] == "ok"
        assert result_slow["speed"] == 0.8
        kokoro_mock.set_speed.assert_called_with(0.8)


class TestVoiceReflexes:
    """Verify immediate stop and repeat reflexes."""

    @pytest.mark.asyncio
    async def test_stop_reflex_in_activation_turn(self):
        app = GraceApp.__new__(GraceApp)
        app.response_gen = MagicMock()
        app.agent_loop = MagicMock()
        app.agent_loop.has_pending_confirmation = False
        app.ws_server = AsyncMock()

        # Mock transcribe returning "stop"
        app.pump = MagicMock()
        app.pump.drain = MagicMock()
        app.pump.get = AsyncMock(return_value=None)
        app.whisper = MagicMock()
        app.whisper.transcribe = MagicMock(return_value="stop")
        app.whisper.add_buffer = MagicMock()
        app.whisper.reset_buffer = MagicMock()
        app.vad = MagicMock()
        app._running = True
        app.config = MagicMock()
        app.config.initial_listen_timeout_seconds = 0.05
        app.config.whisper_vad_threshold = 0.01
        app.wake_word = MagicMock()

        trace_mock = MagicMock()
        trace_mock.stage.return_value.__enter__ = MagicMock()
        trace_mock.stage.return_value.__exit__ = MagicMock()
        trace_mock.stage.return_value.detail = MagicMock()

        res = await app._handle_activation_turn(trace_mock)
        assert res is False
        app.response_gen.stop_speaking.assert_called_once()

    @pytest.mark.asyncio
    async def test_repeat_reflex_replays_last_text(self):
        app = GraceApp.__new__(GraceApp)
        app._last_spoken_text = "The quick brown fox jumps over the lazy dog."
        app._speak_response = AsyncMock()
        app.response_gen = MagicMock()
        app.agent_loop = MagicMock()
        app.agent_loop.has_pending_confirmation = False
        app.ws_server = AsyncMock()

        app.pump = MagicMock()
        app.pump.drain = MagicMock()
        app.pump.get = AsyncMock(return_value=None)
        app.whisper = MagicMock()
        app.whisper.transcribe = MagicMock(return_value="what did you say")
        app.whisper.add_buffer = MagicMock()
        app.whisper.reset_buffer = MagicMock()
        app.vad = MagicMock()
        app._running = True
        app.config = MagicMock()
        app.config.initial_listen_timeout_seconds = 0.05
        app.config.whisper_vad_threshold = 0.01
        app.wake_word = MagicMock()

        trace_mock = MagicMock()
        trace_mock.stage.return_value.__enter__ = MagicMock()
        trace_mock.stage.return_value.__exit__ = MagicMock()
        trace_mock.stage.return_value.detail = MagicMock()

        res = await app._handle_activation_turn(trace_mock)
        assert res is True
        app._speak_response.assert_called_once_with("The quick brown fox jumps over the lazy dog.")
