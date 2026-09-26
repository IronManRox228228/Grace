"""Regression tests for a batch of ship-blocking defects found in review.

Each class covers one defect. Nothing here drives the real microphone,
speakers, screen or desktop - components that would are replaced with mocks
built directly on `GraceApp.__new__(GraceApp)`, the same pattern already used
in test_qol_accessibility.py to exercise one method without paying for the
constructor's real audio/model/network setup.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.main import GraceApp
from grace.response.feedback import FeedbackSounds


def make_app(**attrs):
    """A GraceApp with none of __init__'s real components constructed."""
    app = GraceApp.__new__(GraceApp)
    app._running = True
    app.wake_word = MagicMock()
    app.ws_server = AsyncMock()
    for name, value in attrs.items():
        setattr(app, name, value)
    return app


class TestWakeWordResumesAfterActivation:
    """main.py:591 pauses the detector; only `_run_followup_window` resumed it,
    and main.py:577 now skips that window whenever `turn_result is False` - so
    an aborted turn (silence, a stop command, a transcription error) left the
    detector paused forever and "Grace" was never heard again.
    """

    def test_a_false_turn_result_leaves_the_detector_resumed(self):
        app = make_app(_run_followup_window=AsyncMock())

        async def aborted(trace):
            return False

        app._handle_activation_turn = aborted
        asyncio.run(app._handle_activation())

        app.wake_word.resume.assert_called_once()
        app._run_followup_window.assert_not_awaited()

    def test_a_successful_turn_leaves_resuming_to_the_followup_window(self):
        app = make_app(_run_followup_window=AsyncMock(), config=MagicMock(followup_timeout_seconds=10))

        async def ok(trace):
            return True

        app._handle_activation_turn = ok
        asyncio.run(app._handle_activation())

        # `_run_followup_window` pauses/resumes the detector itself; resuming
        # it again here too would just be redundant, not wrong, but asserting
        # it was NOT called here pins the intended division of responsibility.
        app.wake_word.resume.assert_not_called()
        app._run_followup_window.assert_awaited_once()

    def test_an_exception_still_resumes_the_detector(self):
        app = make_app()

        async def crashes(trace):
            raise RuntimeError("boom")

        app._handle_activation_turn = crashes
        with pytest.raises(RuntimeError):
            asyncio.run(app._handle_activation())

        app.wake_word.resume.assert_called_once()


class TestConfirmationAnswerIsStrict:
    """main.py:466-486 matched a yes/no word anywhere in the utterance, so
    "I am not sure" and "okay open spotify" (an unrelated new request) both
    read as approval of whatever was parked.
    """

    @pytest.mark.parametrize("utterance", [
        "I am not sure", "okay open spotify", "click OK",
    ])
    def test_unrelated_speech_is_not_read_as_an_answer(self, utterance):
        assert GraceApp._confirmation_answer(utterance) is None

    @pytest.mark.parametrize("utterance", [
        "yes", "Yes.", "okay", "yes please", "go ahead",
    ])
    def test_plain_affirmatives_are_recognised(self, utterance):
        assert GraceApp._confirmation_answer(utterance) is True

    @pytest.mark.parametrize("utterance", ["no", "don't"])
    def test_plain_negatives_are_recognised(self, utterance):
        assert GraceApp._confirmation_answer(utterance) is False


class TestFollowupTimeoutCancelsPendingConfirmation:
    """A parked confirmation must not survive its own follow-up window closing
    unanswered - otherwise a later, unrelated "yes" can approve it.
    """

    def test_timing_out_with_nothing_said_cancels_the_pending_step(self, monkeypatch):
        monkeypatch.setattr(FeedbackSounds, "play_listening", lambda *a, **k: None)

        app = make_app(
            vad=MagicMock(),
            whisper=MagicMock(),
            pump=MagicMock(),
            agent_loop=MagicMock(),
            _flush_mic_buffer=MagicMock(),
        )
        app.agent_loop.has_pending_confirmation = True

        # timeout_seconds=0 makes the inner listening loop time out on its
        # very first check, with nothing captured or handled.
        asyncio.run(app._run_followup_window(timeout_seconds=0))

        app.agent_loop.cancel_pending.assert_called_once()
        app.wake_word.resume.assert_called_once()


class TestResumeSpeaksANewConfirmation:
    """`resume_pending` can itself return `safety_confirmation_required` (an
    approved step led straight into another one that also needs asking). The
    old code only ever said "Done." or "Alright, I won't do that." for a
    resume, so the second question was never spoken and its pending step was
    silently dropped.
    """

    def test_a_second_pending_step_is_announced_not_marked_done(self):
        app = make_app(response_gen=AsyncMock())
        app.agent_loop = MagicMock()
        app.agent_loop.resume_pending = AsyncMock(return_value={
            "status": "safety_confirmation_required",
            "confirmation_prompt": "Should I also delete the backup?",
        })

        handled = asyncio.run(app._resolve_pending_confirmation("yes"))

        assert handled is True
        spoken = app.response_gen.generate_and_speak_with_text.call_args.args[0]
        assert spoken == "Should I also delete the backup?"
        assert spoken != "Done."

    def test_an_ordinary_resume_still_says_done(self):
        app = make_app(response_gen=AsyncMock())
        app.agent_loop = MagicMock()
        app.agent_loop.resume_pending = AsyncMock(return_value={"status": "ok"})

        asyncio.run(app._resolve_pending_confirmation("yes"))

        spoken = app.response_gen.generate_and_speak_with_text.call_args.args[0]
        assert spoken == "Done."


class TestTurnCrashesDoNotKillGrace:
    """main.py's main loop only caught KeyboardInterrupt, so any exception
    escaping a turn - a bad transcript, a dispatcher bug - shut Grace down
    entirely, with no way for a voice-only user to restart it.
    """

    def test_a_crashing_turn_is_caught_logged_and_survived(self, monkeypatch):
        played = []
        monkeypatch.setattr(FeedbackSounds, "play_error", lambda *a, **k: played.append(1))

        app = make_app()

        async def crashes():
            raise ValueError("unexpected")

        app._handle_activation = crashes
        asyncio.run(app._run_activation_turn())  # must not raise

        assert played == [1]
        app.wake_word.resume.assert_called_once()

    @pytest.mark.parametrize("exc", [KeyboardInterrupt, asyncio.CancelledError])
    def test_keyboard_interrupt_and_cancellation_are_not_swallowed(self, exc):
        app = make_app()

        async def crashes():
            raise exc()

        app._handle_activation = crashes
        with pytest.raises(exc):
            asyncio.run(app._run_activation_turn())


class TestWebSocketOriginAllowlist:
    """ws_server.py accepted a handshake from any Origin, so any page open in
    the user's browser could connect, read live transcripts, and send a fake
    `{"type": "wake"}`.
    """

    def test_no_origin_header_is_allowed(self):
        from grace.ws_server import DEFAULT_ALLOWED_ORIGINS, is_origin_allowed

        assert is_origin_allowed(None, DEFAULT_ALLOWED_ORIGINS) is True

    @pytest.mark.parametrize("origin", [
        "http://tauri.localhost", "https://tauri.localhost",
        "tauri://localhost", "http://localhost:5173",
    ])
    def test_known_renderer_origins_are_allowed(self, origin):
        from grace.ws_server import DEFAULT_ALLOWED_ORIGINS, is_origin_allowed

        assert is_origin_allowed(origin, DEFAULT_ALLOWED_ORIGINS) is True

    def test_an_arbitrary_web_page_is_rejected(self):
        from grace.ws_server import DEFAULT_ALLOWED_ORIGINS, is_origin_allowed

        assert is_origin_allowed("http://evil.example", DEFAULT_ALLOWED_ORIGINS) is False

    def test_ws_allowed_origins_env_extends_but_does_not_replace_the_defaults(self):
        from grace.ws_server import WsEventServer

        server = WsEventServer(allowed_origins="http://example.test, http://foo.test")

        assert "http://example.test" in server._allowed_origins
        assert "http://foo.test" in server._allowed_origins
        assert "http://localhost:5173" in server._allowed_origins

    def test_handshake_from_a_disallowed_origin_gets_403(self):
        from aiohttp import WSServerHandshakeError, web
        from aiohttp.test_utils import TestClient, TestServer

        from grace.ws_server import WsEventServer

        server = WsEventServer(allowed_origins="")
        app = web.Application()
        app.router.add_get("/", server._handler)

        async def run():
            async with TestClient(TestServer(app)) as client:
                with pytest.raises(WSServerHandshakeError):
                    await client.ws_connect("/", headers={"Origin": "http://evil.example"})

        asyncio.run(run())

    def test_handshake_from_an_allowed_origin_succeeds(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from grace.ws_server import WsEventServer

        server = WsEventServer(allowed_origins="")
        app = web.Application()
        app.router.add_get("/", server._handler)

        async def run():
            async with TestClient(TestServer(app)) as client:
                async with client.ws_connect("/", headers={"Origin": "http://localhost:5173"}) as ws:
                    assert not ws.closed

        asyncio.run(run())
