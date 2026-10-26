"""Offline guarantee and R19 privacy tests.

Ensures that with cloud fallback disabled, Grace makes zero non-loopback network
connections at runtime, and that cloud fallback never sends screenshots without
explicit per-session consent (R19).
"""

import asyncio
import dataclasses
import ipaddress
import socket
from unittest import mock

import pytest

from grace.config import Config
from grace.llm.gemma_client import GemmaClient


def _is_loopback(host: str) -> bool:
    if host.lower() in ("localhost", "tauri.localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback
    except ValueError:
        return False


class TestOfflineGuarantee:
    def test_scripted_turn_makes_zero_non_loopback_connections(self, monkeypatch, tmp_path):
        """A full turn with cloud fallback disabled must make zero non-loopback connections."""
        from grace.harness import replay as replay_mod

        transcript = "open calculator"
        config = dataclasses.replace(
            Config(),
            use_cloud_llm=False,
            whisper_silence_duration_ms=275,
            answer_timeout_seconds=0,
        )

        connections_attempted = []
        real_connect = socket.socket.connect

        def guarded_connect(self, address):
            host = address[0] if isinstance(address, tuple) else str(address)
            port = address[1] if isinstance(address, tuple) and len(address) > 1 else None
            connections_attempted.append((host, port))

            if not _is_loopback(host):
                raise AssertionError(f"PROHIBITED non-loopback connection to {host}:{port}")

            return real_connect(self, address)

        monkeypatch.setattr(socket.socket, "connect", guarded_connect)

        # Mock out external hardware/GUI
        patches = [
            mock.patch("grace.main.Config", lambda: config),
            mock.patch("grace.main.AudioCapture", replay_mod.NullAudio),
            mock.patch("grace.main.WakeWordDetector", replay_mod.NullWakeWord),
            mock.patch("grace.main.KokoroEngine", replay_mod.NullTts),
            mock.patch("grace.main.TTSPlayer", replay_mod.NullTts),
            mock.patch("grace.main.ComputerUse", replay_mod.NullComputerUse),
            mock.patch("grace.main.FeedbackSounds", replay_mod._NullFeedbackSounds),
            mock.patch("grace.main.WhisperStreaming", lambda *a, **k: mock.MagicMock(transcribe=lambda: transcript, warmup=lambda: True)),
            mock.patch("grace.main.WsEventServer", lambda *a, **k: mock.MagicMock(
                start=mock.AsyncMock(), stop=mock.AsyncMock(), emit=mock.AsyncMock(), is_connected=False
            )),
        ]

        def _speech_then_silence(chunk_bytes=1024, speech=4, silence=15):
            loud = (8000).to_bytes(2, "little", signed=True) * (chunk_bytes // 2)
            quiet = b"\x00\x00" * (chunk_bytes // 2)
            return [loud] * speech + [quiet] * silence

        class ScriptedPump:
            def __init__(self, chunks):
                self._chunks = list(chunks)
                self._index = 0
            def start(self, loop=None): return self
            def stop(self): pass
            async def get(self, timeout=None):
                if self._index >= len(self._chunks):
                    return None
                chunk = self._chunks[self._index]
                self._index += 1
                await asyncio.sleep(0.01)
                from grace.audio.pump import _tape
                return _tape(chunk)
            def drain(self): return 0
            def gate(self, duration_seconds: float) -> None: pass

        class FakeWhisperModel:
            def __init__(self, transcript):
                self._transcript = transcript
            def transcribe(self, audio, **kwargs):
                class FakeSegment:
                    def __init__(self, text):
                        self.text = text
                return iter([FakeSegment(self._transcript)]), None

        import grace.main as main_mod
        with ExitStack_helper(patches):
            app = main_mod.GraceApp()
            app._running = True
            app.MIN_WAKE_TO_IDLE_SECONDS = 0
            app.pump = ScriptedPump(_speech_then_silence())
            app.whisper._model = FakeWhisperModel(transcript)
            app.whisper._initialized = True
            app.vad = mock.MagicMock()
            app.vad.process_chunk.return_value = True
            app.capture = mock.MagicMock()
            app.capture.get_rms.return_value = 1000.0
            app._run_answer_window = mock.AsyncMock()
            app.gemma.generate_intent = mock.AsyncMock(
                return_value='{"tool": "converse", "params": {"response": "Hello world"}}'
            )
            app.response_gen.generate_and_speak_with_text = mock.AsyncMock()

            # Run activation turn
            asyncio.run(app._handle_activation())

        # Verify all attempted connections were loopback
        for host, port in connections_attempted:
            assert _is_loopback(host), f"Non-loopback connection detected: {host}:{port}"


class TestR19CloudScreenshotConsent:
    @pytest.mark.asyncio
    async def test_cloud_fallback_refuses_screenshots_without_consent(self):
        client = GemmaClient(api_key="fake-gemini-key", allow_cloud_screenshots=False)
        messages = [
            {"role": "user", "content": "look at this screen", "image_b64": "iVBORw0KGgoAAAANSUhEUg=="}
        ]

        with pytest.raises(PermissionError, match="Cloud fallback is not permitted to send screenshots"):
            await client.chat(messages)

    @pytest.mark.asyncio
    async def test_cloud_fallback_allows_text_without_screenshot_consent(self):
        client = GemmaClient(api_key="fake-gemini-key", allow_cloud_screenshots=False)
        messages = [{"role": "user", "content": "what time is it"}]

        with mock.patch.object(client, "_stream_gemini_response", return_value=_dummy_stream(["4:00"])):
            gen = await client.chat(messages)
            chunks = [c async for c in gen]
            assert chunks == ["4:00"]

    @pytest.mark.asyncio
    async def test_cloud_fallback_allows_screenshots_with_explicit_consent(self):
        client = GemmaClient(api_key="fake-gemini-key", allow_cloud_screenshots=False)
        client.set_cloud_screenshot_consent(True)

        messages = [
            {"role": "user", "content": "look at this screen", "image_b64": "iVBORw0KGgoAAAANSUhEUg=="}
        ]

        with mock.patch.object(client, "_stream_gemini_response", return_value=_dummy_stream(["button at 100, 200"])):
            gen = await client.chat(messages)
            chunks = [c async for c in gen]
            assert chunks == ["button at 100, 200"]


class ExitStack_helper:
    def __init__(self, patches):
        self.patches = patches
    def __enter__(self):
        for p in self.patches:
            p.start()
        return self
    def __exit__(self, *args):
        for p in reversed(self.patches):
            p.stop()


async def _dummy_stream(items):
    for item in items:
        yield item
