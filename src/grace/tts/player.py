"""TTS audio playback.

One persistent playback thread drains a queue of decoded PCM into a single
long-lived output stream. The previous implementation wrote every sentence to a
temp file on disk and spawned a fresh daemon thread to hand it to
``winsound.PlaySound(SND_FILENAME)``, which meant disk I/O, thread churn and an
audible gap between sentences.

``winsound`` is kept as a fallback because it is why this module exists: the
original PortAudio playback path contended with PyAudio's microphone handle.
The chime already plays through sounddevice alongside a live mic, so the
streaming path is the default, but any failure to open the stream degrades to
the old temp-file behaviour rather than losing audio.
"""

import io
import logging
import os
import tempfile
import threading
import time
import wave
from collections import deque
from typing import Optional

logger = logging.getLogger("grace.tts.player")

try:  # pragma: no cover - availability depends on the machine
    import winsound
except ImportError:  # pragma: no cover
    winsound = None

BACKEND_STREAM = "stream"
BACKEND_WINSOUND = "winsound"

# How long the output stream stays open after the last chunk. Reopening costs
# tens of ms and Grace usually speaks several sentences in a row.
STREAM_IDLE_TIMEOUT = 30.0


def decode_wav(wav_bytes: bytes) -> tuple[bytes, int, int, int]:
    """Return (pcm_frames, sample_rate, channels, sample_width) for a WAV blob."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return (
            wf.readframes(wf.getnframes()),
            wf.getframerate(),
            wf.getnchannels(),
            wf.getsampwidth(),
        )


class TTSPlayer:
    """Sequential audio playback queue for TTS output."""

    def __init__(self, sample_rate: int = 24000, backend: Optional[str] = None):
        self._sample_rate = sample_rate
        self._queue: deque = deque()
        self._cv = threading.Condition(threading.RLock())
        self._lock = self._cv  # preserved: callers used to take ._lock
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._active = False
        # Bumped by stop(); the writer drops anything from an older generation
        # so barge-in cannot be overtaken by audio already in flight.
        self._generation = 0
        self._stream = None
        self._stream_params: Optional[tuple[int, int, int]] = None
        self._backend = backend or self._default_backend()

    @staticmethod
    def _default_backend() -> str:
        try:
            import sounddevice  # noqa: F401
            return BACKEND_STREAM
        except Exception:
            return BACKEND_WINSOUND

    # -- public API ------------------------------------------------------

    def play(self, wav_bytes: bytes):
        """Queue WAV audio bytes for playback."""
        if not wav_bytes:
            return
        logger.debug(f"TTSPlayer.play called with {len(wav_bytes)} bytes")
        with self._cv:
            self._queue.append(wav_bytes)
            self._ensure_thread()
            self._cv.notify_all()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until everything queued has finished playing.

        The default budget is derived from how much audio is actually queued.
        The old fixed 15 s cap reported "speech finished" early on any response
        longer than 15 s of audio.
        """
        deadline = time.monotonic() + (timeout if timeout is not None else self._pending_timeout())
        with self._cv:
            while self._queue or self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("TTSPlayer.wait timed out with audio still queued")
                    return False
                self._cv.wait(min(remaining, 0.25))
        return True

    def stop(self):
        """Stop playback immediately and drop anything queued (barge-in)."""
        with self._cv:
            self._queue.clear()
            self._generation += 1
            self._active = False
            self._cv.notify_all()

        stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except Exception as e:
                logger.debug(f"Output stream abort exception: {e}")

        if winsound is not None:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception as e:
                logger.debug(f"winsound stop purge exception: {e}")

        logger.info("TTSPlayer playback cancelled via Barge-In stop()")

    def play_sync(self, wav_bytes: bytes):
        """Play WAV bytes synchronously (blocking)."""
        self._write_chunk(wav_bytes, self._generation)

    def close(self):
        """Shut down the playback thread and release the output stream."""
        with self._cv:
            self._running = False
            self._queue.clear()
            self._cv.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._close_stream()

    @property
    def queue_size(self) -> int:
        """Current number of chunks in the queue."""
        with self._cv:
            return len(self._queue)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def _playing(self) -> bool:
        """True while audio is being written or is still queued."""
        with self._cv:
            return self._active or bool(self._queue)

    # -- internals -------------------------------------------------------

    def _pending_timeout(self) -> float:
        """Enough time for the queued audio to actually play, plus slack."""
        with self._cv:
            queued = list(self._queue)
        seconds = sum(self._get_wav_duration(chunk) for chunk in queued)
        return max(15.0, seconds + 10.0)

    def _ensure_thread(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="TTSPlayer", daemon=True
        )
        self._thread.start()

    def _run(self):
        last_activity = time.monotonic()
        while True:
            with self._cv:
                while self._running and not self._queue:
                    # Drop the output stream once the conversation goes quiet
                    # rather than holding the device open forever.
                    if self._stream is not None and time.monotonic() - last_activity > STREAM_IDLE_TIMEOUT:
                        break
                    self._cv.wait(1.0)

                if not self._running:
                    break
                if not self._queue:
                    if self._stream is not None:
                        self._close_stream()
                    continue

                wav_bytes = self._queue.popleft()
                generation = self._generation
                self._active = True

            try:
                self._write_chunk(wav_bytes, generation)
            except Exception as e:
                logger.error(f"Failed to play audio chunk: {e}")
            finally:
                last_activity = time.monotonic()
                with self._cv:
                    self._active = False
                    self._cv.notify_all()

        self._close_stream()

    def _write_chunk(self, wav_bytes: bytes, generation: int):
        duration = self._get_wav_duration(wav_bytes)
        logger.info(
            f"Playing TTS audio chunk ({duration:.2f}s, {len(wav_bytes)} bytes) via {self._backend}"
        )
        if self._backend == BACKEND_STREAM and self._write_stream(wav_bytes, generation):
            return
        self._write_winsound(wav_bytes)

    def _write_stream(self, wav_bytes: bytes, generation: int) -> bool:
        """Write PCM to the persistent output stream. False means fall back."""
        try:
            pcm, rate, channels, width = decode_wav(wav_bytes)
        except Exception as e:
            logger.debug(f"WAV decode failed ({e}); falling back to winsound")
            return False

        if not pcm:
            return True

        try:
            stream = self._ensure_stream(rate, channels, width)
            if stream is None:
                return False
            # Barge-in may have fired while the stream was being opened.
            if generation != self._generation:
                return True
            stream.write(pcm)
            return True
        except Exception as e:
            logger.warning(f"Streaming playback failed ({e}); falling back to winsound")
            self._close_stream()
            self._backend = BACKEND_WINSOUND
            return False

    def _ensure_stream(self, rate: int, channels: int, width: int):
        params = (rate, channels, width)
        if self._stream is not None and self._stream_params == params:
            return self._stream

        self._close_stream()

        import sounddevice as sd

        dtype = {1: "int8", 2: "int16", 4: "int32"}.get(width)
        if dtype is None:
            logger.debug(f"Unsupported sample width {width}; falling back to winsound")
            return None

        stream = sd.RawOutputStream(samplerate=rate, channels=channels, dtype=dtype)
        stream.start()
        self._stream = stream
        self._stream_params = params
        logger.debug(f"Opened output stream {rate}Hz {channels}ch {dtype}")
        return stream

    def _close_stream(self):
        stream, self._stream = self._stream, None
        self._stream_params = None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            logger.debug(f"Output stream close exception: {e}")

    def _write_winsound(self, wav_bytes: bytes):
        if winsound is None:
            logger.error("No audio backend available; dropping chunk")
            return
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name
            winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
        except Exception as e:
            logger.error(f"Failed to play audio chunk via winsound: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _get_wav_duration(self, wav_bytes: bytes) -> float:
        """Calculate duration of WAV audio bytes in seconds."""
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                return nframes / float(framerate) if framerate > 0 else 0.5
        except Exception:
            return 1.0
