import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from grace.audio.capture import AudioCapture

logger = logging.getLogger("grace.vad")


@dataclass
class SilenceState:
    """Tracks voice activity state."""
    is_silent: bool = True
    silence_start: float = field(default_factory=time.time)
    last_speech_time: float = 0.0
    total_silence_ms: float = 0.0
    speech_chunks: list[bytes] = None  # type: ignore[assignment]


class VadDetector:
    """Voice Activity Detection using energy thresholding.

    Monitors audio chunks and tracks whether the user is speaking.
    Emits a callback when silence is detected after speech (turn end).

    Two clocks
    ----------
    By default silence accumulates against ``time.time()``, which is what a live
    microphone wants: the wall is the only clock a dropped chunk cannot hide
    from.

    Pass ``sample_rate`` to accumulate against *audio* time instead - the
    duration of the samples actually handed to :meth:`process_chunk`, which
    advances by exactly ``len(chunk) / (rate * width * channels)`` seconds per
    chunk regardless of how fast they arrive. The tape harness uses this, and it
    is the difference between a deterministic oracle and a coin flip: replay
    feeds a recorded chunk sequence as fast as the scheduler allows, so under a
    wall clock the ~300ms silence tail of a generated turn can land inside a
    275ms window and the audio runs out before the turn closes. That surfaced as
    "recorded audio ended before the VAD closed the turn" on roughly 2 of 27
    tapes, varying run to run on an unchanged corpus.

    Audio time is arguably the more correct clock for the live path too - it
    cannot truncate a speaker because the capture thread was descheduled - but
    that is a behavioural change to the listening window and is deliberately not
    made here.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        silence_duration_ms: int = 1200,
        sample_rate: Optional[int] = None,
        sample_width: int = 2,
        channels: int = 1,
    ) -> None:
        self._threshold = threshold
        self._silence_duration_ms = silence_duration_ms
        self._bytes_per_second: Optional[float] = (
            float(sample_rate * sample_width * channels) if sample_rate else None
        )
        self._audio_seconds = 0.0
        self._state = SilenceState()
        if self._bytes_per_second:
            self._state.silence_start = 0.0
        self._on_silence: Optional[Callable[[SilenceState], None]] = None
        self._on_speech: Optional[Callable[[SilenceState], None]] = None
        self._has_detected_speech = False

    @property
    def uses_audio_clock(self) -> bool:
        return self._bytes_per_second is not None

    def _now(self) -> float:
        """Seconds elapsed, on whichever clock this detector was built with."""
        if self._bytes_per_second is not None:
            return self._audio_seconds
        return time.time()

    @property
    def is_speaking(self) -> bool:
        return not self._state.is_silent

    @property
    def has_detected_speech(self) -> bool:
        return self._has_detected_speech

    def set_silence_callback(self, callback: Callable[[SilenceState], None]) -> None:
        self._on_silence = callback

    def set_speech_callback(self, callback: Callable[[SilenceState], None]) -> None:
        self._on_speech = callback

    def process_chunk(self, chunk: bytes, audio: Optional["AudioCapture"] = None) -> bool:
        """Process a single audio chunk.

        Returns True if a silence turn-end was detected.
        """
        import math

        # Calculate RMS energy
        if audio:
            rms = audio.get_rms(chunk)
            # Normalize: typical RMS for 16-bit audio ranges 0-32767
            # Threshold is relative to max possible (32767)
            normalized_rms = rms / 32767.0
        else:
            if len(chunk) < 2 or len(chunk) % 2 != 0:
                normalized_rms = 0.0
            else:
                import struct
                samples = struct.unpack(f"<{len(chunk) // 2}h", chunk)
                normalized_rms = (sum(s * s for s in samples) / len(samples)) ** 0.5 / 32767.0

        # Advanced before the chunk is judged, so a chunk's own duration counts
        # towards the silence it is part of - the same way wall-clock time has
        # already passed by the time a live chunk is handed over.
        if self._bytes_per_second is not None:
            self._audio_seconds += len(chunk) / self._bytes_per_second

        now = self._now()

        if normalized_rms >= self._threshold:
            # Speech detected
            self._has_detected_speech = True
            if self._state.is_silent:
                logger.debug(f"RMS={normalized_rms:.4f} (threshold={self._threshold}) — speech STARTED")
                self._state.is_silent = False
                self._state.last_speech_time = now
                self._state.speech_chunks = []
                if self._on_speech:
                    self._on_speech(self._state)
        else:
            # Silent
            if not self._state.is_silent:
                logger.debug(f"RMS={normalized_rms:.4f} — silence STARTED (was speaking)")
                self._state.is_silent = True
                self._state.silence_start = now
                self._state.total_silence_ms = 0.0

        # Track silence duration (only matters if speech was detected before)
        if self._state.is_silent and self._has_detected_speech:
            self._state.total_silence_ms = (now - self._state.silence_start) * 1000

            # Check if silence duration threshold exceeded
            if self._state.total_silence_ms >= self._silence_duration_ms:
                logger.info(f"VAD: turn-end detected ({self._state.total_silence_ms:.0f}ms silence after speech)")
                if self._on_silence:
                    self._on_silence(self._state)
                return True

        return False

    def reset(self) -> None:
        """Reset detection state.

        The audio clock keeps running across turns. It measures elapsed audio,
        not elapsed turn, and rewinding it would make the second turn of a
        session start from a silence window the first turn had already filled.
        """
        self._state = SilenceState()
        if self._bytes_per_second:
            self._state.silence_start = self._audio_seconds
        self._has_detected_speech = False
