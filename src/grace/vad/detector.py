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
    """

    def __init__(
        self,
        threshold: float = 0.5,
        silence_duration_ms: int = 1200,
    ) -> None:
        self._threshold = threshold
        self._silence_duration_ms = silence_duration_ms
        self._state = SilenceState()
        self._on_silence: Optional[Callable[[SilenceState], None]] = None
        self._on_speech: Optional[Callable[[SilenceState], None]] = None
        self._has_detected_speech = False

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

        now = time.time()

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
        """Reset detection state."""
        self._state = SilenceState()
        self._has_detected_speech = False
