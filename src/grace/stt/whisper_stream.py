"""Streaming speech recognition using faster-whisper.

Accumulates audio chunks and transcribes them when requested.
Uses CUDA inference on RTX 4060 via faster-whisper.
"""

import logging
import os
from typing import Optional

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np

logger = logging.getLogger("grace.whisper")


class WhisperStreaming:
    """Whisper speech-to-text engine.

    Accumulates audio chunks and transcribes them on demand.
    Supports CUDA inference via faster-whisper.
    """

    def __init__(self, model_path: str, device: str = "cuda", compute_type: str = "float16"):
        self._model_path = model_path
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._buffer = bytearray()
        self._last_transcript = ""
        self._initialized = False
        # Whisper always resamples to 16 kHz internally; used only to size the
        # one second of silence fed to warmup().
        self._sample_rate_hint = 16000

    def _initialize(self):
        if self._initialized:
            return
        from faster_whisper import WhisperModel
        import torch

        device = self._device
        compute_type = self._compute_type
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU for Whisper")
            device = "cpu"
            compute_type = "int8"

        self._model = WhisperModel(
            self._model_path,
            device=device,
            compute_type=compute_type,
            cpu_threads=4,
        )
        self._device = device
        self._initialized = True

    def warmup(self) -> bool:
        """Load the model and run one throwaway inference.

        Without this the user's first utterance pays the full model load plus
        the first-call CUDA kernel autotune, which is seconds of dead air.
        Safe to call from a background thread; failures are non-fatal because
        transcribe() will initialise lazily anyway.
        """
        try:
            self._initialize()
            silence = np.zeros(self._sample_rate_hint, dtype=np.float32)
            segments, _info = self._model.transcribe(silence, language="en", beam_size=1, vad_filter=False)
            # faster-whisper is lazy: the generator must be drained to do work.
            for _ in segments:
                pass
            logger.info(f"Whisper warmed up on {self._device}")
            return True
        except Exception as e:
            logger.warning(f"Whisper warmup failed: {e}")
            return False

    def add_chunk(self, chunk: bytes):
        """Add an audio chunk to the buffer."""
        self._buffer.extend(chunk)

    def add_buffer(self, buffer: bytearray):
        """Add a pre-built buffer to the audio buffer."""
        self._buffer.extend(buffer)

    def get_buffer(self) -> bytes:
        """Get the current audio buffer contents."""
        return bytes(self._buffer)

    def clear_buffer(self):
        """Clear the audio buffer."""
        self._buffer.clear()

    def reset_buffer(self):
        """Clear the buffer and reset state."""
        self._buffer.clear()
        self._last_transcript = ""

    def _transcribe_pcm(self, raw_bytes: bytes) -> str:
        """Convert int16 PCM bytes to float32 audio and run Whisper inference."""
        if not self._initialized:
            self._initialize()

        buf = raw_bytes
        if len(buf) % 2 != 0:
            buf = buf[:-1]
        audio_array = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _info = self._model.transcribe(
            audio_array,
            language="en",
            beam_size=2,
            vad_filter=False,
        )
        return "".join(segment.text for segment in segments).strip()

    def transcribe(self) -> str:
        """Transcribe the accumulated audio buffer.

        Returns the transcript text. If no buffer, returns empty string.
        """
        if not self._buffer:
            return ""

        self._last_transcript = self._transcribe_pcm(bytes(self._buffer))
        self.clear_buffer()
        return self._last_transcript

    def transcribe_bytes(self, raw_bytes: bytes) -> str:
        """Transcribe a specific unfragmented raw PCM audio buffer directly."""
        if not raw_bytes:
            return ""
        return self._transcribe_pcm(raw_bytes)

    @property
    def last_transcript(self) -> str:
        return self._last_transcript
