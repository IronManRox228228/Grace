"""A stand-in microphone that plays WAV files dropped into a folder.

Dev-only, for simulated voice runs: set GRACE_FAKE_MIC to a folder and Grace
hears silence until a .wav appears there, then hears that file at real-time
pace, exactly as the real mic would deliver it (wake word, VAD and Whisper all
run unchanged). Files are played oldest name first and moved to `played/`.
Each file is preceded by `lead_in_s` of silence so the listener is ready
before speech starts.
Files must be 16-bit PCM, mono, at the configured sample rate.
"""

import logging
import os
import shutil
import time
import wave
from typing import Optional

from grace.audio.capture import AudioCapture

logger = logging.getLogger("grace.audio.fake_mic")


class FakeMicCapture(AudioCapture):
    """AudioCapture with the device swapped for a folder of WAV files."""

    def __init__(self, folder: str, sample_rate: int = 16000, chunk_size: int = 512,
                 channels: int = 1, width: int = 2, lead_in_s: float = 1.0) -> None:
        super().__init__(sample_rate=sample_rate, chunk_size=chunk_size,
                         channels=channels, width=width)
        self._folder = folder
        self._lead_in = b"\x00" * (int(sample_rate * lead_in_s) * width * channels)
        self._pending = b""
        self._next_at = 0.0

    def list_devices(self) -> list[dict]:
        return [{"index": 0, "name": f"Fake mic ({self._folder})"}]

    def start(self) -> "FakeMicCapture":
        os.makedirs(os.path.join(self._folder, "played"), exist_ok=True)
        logger.info(f"Audio capture: FAKE MIC reading WAV files from {self._folder}")
        self._running = True
        self._next_at = time.monotonic()
        return self

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._running = False

    def _load_next(self) -> Optional[bytes]:
        try:
            names = sorted(n for n in os.listdir(self._folder) if n.lower().endswith(".wav"))
        except OSError:
            return None
        for name in names:
            path = os.path.join(self._folder, name)
            try:
                with wave.open(path, "rb") as w:
                    ok = (w.getframerate() == self._sample_rate and w.getnchannels() == self._channels
                          and w.getsampwidth() == self._width)
                    data = w.readframes(w.getnframes()) if ok else None
            except (OSError, EOFError, wave.Error):
                continue  # still being written; try again next chunk
            shutil.move(path, os.path.join(self._folder, "played", name))
            if data is None:
                logger.error(f"Fake mic skipped {name}: needs {self._sample_rate} Hz, "
                             f"{self._channels} channel(s), {self._width * 8}-bit")
                continue
            logger.info(f"Fake mic playing {name} ({len(data) / (self._sample_rate * self._width):.1f}s)")
            return self._lead_in + data
        return None

    def get_chunk(self) -> bytes:
        if not self._running:
            raise RuntimeError("Audio capture is not running. Call start() first.")
        # Pace reads like a real device: one chunk per chunk-duration.
        self._next_at += self._chunk_size / self._sample_rate
        delay = self._next_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_at = time.monotonic()

        size = self._chunk_size * self._width * self._channels
        if not self._pending:
            self._pending = self._load_next() or b""
        chunk, self._pending = self._pending[:size], self._pending[size:]
        return chunk.ljust(size, b"\x00")
