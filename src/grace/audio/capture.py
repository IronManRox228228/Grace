import logging
from typing import Optional

import numpy as np
import pyaudio

logger = logging.getLogger("grace.audio")


class AudioCapture:
    """Microphone capture using pyaudio.

    Opens a microphone stream and yields audio chunks on demand.
    Provides a get() method for pulling chunks into the wake word detector
    and Whisper streaming pipeline.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        channels: int = 1,
        width: int = 2,
        device_index: int = -1,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._channels = channels
        self._width = width
        self._device_index = device_index
        self._stream: Optional[pyaudio.Stream] = None
        self._running = False
        self._pyaudio: Optional[pyaudio.PyAudio] = None

    @property
    def is_running(self) -> bool:
        return self._running

    def get_device_count(self) -> int:
        """Return the number of available audio devices."""
        if not self._pyaudio:
            self._init_pyaudio()
        return self._pyaudio.get_device_count()  # type: ignore[union-attr]

    def get_device_name(self, index: int) -> str:
        """Return the name of an audio device by index."""
        if not self._pyaudio:
            self._init_pyaudio()
        device_info = self._pyaudio.get_device_info_by_index(index)  # type: ignore[union-attr]
        return device_info.get("name", f"Device {index}")

    def list_devices(self) -> list[dict]:
        """List all available audio devices."""
        if not self._pyaudio:
            self._init_pyaudio()
        devices = []
        for i in range(self._pyaudio.get_device_count()):  # type: ignore[union-attr]
            info = self._pyaudio.get_device_info_by_index(i)  # type: ignore[union-attr]
            devices.append({
                "index": i,
                "name": info.get("name", "Unknown"),
                "max_input_channels": info.get("maxInputChannels", 0),
                "default_sample_rate": info.get("defaultSampleRate", 0.0),
            })
        return devices

    def _init_pyaudio(self) -> None:
        """Initialize pyaudio."""
        self._pyaudio = pyaudio.PyAudio()

    def start(self) -> "AudioCapture":
        """Start the microphone stream.

        Returns self for chaining.
        """
        if self._pyaudio is None:
            self._init_pyaudio()

        # Resolve device index
        device_index = self._device_index
        if device_index == -1:
            # Use default input device
            device_index = self._pyaudio.get_default_input_device_info()["index"]  # type: ignore[union-attr]

        device_info = self._pyaudio.get_device_info_by_index(device_index)
        device_name = device_info.get("name", f"Device {device_index}")
        logger.info(
            f"Audio capture: device='{device_name}' (index={device_index}), "
            f"rate={self._sample_rate}, chunk={self._chunk_size}, "
            f"channels={self._channels}, width={self._width}"
        )

        self._stream = self._pyaudio.open(  # type: ignore[union-attr]
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            input_device_index=device_index,
        )
        self._running = True
        return self

    def stop(self) -> None:
        """Stop and close the microphone stream."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        self._running = False

    def get_chunk(self) -> bytes:
        """Read one chunk of audio from the microphone.

        Returns the raw PCM audio bytes.
        """
        if not self._running:
            raise RuntimeError("Audio capture is not running. Call start() first.")
        if not self._stream:
            return b"\x00" * (self._chunk_size * self._width)
        try:
            if hasattr(self._stream, "is_active") and not self._stream.is_active():
                logger.warning("PyAudio stream is inactive, attempting restart...")
                try:
                    self._stream.start_stream()
                except Exception:
                    pass
            return self._stream.read(self._chunk_size, exception_on_overflow=False)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Audio read error ignored: {e}")
            return b"\x00" * (self._chunk_size * self._width)

    def get_chunk_as_int16(self) -> list[int]:
        """Read one chunk and return as list of int16 values."""
        import struct

        raw = self.get_chunk()
        return list(struct.unpack(f"<{len(raw) // 2}h", raw))

    def get_rms(self, chunk: bytes) -> float:
        """Calculate RMS energy of a PCM chunk for VAD.

        Runs ~31x/second on every captured chunk, so it uses numpy rather than
        a Python-level sum over unpacked samples. An odd byte count still
        raises, matching struct.unpack's behaviour.
        """
        if len(chunk) % 2 != 0:
            raise ValueError(f"PCM16 chunk has an odd byte count: {len(chunk)}")
        samples = np.frombuffer(chunk, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

    def close(self) -> None:
        """Fully clean up resources."""
        self.stop()
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
