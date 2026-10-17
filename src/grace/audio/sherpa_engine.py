"""Sherpa-ONNX engines for Grace runtime slimming (PLAN.md §4, Task 08).

Implements ASR, TTS, VAD, and Wake Word (Keyword Spotting) behind Grace's existing
component interfaces using sherpa-onnx:
- ASR: Parakeet-TDT 0.6B int8 (or Moonshine Base int8 on the CPU floor tier).
- TTS: Kokoro-82M ONNX.
- VAD: Silero VAD ONNX.
- Wake Word: Keyword spotting (KWS).

Rule: These engines do not download model weights automatically. If a required model
file is missing, the code fails with a clear message directing to docs/MODELS.md.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("grace.sherpa")


def _require_sherpa():
    """Ensure sherpa_onnx is available or raise informative ImportError."""
    try:
        import sherpa_onnx
        return sherpa_onnx
    except ImportError as e:
        raise ImportError(
            "sherpa-onnx is required for sherpa audio engines. "
            "Install it via: pip install sherpa-onnx"
        ) from e


def _verify_model_file(path: str, description: str) -> str:
    """Verify that a model file or directory exists, or raise helpful FileNotFoundError."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required Sherpa-ONNX model for {description}: '{path}'. "
            f"Please download the weights as documented in docs/MODELS.md"
        )
    return path


class SherpaASR:
    """Offline speech-to-text using sherpa-onnx (Parakeet-TDT or Moonshine Base).

    Mirrors the WhisperStreaming interface:
      transcribe_bytes(raw_pcm: bytes) -> str
      reset_buffer() -> None
      warmup() -> None
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "parakeet",
        num_threads: int = 4,
        sample_rate: int = 16000,
    ):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.num_threads = num_threads
        self.sample_rate = sample_rate
        self._recognizer = None
        self._initialized = False

    def _initialize(self) -> None:
        if self._initialized:
            return
        _verify_model_file(self.model_path, f"ASR ({self.model_type})")
        sherpa = _require_sherpa()

        logger.info(f"Initializing Sherpa-ONNX ASR ({self.model_type}) from {self.model_path} with {self.num_threads} threads...")
        try:
            if self.model_type == "parakeet":
                # Nemo Parakeet TDT CTC int8
                model_file = os.path.join(self.model_path, "model.int8.onnx")
                tokens_file = os.path.join(self.model_path, "tokens.txt")
                if not os.path.exists(model_file):
                    model_file = os.path.join(self.model_path, "model.onnx")
                _verify_model_file(model_file, "Parakeet model.onnx")
                _verify_model_file(tokens_file, "Parakeet tokens.txt")

                self._recognizer = sherpa.OfflineRecognizer.from_nemo_ctc(
                    model=model_file,
                    tokens=tokens_file,
                    num_threads=self.num_threads,
                    sample_rate=self.sample_rate,
                )
            elif self.model_type == "moonshine":
                # Moonshine Base
                preprocessor = os.path.join(self.model_path, "preprocess.onnx")
                encoder = os.path.join(self.model_path, "encode.int8.onnx")
                uncached_decoder = os.path.join(self.model_path, "uncached_decode.int8.onnx")
                cached_decoder = os.path.join(self.model_path, "cached_decode.int8.onnx")
                tokens = os.path.join(self.model_path, "tokens.txt")
                _verify_model_file(preprocessor, "Moonshine preprocess.onnx")
                _verify_model_file(encoder, "Moonshine encode.onnx")
                _verify_model_file(tokens, "Moonshine tokens.txt")

                self._recognizer = sherpa.OfflineRecognizer.from_moonshine(
                    preprocessor=preprocessor,
                    encoder=encoder,
                    uncached_decoder=uncached_decoder,
                    cached_decoder=cached_decoder,
                    tokens=tokens,
                    num_threads=self.num_threads,
                )
            else:
                raise ValueError(f"Unknown Sherpa ASR model_type: '{self.model_type}'")
            self._initialized = True
            logger.info("Sherpa-ONNX ASR initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Sherpa-ONNX ASR: {e}")
            raise

    def transcribe_bytes(self, raw_pcm: bytes) -> str:
        """Transcribe 16-bit 16kHz mono PCM audio bytes into text."""
        if not raw_pcm:
            return ""
        self._initialize()
        import numpy as np

        samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, samples)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def reset_buffer(self) -> None:
        pass

    def warmup(self) -> None:
        """Run 1 second of silence to pre-warm the recognizer."""
        try:
            self.transcribe_bytes(bytes(32000))
        except Exception as e:
            logger.debug(f"Sherpa ASR warmup notice: {e}")


class SherpaTTS:
    """Text-to-speech using sherpa-onnx Kokoro-82M ONNX model.

    Mirrors the TTS player/generator interface:
      generate_and_speak(text: str) -> None
      speak(text: str) -> None
      stop_speaking() -> None
      shutdown() -> None
    """

    def __init__(
        self,
        model_path: str,
        voices_path: Optional[str] = None,
        num_threads: int = 4,
        speed: float = 1.0,
    ):
        self.model_path = model_path
        self.voices_path = voices_path
        self.num_threads = num_threads
        self.speed = speed
        self._tts = None
        self._initialized = False
        self._stop_event = threading.Event()

    def _initialize(self) -> None:
        if self._initialized:
            return
        _verify_model_file(self.model_path, "Kokoro ONNX directory")
        sherpa = _require_sherpa()

        model_file = os.path.join(self.model_path, "model.onnx")
        voices_bin = self.voices_path or os.path.join(self.model_path, "voices.bin")
        tokens = os.path.join(self.model_path, "tokens.txt")
        data_dir = os.path.join(self.model_path, "espeak-ng-data")

        _verify_model_file(model_file, "Kokoro model.onnx")
        _verify_model_file(voices_bin, "Kokoro voices.bin")
        _verify_model_file(tokens, "Kokoro tokens.txt")

        logger.info(f"Initializing Sherpa-ONNX Kokoro TTS from {model_file}...")
        self._tts = sherpa.OfflineTts.from_kokoro(
            model=model_file,
            voices=voices_bin,
            tokens=tokens,
            data_dir=data_dir if os.path.exists(data_dir) else "",
            num_threads=self.num_threads,
        )
        self._initialized = True
        logger.info("Sherpa-ONNX Kokoro TTS initialized successfully.")

    def generate_and_speak(self, text: str) -> None:
        """Generate audio for text and play via sounddevice."""
        if not text or not text.strip():
            return
        self._initialize()
        self._stop_event.clear()

        import sounddevice as sd

        audio = self._tts.generate(text.strip(), sid=0, speed=self.speed)
        if audio and not self._stop_event.is_set():
            sd.play(audio.samples, samplerate=audio.sample_rate)
            while sd.get_stream() and sd.get_stream().active:
                if self._stop_event.is_set():
                    sd.stop()
                    break
                time.sleep(0.05)

    def speak(self, text: str) -> None:
        self.generate_and_speak(text)

    def stop_speaking(self) -> None:
        self._stop_event.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def shutdown(self) -> None:
        self.stop_speaking()


class SherpaVAD:
    """Voice activity detection using sherpa-onnx Silero VAD ONNX model.

    Mirrors SileroVAD detector interface:
      process_chunk(chunk: bytes, capture: Any) -> bool
      reset() -> None
    """

    def __init__(self, model_path: str, threshold: float = 0.5, sample_rate: int = 16000):
        self.model_path = model_path
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._vad = None
        self._initialized = False

    def _initialize(self) -> None:
        if self._initialized:
            return
        _verify_model_file(self.model_path, "Silero VAD ONNX model")
        sherpa = _require_sherpa()

        config = sherpa.VadModelConfig()
        config.silero_vad.model = self.model_path
        config.silero_vad.threshold = self.threshold
        config.sample_rate = self.sample_rate
        self._vad = sherpa.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        self._initialized = True

    def process_chunk(self, chunk: bytes, capture: Any = None) -> bool:
        """Feed a PCM chunk. Returns True if a speech boundary has been detected."""
        if not chunk:
            return False
        self._initialize()
        import numpy as np

        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self._vad.accept_waveform(samples)
        if not self._vad.is_speech_detected():
            return False
        return not self._vad.empty()

    def reset(self) -> None:
        if self._vad:
            self._vad.clear()


class SherpaKWS:
    """Offline keyword spotting (wake word) using sherpa-onnx Zipformer KWS.

    Mirrors WakeWordDetector interface:
      detected (property: bool)
      reset() -> None
      pause() -> None
      resume() -> None
      arm_cancel_watch(callback) -> None
      disarm_cancel_watch() -> None
      stop() -> None
    """

    def __init__(self, model_path: str, keywords_file: str, num_threads: int = 2):
        self.model_path = model_path
        self.keywords_file = keywords_file
        self.num_threads = num_threads
        self._spotter = None
        self._initialized = False
        self._paused = False
        self._detected = False
        self._cancel_callback = None

    def _initialize(self) -> None:
        if self._initialized:
            return
        _verify_model_file(self.model_path, "Sherpa KWS model directory")
        _verify_model_file(self.keywords_file, "Sherpa KWS keywords.txt")
        sherpa = _require_sherpa()

        encoder = os.path.join(self.model_path, "encoder.onnx")
        decoder = os.path.join(self.model_path, "decoder.onnx")
        joiner = os.path.join(self.model_path, "joiner.onnx")
        tokens = os.path.join(self.model_path, "tokens.txt")

        _verify_model_file(encoder, "KWS encoder.onnx")
        _verify_model_file(decoder, "KWS decoder.onnx")
        _verify_model_file(joiner, "KWS joiner.onnx")
        _verify_model_file(tokens, "KWS tokens.txt")

        self._spotter = sherpa.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=self.keywords_file,
            num_threads=self.num_threads,
        )
        self._stream = self._spotter.create_stream()
        self._initialized = True

    @property
    def detected(self) -> bool:
        return self._detected

    def reset(self) -> None:
        self._detected = False
        if self._spotter:
            self._stream = self._spotter.create_stream()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def arm_cancel_watch(self, callback: Callable[[], None]) -> None:
        self._cancel_callback = callback

    def disarm_cancel_watch(self) -> None:
        self._cancel_callback = None

    def stop(self) -> None:
        self.pause()
        self.reset()
