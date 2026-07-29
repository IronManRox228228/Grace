import logging
import queue
import threading
import time
from typing import Optional, Callable, Any

logger = logging.getLogger("grace.wake_word")


class WakeWordDetector:
    """Vosk acoustic keyword spotter running in a background thread."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        keyword: str = "grace",
        threshold: float = 0.8,
        sample_rate: int = 16000,
        accumulate_bytes: int = 2560,  # ~80ms at 16kHz 16-bit mono
    ) -> None:
        self._keyword = keyword.lower()
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._model_path = model_path
        self._model: Optional[Any] = None
        self._rec: Optional[Any] = None
        self._accumulate_bytes = accumulate_bytes

        self._callback: Optional[Callable[[], None]] = None
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._event = threading.Event()
        self._last_detection_time = 0.0
        self._audio_buffer = bytearray()
        self._audio_queue: Optional[queue.Queue] = None

    @property
    def keyword(self) -> str:
        return self._keyword

    @keyword.setter
    def keyword(self, value: str) -> None:
        self._keyword = value.lower()

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_running(self) -> bool:
        return self._running and not self._paused

    @property
    def detected(self) -> bool:
        return self._event.is_set()

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Set the function to call when the keyword is detected."""
        self._callback = callback

    def _init_vosk(self) -> None:
        """Initialize Vosk acoustic model."""
        if self._rec is not None or not self._model_path:
            return

        try:
            from vosk import Model, KaldiRecognizer
            self._model = Model(self._model_path)
            self._rec = KaldiRecognizer(self._model, self._sample_rate)
            logger.info(f"Vosk wake word engine loaded from {self._model_path}")
        except Exception as exc:
            logger.error(f"Failed to load Vosk wake word model: {exc}")

    def _run_loop(self) -> None:
        """Background loop that reads audio frames and detects wake words via Vosk."""
        logger.info("Vosk wake word detector thread started")
        self._init_vosk()

        while self._running:
            if self._paused or self._audio_queue is None:
                time.sleep(0.01)
                continue

            try:
                frame = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                self._flush_audio_buffer()
                continue

            self._audio_buffer.extend(frame)
            if len(self._audio_buffer) >= self._accumulate_bytes:
                self._flush_audio_buffer()

        self._flush_audio_buffer()
        logger.info("Vosk wake word detector thread stopped")

    def _flush_audio_buffer(self) -> None:
        """Flush accumulated PCM audio to Vosk engine."""
        if not self._audio_buffer or self._rec is None:
            return

        data = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        try:
            if self._rec.AcceptWaveform(data):
                res = self._rec.Result()
                if res:
                    self._check_vosk_keyword(res)
            else:
                part = self._rec.PartialResult()
                if part:
                    self._check_vosk_keyword(part, is_partial=True)
        except Exception as e:
            logger.debug(f"Vosk AcceptWaveform error ignored: {e}")

    def _trigger_detection(self, label: str, score: float) -> None:
        """Trigger wake word activation event."""
        now = time.time()
        if now - self._last_detection_time > 1.0:
            logger.info(f"Wake word detected! label='{label}', score={score:.3f} (threshold={self._threshold})")
            self._last_detection_time = now
            self._event.set()
            if self._callback:
                self._callback()

    def _check_keyword(self, result_json: str, is_partial: bool = False) -> None:
        """Compatibility method for tests checking keyword parsing."""
        self._check_vosk_keyword(result_json, is_partial)

    def _check_vosk_keyword(self, result_json: str, is_partial: bool = False) -> None:
        """Check Vosk recognition output for target wake word."""
        import json
        try:
            data = json.loads(result_json)
            text = data.get("partial" if is_partial else "text", "").lower().strip()
        except Exception:
            return

        matched = (self._keyword in text) or ("grace" in text) or ("hey grace" in text)
        if matched:
            self._trigger_detection("vosk_" + text, 0.85)

    def start(self, audio_queue: Optional["queue.Queue"] = None) -> None:
        """Start Vosk wake word detection."""
        self._audio_queue = audio_queue
        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the wake word detector."""
        self._running = False
        self._paused = False
        self._event.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def pause(self) -> None:
        """Pause wake word detection without stopping the thread."""
        self._paused = True
        self._event.clear()

    def resume(self) -> None:
        """Resume wake word detection after pause."""
        self._paused = False
        self._event.clear()
        self._audio_buffer.clear()

    def wait_for_detection(self, timeout: Optional[float] = None) -> bool:
        """Wait until the keyword is detected or timeout."""
        detected = self._event.wait(timeout=timeout)
        if detected:
            self._event.clear()
        return detected

    def reset(self) -> None:
        """Reset detection event."""
        self._event.clear()
        self._audio_buffer.clear()
        if self._rec is not None:
            try:
                self._rec.Reset()
            except Exception:
                pass
