import concurrent.futures
import io
import logging
import os
import queue
import threading
import warnings
from typing import Optional, Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

from grace.text.sentence_split import split_sentences

logger = logging.getLogger("grace.kokoro")


class KokoroWorker:
    """A single Kokoro TTS worker on GPU with its own KPipeline instance running on a dedicated thread."""

    def __init__(
        self,
        model_path: str,
        voices_path: str,
        device: str = "cuda",
        dtype: str = "float16",
    ):
        self._model_path = model_path
        self._voices_path = voices_path
        self._device = device
        self._dtype = dtype
        self._pipeline = None
        self._voice = None
        self._worker_id = id(self)
        self._initialized = False
        self._lock = threading.Lock()
        self._init_error: Optional[str] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"KokoroWorker-{self._worker_id}"
        )

    def _initialize(self) -> Optional[str]:
        """Initialize the worker on its dedicated thread. Returns error message on failure, None on success."""
        future = self._executor.submit(self._do_initialize)
        try:
            return future.result(timeout=60)
        except Exception as e:
            err = f"Kokoro worker init failed/timed out: {e}"
            logger.error(err)
            return err

    def _do_initialize(self) -> Optional[str]:
        if self._initialized:
            return None

        try:
            import torch
            from kokoro import KPipeline

            device = self._device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU for Kokoro (will fail init)")
                device = "cpu"

            logger.info(f"Loading KPipeline on {device} for worker {self._worker_id} in dedicated thread...")
            self._pipeline = KPipeline(lang_code="a", device=device)

            if os.path.exists(self._voices_path):
                self._voice = torch.load(self._voices_path, weights_only=True, map_location=device)
                if hasattr(self._pipeline, "voices"):
                    self._pipeline.voices[os.path.basename(self._voices_path).replace(".pt", "")] = self._voice
                    self._pipeline.voices["af_bella"] = self._voice
                logger.info(f"Loaded Kokoro voice tensor from {self._voices_path}")

            # Fail if we fell back to CPU - latency budget violation
            if device == "cpu":
                error_msg = f"Kokoro worker initialized on CPU (requested CUDA). Latency budget exceeded."
                logger.error(error_msg)
                self._init_error = error_msg
                self._pipeline = None
                self._initialized = False
                return error_msg

            self._device = device
            self._initialized = True
            logger.info(f"Kokoro worker initialized on {device}")
            return None
        except ImportError as e:
            error_msg = f"kokoro package not installed: {e}"
            logger.error(error_msg)
            self._init_error = error_msg
            self._initialized = False
            return error_msg
        except Exception as e:
            error_msg = f"Failed to initialize Kokoro worker: {e}"
            logger.error(error_msg)
            self._init_error = error_msg
            self._initialized = False
            return error_msg

    def synthesize(self, text: str, voice_name: str = "af_bella") -> Optional[bytes]:
        """Synthesize speech for the given text on the dedicated thread."""
        future = self._executor.submit(self._do_synthesize, text, voice_name)
        try:
            return future.result(timeout=30)
        except Exception as e:
            logger.error(f"Kokoro synthesis execution error: {e}")
            return None

    def _do_synthesize(self, text: str, voice_name: str = "af_bella") -> Optional[bytes]:
        if not self._initialized:
            err = self._do_initialize()
            if err:
                return None
        if not self._pipeline:
            logger.error("Kokoro pipeline not loaded")
            return None

        try:
            voice = voice_name if voice_name else "af_bella"
            logger.debug(f"Entering KPipeline execution for '{text[:40]}'")
            audio_chunks = []
            for _gs, _ps, audio in self._pipeline(text, voice=voice):
                if audio is None:
                    continue
                if hasattr(audio, "cpu"):
                    audio = audio.cpu()
                if hasattr(audio, "numpy"):
                    audio_chunks.append(audio.numpy())
                else:
                    audio_chunks.append(np.array(audio))

            if not audio_chunks:
                logger.warning(f"Kokoro returned no audio for: '{text[:30]}'")
                return None

            audio_np = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
            logger.debug(f"KPipeline execution finished for '{text[:40]}'")
            return self._to_wav(audio_np)
        except Exception as e:
            logger.error(f"Kokoro synthesis failed: {e}")
            return None

    def _to_wav(self, audio) -> bytes:
        """Convert audio tensor/array to WAV bytes."""
        if hasattr(audio, "numpy"):
            audio_np = audio.numpy()
        else:
            audio_np = np.array(audio)

        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        audio_int16 = (audio_np * 32767).astype(np.int16)

        with io.BytesIO() as wav_buf:
            import wave
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_int16.tobytes())
            return wav_buf.getvalue()


class KokoroEngine:
    """Kokoro TTS engine with round-robin worker pool, each with dedicated KPipeline."""

    def __init__(
        self,
        model_path: str,
        voices_path: str,
        num_workers: int = 3,
        device: str = "cuda",
        dtype: str = "float16",
    ):
        self._num_workers = num_workers
        self._device = device
        self._dtype = dtype
        self._model_path = model_path
        self._voices_path = voices_path
        self._current_idx = 0
        self._workers = []
        self._results_queue = queue.Queue()
        self._lock = threading.Lock()
        self._initialized = False
        self._init_errors: list[str] = []

    def initialize(self):
        """Initialize all workers. Collects failures and raises if all workers failed."""
        if self._initialized:
            return

        self._init_errors = []
        successful_workers = 0

        for i in range(self._num_workers):
            worker = KokoroWorker(
                self._model_path,
                self._voices_path,
                self._device,
                self._dtype,
            )
            error = worker._initialize()
            if error:
                self._init_errors.append(f"Worker {i}: {error}")
            else:
                self._workers.append(worker)
                successful_workers += 1
                logger.info(f"Kokoro worker {i} created on {worker._device}")

        # If zero workers requested, that's a valid (if useless) config - don't raise
        if self._num_workers == 0:
            logger.info("KokoroEngine initialized with 0 workers (no TTS available)")
            self._initialized = True
            return

        if successful_workers == 0:
            error_summary = "; ".join(self._init_errors) if self._init_errors else "No workers initialized"
            raise RuntimeError(f"KokoroEngine initialization failed: {error_summary}")

        if self._init_errors:
            # Some workers failed - log warnings but continue with working ones
            logger.warning(f"KokoroEngine: {len(self._init_errors)} worker(s) failed: {'; '.join(self._init_errors)}")

        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def synthesize(self, text: str, voice: str = "af_bella") -> Optional[bytes]:
        """Synthesize speech for the given text using round-robin worker selection."""
        if not self._initialized:
            self.initialize()

        if not self._workers:
            logger.error("No Kokoro workers available")
            return None

        idx = self._current_idx % len(self._workers)
        self._current_idx += 1

        worker = self._workers[idx]
        return worker.synthesize(text, voice)

    def synthesize_sentences(
        self,
        text: str,
        voice: str = "af_bella",
    ) -> list[bytes]:
        """Split text into sentences and synthesize each one."""
        sentences = self._split_sentences(text)
        results = []
        for sentence in sentences:
            wav = self.synthesize(sentence, voice)
            if wav:
                results.append(wav)
        return results

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        return split_sentences(text)