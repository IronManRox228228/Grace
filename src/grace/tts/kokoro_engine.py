"""Kokoro TTS engine.

One KModel is loaded and shared by every worker. Upstream is explicit about
this (``kokoro/pipeline.py``: "If you have multiple KPipelines, you should reuse
one KModel instance across all of them") - the previous implementation gave each
worker its own ``KPipeline()`` with no model argument, so every worker built a
full independent 82M-parameter model. Workers exist for concurrency, not for
extra copies of the weights: what is genuinely per-worker is the misaki/espeak
G2P frontend, which is CPU-bound and benefits from running in parallel.

Measured on this machine (RTX, CUDA):
  * model load from local files : ~1.5 s   (~6.7 s when resolved via HF hub)
  * first synthesis (cold)      : ~1.1 s   (cuDNN autotune, iSTFT init, G2P)
  * subsequent synthesis        : ~70-100 ms
Hence warmup: it moves that one-off ~1 s off the user's first spoken sentence.
"""

import concurrent.futures
import io
import logging
import os
import threading
import warnings
from collections import OrderedDict
from typing import Optional, Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

from grace.text.sentence_split import split_sentences

logger = logging.getLogger("grace.kokoro")

KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000

# The shared model and voice pack, created once per process. Guarded by a lock
# because workers initialize concurrently and would otherwise race to build it.
_shared_lock = threading.Lock()
_shared_model: Optional[Any] = None
_shared_model_key: Optional[tuple] = None
_shared_voices: dict[str, Any] = {}


def _resolve_local_model_files(model_path: str) -> tuple[Optional[str], Optional[str]]:
    """Return (weights_path, config_path) if the configured weights exist locally.

    Passing explicit paths to KModel skips HuggingFace hub resolution entirely,
    which measured ~1.5 s versus ~6.7 s. Returning (None, None) lets KModel fall
    back to the hub cache, which is what happens when KOKORO_MODEL_PATH is
    unset or stale.
    """
    if not model_path or not os.path.isfile(model_path):
        return None, None
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    if not os.path.isfile(config_path):
        return model_path, None
    return model_path, config_path


def _resolve_dtype(dtype: str) -> str:
    """Normalise the requested dtype, warning when it cannot be honoured.

    kokoro 0.9.4 generates float32 tensors internally regardless of the module
    dtype, so a half-precision model raises "Input and parameter tensors are not
    the same dtype" on the first forward pass. Rather than accept the setting
    and silently ignore it (which is what this code used to do), say so.
    """
    requested = (dtype or "float32").lower()
    if requested in ("float16", "fp16", "half"):
        logger.warning(
            "KOKORO_DTYPE=float16 is not supported by kokoro %s - its internal "
            "tensors stay float32 and half precision fails at the first forward "
            "pass. Using float32.", _kokoro_version()
        )
        return "float32"
    return "float32"


def _kokoro_version() -> str:
    try:
        import kokoro
        return getattr(kokoro, "__version__", "unknown")
    except Exception:
        return "unknown"


def get_shared_model(model_path: str, device: str, dtype: str):
    """Load the KModel once per (device, dtype) and reuse it for every worker."""
    global _shared_model, _shared_model_key

    key = (device, dtype)
    with _shared_lock:
        if _shared_model is not None and _shared_model_key == key:
            logger.debug("Reusing shared KModel for %s", key)
            return _shared_model

        from kokoro import KModel

        weights, config = _resolve_local_model_files(model_path)
        if weights:
            logger.info(f"Loading shared KModel from local weights: {weights}")
        else:
            logger.info("Loading shared KModel from HuggingFace cache")

        model = KModel(repo_id=KOKORO_REPO_ID, config=config, model=weights)
        model = model.to(device).eval()

        _shared_model = model
        _shared_model_key = key
        return model


def get_shared_voice(voices_path: str, device: str):
    """Load a voice pack once and reuse the tensor across all pipelines."""
    if not voices_path or not os.path.isfile(voices_path):
        return None

    with _shared_lock:
        cached = _shared_voices.get(voices_path)
        if cached is not None:
            return cached

        import torch

        voice = torch.load(voices_path, weights_only=True, map_location=device)
        _shared_voices[voices_path] = voice
        logger.info(f"Loaded Kokoro voice tensor from {voices_path}")
        return voice


def _reset_shared_state() -> None:
    """Drop the shared model and voice cache. For tests."""
    global _shared_model, _shared_model_key
    with _shared_lock:
        _shared_model = None
        _shared_model_key = None
        _shared_voices.clear()


class KokoroWorker:
    """One Kokoro pipeline pinned to a dedicated thread.

    The pipeline is lightweight - it holds the G2P frontend and the voice table
    and delegates to the shared KModel for the actual forward pass.
    """

    def __init__(
        self,
        model_path: str,
        voices_path: str,
        device: str = "cuda",
        dtype: str = "float32",
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
            return future.result(timeout=120)
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

            # Fail before doing any expensive work - CPU synthesis blows the
            # latency budget and a half-working TTS is worse than a clear error.
            if device == "cpu":
                error_msg = f"Kokoro worker initialized on CPU (requested CUDA). Latency budget exceeded."
                logger.error(error_msg)
                self._init_error = error_msg
                self._pipeline = None
                self._initialized = False
                return error_msg

            dtype = _resolve_dtype(self._dtype)
            model = get_shared_model(self._model_path, device, dtype)

            logger.info(f"Building KPipeline on {device} for worker {self._worker_id} (shared KModel)...")
            self._pipeline = KPipeline(
                lang_code="a",
                repo_id=KOKORO_REPO_ID,
                model=model,
            )

            self._voice = get_shared_voice(self._voices_path, device)
            if self._voice is not None and hasattr(self._pipeline, "voices"):
                # Pre-seed the voice table under both names so load_voice never
                # reaches for the hub mid-synthesis.
                name = os.path.basename(self._voices_path).replace(".pt", "")
                self._pipeline.voices[name] = self._voice
                self._pipeline.voices["af_bella"] = self._voice

            self._device = device
            self._dtype = dtype
            self._initialized = True
            logger.info(f"Kokoro worker initialized on {device} ({dtype})")
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

    def submit(self, text: str, voice_name: str = "af_bella") -> concurrent.futures.Future:
        """Queue a synthesis without waiting, so callers can pipeline sentences."""
        return self._executor.submit(self._do_synthesize, text, voice_name)

    def warmup(self, text: str = "Ready.") -> bool:
        """Run one throwaway synthesis to pay the cold-start cost up front.

        The first call performs cuDNN autotuning, iSTFT setup and the first G2P
        pass - about 1 s on this machine. Doing it at startup rather than on the
        user's first sentence is the whole point.
        """
        if not self._initialized or self._pipeline is None:
            return False
        try:
            wav = self.synthesize(text)
            return bool(wav)
        except Exception as e:
            logger.debug(f"Kokoro warmup skipped: {e}")
            return False

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
            audio_chunks = []
            for _gs, _ps, audio in self._pipeline(text, voice=voice):
                if audio is None:
                    continue
                if hasattr(audio, "cpu"):
                    audio = audio.cpu()
                if hasattr(audio, "float"):
                    audio = audio.float()
                if hasattr(audio, "numpy"):
                    audio_chunks.append(audio.numpy())
                else:
                    audio_chunks.append(np.array(audio))

            if not audio_chunks:
                logger.warning(f"Kokoro returned no audio for: '{text[:30]}'")
                return None

            audio_np = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
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
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())
            return wav_buf.getvalue()

    def shutdown(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass


class KokoroEngine:
    """Pool of Kokoro pipelines sharing one KModel, with an LRU phrase cache."""

    def __init__(
        self,
        model_path: str,
        voices_path: str,
        num_workers: int = 2,
        device: str = "cuda",
        dtype: str = "float32",
        warmup: bool = True,
        cache_size: int = 32,
    ):
        self._num_workers = num_workers
        self._device = device
        self._dtype = dtype
        self._model_path = model_path
        self._voices_path = voices_path
        self._warmup = warmup
        self._cache_size = max(0, cache_size)
        self._current_idx = 0
        self._workers: list[KokoroWorker] = []
        self._lock = threading.Lock()
        self._initialized = False
        self._init_errors: list[str] = []
        self._cache: "OrderedDict[tuple[str, str], bytes]" = OrderedDict()
        self._cache_lock = threading.Lock()

    def initialize(self):
        """Initialize all workers. Collects failures and raises if all workers failed."""
        if self._initialized:
            return

        self._init_errors = []

        # Workers are built concurrently. The shared KModel load is serialised
        # by its own lock, so only the first worker pays for it; the others go
        # straight to building their (comparatively cheap) G2P pipeline.
        workers = [
            KokoroWorker(self._model_path, self._voices_path, self._device, self._dtype)
            for _ in range(self._num_workers)
        ]

        results: list[tuple[int, KokoroWorker, Optional[str]]] = []
        if workers:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
                futures = {pool.submit(w._initialize): (i, w) for i, w in enumerate(workers)}
                for future in concurrent.futures.as_completed(futures):
                    i, w = futures[future]
                    try:
                        error = future.result()
                    except Exception as e:  # pragma: no cover - defensive
                        error = f"Kokoro worker init raised: {e}"
                    results.append((i, w, error))

        for i, worker, error in sorted(results, key=lambda r: r[0]):
            if error:
                self._init_errors.append(f"Worker {i}: {error}")
                worker.shutdown()
            else:
                self._workers.append(worker)
                logger.info(f"Kokoro worker {i} created on {worker._device}")

        # If zero workers requested, that's a valid (if useless) config - don't raise
        if self._num_workers == 0:
            logger.info("KokoroEngine initialized with 0 workers (no TTS available)")
            self._initialized = True
            return

        if not self._workers:
            error_summary = "; ".join(self._init_errors) if self._init_errors else "No workers initialized"
            raise RuntimeError(f"KokoroEngine initialization failed: {error_summary}")

        if self._init_errors:
            # Some workers failed - log warnings but continue with working ones
            logger.warning(f"KokoroEngine: {len(self._init_errors)} worker(s) failed: {'; '.join(self._init_errors)}")

        self._initialized = True

        if self._warmup:
            self.warmup()

    def warmup(self) -> int:
        """Pay the cold-start cost on every worker now, in parallel."""
        if not self._workers:
            return 0

        import time
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._workers)) as pool:
            warmed = sum(1 for ok in pool.map(lambda w: w.warmup(), self._workers) if ok)

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(f"Kokoro warmup complete: {warmed}/{len(self._workers)} workers in {elapsed_ms:.0f} ms")
        return warmed

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _cache_get(self, key: tuple[str, str]) -> Optional[bytes]:
        if self._cache_size <= 0:
            return None
        with self._cache_lock:
            wav = self._cache.get(key)
            if wav is not None:
                self._cache.move_to_end(key)
            return wav

    def _cache_put(self, key: tuple[str, str], wav: bytes) -> None:
        if self._cache_size <= 0 or not wav:
            return
        with self._cache_lock:
            self._cache[key] = wav
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _next_worker(self) -> Optional[KokoroWorker]:
        """Round-robin worker selection, safe across threads."""
        with self._lock:
            if not self._workers:
                return None
            worker = self._workers[self._current_idx % len(self._workers)]
            self._current_idx += 1
            return worker

    def synthesize(self, text: str, voice: str = "af_bella") -> Optional[bytes]:
        """Synthesize speech, serving repeated phrases from the cache."""
        if not self._initialized:
            self.initialize()

        if not self._workers:
            logger.error("No Kokoro workers available")
            return None

        # Grace repeats a handful of canned lines ("Goal executed.", "I'm here
        # to help!", ...) on almost every turn; re-synthesising them is pure
        # waste.
        key = (text, voice)
        cached = self._cache_get(key)
        if cached is not None:
            logger.debug(f"Kokoro cache hit for '{text[:40]}'")
            return cached

        worker = self._next_worker()
        if worker is None:
            return None

        wav = worker.synthesize(text, voice)
        if wav:
            self._cache_put(key, wav)
        return wav

    def submit(self, text: str, voice: str = "af_bella"):
        """Start synthesis without blocking.

        Returns either a Future or, on a cache hit, the WAV bytes directly.
        Callers should use :func:`resolve` to normalise the two.
        """
        if not self._initialized:
            self.initialize()
        if not self._workers:
            return None

        key = (text, voice)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        worker = self._next_worker()
        if worker is None:
            return None
        return worker.submit(text, voice)

    def resolve(self, pending, text: str, voice: str = "af_bella") -> Optional[bytes]:
        """Resolve whatever :func:`submit` returned into WAV bytes."""
        if pending is None:
            return None
        if isinstance(pending, (bytes, bytearray)):
            return bytes(pending)
        try:
            wav = pending.result(timeout=30)
        except Exception as e:
            logger.error(f"Kokoro synthesis execution error: {e}")
            return None
        if wav:
            self._cache_put((text, voice), wav)
        return wav

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

    def shutdown(self) -> None:
        for worker in self._workers:
            worker.shutdown()

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        return split_sentences(text)
