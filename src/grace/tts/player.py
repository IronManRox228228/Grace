"""TTS audio playback via native Windows winsound.

Streams WAV audio directly to default Windows speakers using winsound with SND_FILENAME | SND_ASYNC,
eliminating PortAudio device handle lock contention with PyAudio mic capture.
"""

import logging
import os
import tempfile
import threading
import time
import winsound
from typing import Optional

logger = logging.getLogger("grace.tts.player")


class TTSPlayer:
    """Low-latency audio playback queue for TTS output using native winsound."""

    def __init__(self, sample_rate: int = 24000):
        self._sample_rate = sample_rate
        self._queue: list[bytes] = []
        self._playing = False
        self._lock = threading.RLock()
        self._current_tmp_file: Optional[str] = None

    def play(self, wav_bytes: bytes):
        """Queue WAV audio bytes for playback."""
        logger.debug(f"TTSPlayer.play called with {len(wav_bytes)} bytes")
        with self._lock:
            self._queue.append(wav_bytes)
            if not self._playing:
                self._play_next()
        logger.debug("TTSPlayer.play successfully queued chunk")

    def wait(self, timeout: float = 15.0) -> bool:
        """Wait until all queued audio chunks have finished playing."""
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if not self._playing and not self._queue:
                    return True
            time.sleep(0.05)
        return False

    def play_sync(self, wav_bytes: bytes):
        """Play WAV bytes synchronously (blocking)."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name
            winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
        except Exception as e:
            logger.error(f"winsound sync playback failed: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _play_next(self):
        """Play the next queued item."""
        with self._lock:
            if not self._queue:
                self._playing = False
                return
            wav_bytes = self._queue.pop(0)
            self._playing = True

        def _worker():
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(wav_bytes)
                    tmp_path = tmp.name
                
                duration = self._get_wav_duration(wav_bytes)
                logger.info(f"Playing TTS audio chunk ({duration:.2f}s, {len(wav_bytes)} bytes) via winsound SND_FILENAME")
                
                # Play synchronously inside background thread for sequential audio output
                winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
            except Exception as e:
                logger.error(f"Failed to play audio chunk via winsound: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                self._play_next()

        threading.Thread(target=_worker, daemon=True).start()

    def _get_wav_duration(self, wav_bytes: bytes) -> float:
        """Calculate duration of WAV audio bytes in seconds."""
        import wave
        import io
        try:
            with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                return nframes / float(framerate) if framerate > 0 else 0.5
        except Exception:
            return 1.0

    def stop(self):
        """Stop current playback immediately (Barge-In interruption)."""
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
            winsound.PlaySound(None, 0)
        except Exception as e:
            logger.debug(f"winsound stop purge exception: {e}")
        with self._lock:
            self._queue.clear()
            self._playing = False
            logger.info("TTSPlayer playback cancelled via Barge-In stop()")

    @property
    def queue_size(self) -> int:
        """Current number of chunks in the queue."""
        return len(self._queue)
