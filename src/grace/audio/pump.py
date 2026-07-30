"""Microphone pump: moves blocking mic reads off the asyncio event loop.

`AudioCapture.get_chunk()` is a blocking `stream.read()` of ~32 ms. The main
loop and both listening loops used to call it directly inside `async def`, so
for roughly 32 ms out of every 32 ms nothing else on the loop could run - not
the WebSocket server, not TTS scheduling, not the agent's HTTP calls.

The pump reads on a dedicated thread and hands chunks to the loop through an
asyncio queue. When the loop is busy the queue drops its *oldest* chunks rather
than blocking the reader, because stale microphone audio is worthless and a
blocked reader would cause a real PortAudio overflow.
"""

import asyncio
import logging
import queue
import threading
from typing import Optional

logger = logging.getLogger("grace.audio.pump")

# ~2 seconds of 512-sample chunks at 16 kHz. Enough to absorb a slow turn of
# the event loop, small enough that nothing badly stale survives.
DEFAULT_MAXSIZE = 64


class AudioPump:
    """Owns a reader thread feeding an asyncio queue of PCM chunks."""

    def __init__(self, capture, maxsize: int = DEFAULT_MAXSIZE):
        self._capture = capture
        self._maxsize = maxsize
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._dropped = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def dropped_chunks(self) -> int:
        """Chunks discarded because the loop could not keep up. Diagnostic only."""
        return self._dropped

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> "AudioPump":
        if self._running:
            return self
        self._loop = loop or asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._maxsize)
        self._running = True
        self._thread = threading.Thread(target=self._reader, name="grace-audio-pump", daemon=True)
        self._thread.start()
        logger.info("Audio pump started")
        return self

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            # One chunk read (~32 ms) is the longest the thread can be stuck.
            thread.join(timeout=1.0)
        logger.info(f"Audio pump stopped ({self._dropped} chunks dropped)")

    def _reader(self) -> None:
        while self._running:
            try:
                chunk = self._capture.get_chunk()
            except Exception as e:
                logger.warning(f"Audio pump read failed: {e}")
                # Don't spin at full speed on a dead device.
                threading.Event().wait(0.05)
                continue
            if not chunk or not self._running:
                continue
            loop = self._loop
            if loop is None:
                continue
            try:
                loop.call_soon_threadsafe(self._offer, chunk)
            except RuntimeError:
                # Loop closed underneath us during shutdown.
                break

    def _offer(self, chunk: bytes) -> None:
        """Enqueue on the loop thread, evicting the oldest chunk when full."""
        q = self._queue
        if q is None:
            return
        if q.full():
            try:
                q.get_nowait()
                self._dropped += 1
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(chunk)
        except asyncio.QueueFull:
            self._dropped += 1

    async def get(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Await the next chunk. Returns None on timeout or when not running."""
        q = self._queue
        if q is None:
            return None
        if timeout is None:
            return await q.get()
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def drain(self) -> int:
        """Discard buffered audio so a new listening window starts clean.

        Without this, the first thing Whisper hears after a long agent step is
        whatever was said *during* it.
        """
        q = self._queue
        if q is None:
            return 0
        count = 0
        while True:
            try:
                q.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        return count


class SyncChunkSource:
    """Fallback source used when no pump is running (tests, headless use)."""

    def __init__(self, capture):
        self._capture = capture

    async def get(self, timeout: Optional[float] = None) -> Optional[bytes]:
        return await asyncio.to_thread(self._capture.get_chunk)

    def drain(self) -> int:
        return 0
