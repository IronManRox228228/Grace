"""Tests for the microphone pump.

The pump exists so blocking mic reads stop stalling the event loop. The two
behaviours worth pinning down are that the loop really is free while the reader
blocks, and that a backed-up queue sheds its *oldest* audio rather than
blocking the reader or handing Whisper stale speech.
"""

import asyncio
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.audio.pump import AudioPump, SyncChunkSource


class FakeCapture:
    """A microphone that blocks for `read_delay` and emits numbered chunks."""

    def __init__(self, read_delay=0.005, payload_size=4):
        self.read_delay = read_delay
        self.payload_size = payload_size
        self.reads = 0
        self._lock = threading.Lock()

    def get_chunk(self) -> bytes:
        time.sleep(self.read_delay)
        with self._lock:
            self.reads += 1
            n = self.reads
        return n.to_bytes(self.payload_size, "little")


class ExplodingCapture:
    def __init__(self):
        self.calls = 0

    def get_chunk(self) -> bytes:
        self.calls += 1
        raise OSError("device gone")


def chunk_number(chunk: bytes) -> int:
    return int.from_bytes(chunk, "little")


class TestPumpDelivery:
    @pytest.mark.asyncio
    async def test_delivers_chunks_in_order(self):
        pump = AudioPump(FakeCapture(), maxsize=16).start()
        try:
            first = chunk_number(await pump.get(timeout=2.0))
            second = chunk_number(await pump.get(timeout=2.0))
            third = chunk_number(await pump.get(timeout=2.0))
        finally:
            pump.stop()
        assert [first, second, third] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_get_returns_none_on_timeout(self):
        # A capture slower than the timeout must not hang the caller.
        pump = AudioPump(FakeCapture(read_delay=1.0), maxsize=4).start()
        try:
            assert await pump.get(timeout=0.05) is None
        finally:
            pump.stop()

    @pytest.mark.asyncio
    async def test_get_before_start_returns_none(self):
        pump = AudioPump(FakeCapture())
        assert await pump.get(timeout=0.01) is None
        assert pump.is_running is False

    @pytest.mark.asyncio
    async def test_read_errors_do_not_kill_the_pump(self):
        capture = ExplodingCapture()
        pump = AudioPump(capture, maxsize=4).start()
        try:
            await asyncio.sleep(0.2)
            assert await pump.get(timeout=0.05) is None
            # It kept retrying rather than exiting on the first failure...
            assert capture.calls > 1
            # ...and it backed off instead of spinning at full speed.
            assert capture.calls < 200
        finally:
            pump.stop()


class TestEventLoopIsFree:
    @pytest.mark.asyncio
    async def test_loop_keeps_running_during_blocking_reads(self):
        """The regression this class exists for.

        With direct capture.get_chunk() calls inside `async def`, a coroutine
        ticking alongside the reads made almost no progress. Here it should
        tick freely while the reader thread blocks.
        """
        pump = AudioPump(FakeCapture(read_delay=0.02), maxsize=64).start()
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                # sleep(0) is a bare yield. A timed sleep would measure
                # Windows' ~15.6ms timer granularity instead of loop freedom.
                await asyncio.sleep(0)

        task = asyncio.create_task(ticker())
        try:
            for _ in range(3):
                assert await pump.get(timeout=2.0) is not None
        finally:
            task.cancel()
            pump.stop()

        # 3 chunks x 20ms of blocking reads. A free loop gets through thousands
        # of yields in 60ms; a loop blocked on the reads gets a literal handful.
        assert ticks > 1000


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_full_queue_drops_oldest_not_newest(self):
        pump = AudioPump(FakeCapture(read_delay=0.001), maxsize=4).start()
        try:
            # Let the reader run well past the queue's capacity.
            await asyncio.sleep(0.3)
            first = chunk_number(await pump.get(timeout=1.0))
        finally:
            pump.stop()

        assert pump.dropped_chunks > 0
        # The surviving chunks are recent ones, not chunk #1.
        assert first > 4

    @pytest.mark.asyncio
    async def test_drain_discards_buffered_audio(self):
        pump = AudioPump(FakeCapture(read_delay=0.001), maxsize=32).start()
        try:
            await asyncio.sleep(0.1)
            assert pump.drain() > 0
            assert pump.drain() == 0
        finally:
            pump.stop()

    def test_drain_before_start_is_safe(self):
        assert AudioPump(FakeCapture()).drain() == 0


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self):
        pump = AudioPump(FakeCapture()).start()
        pump.stop()
        pump.stop()
        assert pump.is_running is False

    @pytest.mark.asyncio
    async def test_double_start_does_not_spawn_two_readers(self):
        pump = AudioPump(FakeCapture()).start()
        thread = pump._thread
        try:
            pump.start()
            assert pump._thread is thread
        finally:
            pump.stop()

    @pytest.mark.asyncio
    async def test_reader_thread_exits_on_stop(self):
        capture = FakeCapture(read_delay=0.005)
        pump = AudioPump(capture).start()
        thread = pump._thread
        pump.stop()
        assert thread is not None and not thread.is_alive()


class TestSyncChunkSource:
    @pytest.mark.asyncio
    async def test_reads_through_to_capture(self):
        source = SyncChunkSource(FakeCapture())
        assert chunk_number(await source.get()) == 1
        assert source.drain() == 0
