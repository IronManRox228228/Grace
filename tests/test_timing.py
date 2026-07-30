"""Tests for the per-turn latency instrumentation."""

import asyncio
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.util import timing
from grace.util.timing import (
    TurnTrace,
    current_trace,
    end_turn,
    mark,
    mark_event,
    stage,
    start_turn,
)


@pytest.fixture(autouse=True)
def _clear_current_trace():
    """No test should leak an active trace into the next one."""
    end_turn()
    yield
    end_turn()


class TestTurnTrace:
    def test_stage_records_duration(self):
        trace = TurnTrace()
        with trace.stage("work"):
            time.sleep(0.01)

        assert len(trace.stages) == 1
        record = trace.stages[0]
        assert record.name == "work"
        assert record.duration_ms >= 9.0
        assert not record.is_open

    def test_nested_stages_track_depth(self):
        trace = TurnTrace()
        with trace.stage("outer"):
            with trace.stage("inner"):
                pass
            with trace.stage("sibling"):
                pass

        depths = {s.name: s.depth for s in trace.stages}
        assert depths == {"outer": 0, "inner": 1, "sibling": 1}
        # Depth must unwind fully, otherwise later stages drift right forever.
        assert trace._depth == 0

    def test_stage_closes_on_exception(self):
        trace = TurnTrace()
        with pytest.raises(ValueError):
            with trace.stage("boom"):
                raise ValueError("boom")

        assert trace.stages[0].duration_ms is not None
        assert trace._depth == 0

    def test_async_stage(self):
        trace = TurnTrace()

        async def run():
            async with trace.stage("async_work"):
                # A real sleep, not asyncio.sleep: asyncio compensates for the
                # ~15.6ms Windows timer resolution and can wake measurably
                # early, which makes a duration assertion flaky.
                time.sleep(0.01)

        asyncio.run(run())
        assert trace.stages[0].name == "async_work"
        assert trace.stages[0].duration_ms >= 9.0

    def test_detail_is_attached(self):
        trace = TurnTrace()
        with trace.stage("uia") as s:
            s.detail("42 elements")

        assert trace.stages[0].detail == "42 elements"
        assert "42 elements" in trace.summary()

    def test_mark_records_external_duration(self):
        trace = TurnTrace()
        trace.mark("external", 123.4, detail="from a worker")

        assert trace.stages[0].duration_ms == 123.4
        assert trace.stages[0].detail == "from a worker"

    def test_first_event_wins(self):
        trace = TurnTrace()
        first = trace.mark_event("first_audio_out")
        time.sleep(0.01)
        trace.mark_event("first_audio_out")

        assert trace.events["first_audio_out"] == first

    def test_to_dict_is_json_shaped(self):
        trace = TurnTrace(label="activation")
        with trace.stage("whisper"):
            pass
        trace.mark_event("first_audio_out")

        data = trace.to_dict()
        assert data["label"] == "activation"
        assert isinstance(data["total_ms"], float)
        assert data["stages"][0]["name"] == "whisper"
        assert "first_audio_out" in data["events"]

    def test_summary_is_multiline_and_mentions_stages(self):
        trace = TurnTrace(label="activation")
        with trace.stage("listen"):
            pass
        summary = trace.summary()

        assert "activation" in summary
        assert "listen" in summary
        assert "\n" in summary


class TestModuleLevelHelpers:
    def test_start_turn_installs_and_clears_current(self):
        assert current_trace() is None
        with start_turn("activation") as trace:
            assert current_trace() is trace
        assert current_trace() is None

    def test_helpers_are_noops_without_active_turn(self):
        # Library code calls these outside a turn (tests, warmup, scripts) and
        # must not blow up or need a guard at every call site.
        with stage("orphan"):
            pass
        mark("orphan", 1.0)
        mark_event("orphan")
        assert current_trace() is None

    def test_helpers_record_into_active_turn(self):
        with start_turn("activation") as trace:
            with stage("uia") as s:
                s.detail("7 elements")
            mark("external", 5.0)
            mark_event("first_audio_out")

        names = [s.name for s in trace.stages]
        assert names == ["uia", "external"]
        assert trace.stages[0].detail == "7 elements"
        assert "first_audio_out" in trace.events

    def test_async_helper_stage_records(self):
        async def run():
            with start_turn("activation") as trace:
                async with stage("planner"):
                    time.sleep(0.01)  # see test_async_stage
                return trace

        trace = asyncio.run(run())
        assert trace.stages[0].name == "planner"
        assert trace.stages[0].duration_ms >= 9.0

    def test_stage_visible_from_worker_thread(self):
        # asyncio.to_thread work (Kokoro, Whisper, UIA) must land in the same
        # trace, which is why this is a guarded global rather than a ContextVar.
        async def run():
            with start_turn("activation") as trace:
                def worker():
                    with stage("in_thread"):
                        pass

                await asyncio.to_thread(worker)
                return trace

        trace = asyncio.run(run())
        assert [s.name for s in trace.stages] == ["in_thread"]

    def test_end_turn_returns_trace(self):
        start_turn("activation")
        trace = end_turn()
        assert isinstance(trace, TurnTrace)
        assert current_trace() is None
        assert end_turn() is None

    def test_start_turn_replaces_previous(self):
        start_turn("first")
        with start_turn("second") as trace:
            assert current_trace() is trace
            assert trace.label == "second"

    def test_current_trace_cleared_even_if_body_raises(self):
        with pytest.raises(RuntimeError):
            with start_turn("activation"):
                raise RuntimeError("boom")
        assert current_trace() is None


class TestNullStage:
    def test_null_stage_supports_detail_chaining(self):
        with stage("orphan").detail("ignored"):
            pass

    def test_null_stage_propagates_exceptions(self):
        with pytest.raises(ValueError):
            with stage("orphan"):
                raise ValueError("boom")

    def test_null_async_stage(self):
        async def run():
            async with stage("orphan"):
                pass

        asyncio.run(run())
