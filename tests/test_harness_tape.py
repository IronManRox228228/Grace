"""Tests for tape loading and the replay diff engine.

The diff engine decides whether the Rust port shipped, so its own failure modes
matter: a comparison that is too strict blocks the migration on noise, and one
that is too loose lets a real regression through. Each test below pins which
side of that line a given class of difference falls on.
"""

import json
import os

import pytest

from grace.harness.recorder import Recorder
from grace.harness.tape import (
    Tape,
    diff_dispatches,
    diff_event_streams,
    diff_prompts,
    diff_stage_timings,
    diff_transcripts,
    summarise,
)


@pytest.fixture
def tape(tmp_path):
    """A small but complete recorded session, round-tripped through Recorder."""
    rec = Recorder(str(tmp_path), session_id="s1")
    rec.record_event({"type": "WakeWordDetected"})
    rec.record_event({"type": "FinalTranscript", "text": "open notepad"})
    rec.record_event({"type": "TurnTrace", "trace": {
        "label": "activation", "total_ms": 900.0,
        "stages": [{"name": "whisper", "depth": 0, "ms": 400.0, "detail": None},
                   {"name": "whisper", "depth": 0, "ms": 800.0, "detail": None}],
        "events": {},
    }})
    rec.record_event({"type": "ConversationFinished"})
    rec.record_llm(kind="chat", request={"messages": [{"role": "user", "content": "open notepad"}]},
                   chunks=['{"tool":', ' "open_app"}'])
    rec.record_stt(b"\x01\x02" * 64, "open notepad")
    rec.record_dispatch("open_app", {"name": "notepad"}, {"status": "ok"})
    rec.record_audio_chunk(b"\xaa" * 32)
    rec.record_audio_chunk(b"\xbb" * 32)
    rec.close()
    return Tape.load(rec.dir)


def test_load_reads_every_stream(tape):
    assert len(tape.events) == 4
    assert len(tape.llm) == 1
    assert len(tape.stt) == 1
    assert len(tape.dispatch) == 1
    assert len(tape.audio) == 2


def test_event_stream_excludes_diagnostics(tape):
    """TurnTrace carries wall-clock durations that differ on every run."""
    assert [e["type"] for e in tape.event_stream] == [
        "WakeWordDetected", "FinalTranscript", "ConversationFinished",
    ]


def test_audio_chunks_replay_in_order_with_arrival_offsets(tape):
    """Offsets are load-bearing: the VAD accumulates silence in wall-clock."""
    chunks = list(tape.audio_chunks())
    assert [payload for _, payload in chunks] == [b"\xaa" * 32, b"\xbb" * 32]
    assert all(isinstance(offset, float) for offset, _ in chunks)
    assert chunks[0][0] <= chunks[1][0]


def test_pcm_round_trips(tape):
    digest = tape.stt[0]["pcm_digest"]
    assert tape.pcm(digest) == b"\x01\x02" * 64


def test_llm_index_keeps_repeats_in_order(tmp_path):
    """The agent loop legitimately re-plans against an unchanged screen."""
    rec = Recorder(str(tmp_path), session_id="s2")
    request = {"messages": [{"role": "user", "content": "same"}]}
    rec.record_llm(kind="chat", request=request, chunks=["first"])
    rec.record_llm(kind="chat", request=request, chunks=["second"])
    rec.close()

    index = Tape.load(rec.dir).llm_by_digest()
    (responses,) = index.values()
    assert [r["chunks"] for r in responses] == [["first"], ["second"]]


def test_stage_budget_uses_the_percentile_not_the_mean(tape):
    """A budget built from the mean would fail on the tape that produced it."""
    assert tape.stage_budget()["whisper"] == 800.0


class TestEventDiff:
    def test_identical_streams_pass(self):
        stream = [{"type": "Idle"}, {"type": "ListeningStarted"}]
        assert diff_event_streams(stream, list(stream)).ok

    def test_reordering_fails(self):
        """Same events in a different order is a different user experience."""
        a = [{"type": "ListeningStarted"}, {"type": "ListeningStopped"}]
        b = [{"type": "ListeningStopped"}, {"type": "ListeningStarted"}]
        assert not diff_event_streams(a, b).ok

    def test_payload_difference_fails(self):
        a = [{"type": "FinalTranscript", "text": "open notepad"}]
        b = [{"type": "FinalTranscript", "text": "open notepod"}]
        diff = diff_event_streams(a, b)
        assert not diff.ok
        assert "payload" in diff.mismatches[0]

    def test_missing_and_extra_events_are_reported_distinctly(self):
        assert "missing" in diff_event_streams([{"type": "Idle"}], []).mismatches[0]
        assert "extra" in diff_event_streams([], [{"type": "Idle"}]).mismatches[0]

    def test_chunk_boundaries_are_behaviour(self):
        """Same final text, different ResponseChunk splits, must fail.

        This is the SSE-buffering regression: identical prose, different TTS
        sentence boundaries and a different streaming feel.
        """
        a = [{"type": "ResponseChunk", "text": "Sure."}, {"type": "ResponseChunk", "text": " Opening."}]
        b = [{"type": "ResponseChunk", "text": "Sure. Opening."}]
        assert not diff_event_streams(a, b).ok


class TestDispatchDiff:
    def test_same_calls_pass(self):
        calls = [{"tool": "open_app", "params": {"name": "notepad"}}]
        assert diff_dispatches(calls, list(calls)).ok

    def test_different_route_to_the_same_place_fails(self):
        expected = [{"tool": "open_app", "params": {"name": "notepad"}}]
        actual = [
            {"tool": "cua_launch", "params": {"name": "notepad"}},
            {"tool": "cua_click", "params": {"target": "New"}},
        ]
        assert not diff_dispatches(expected, actual).ok

    def test_parameter_drift_fails(self):
        expected = [{"tool": "adjust_volume", "params": {"level": 50}}]
        actual = [{"tool": "adjust_volume", "params": {"level": 60}}]
        assert not diff_dispatches(expected, actual).ok


class TestPromptDiff:
    def test_identical_digests_pass(self):
        record = {"request_digest": "abc", "request": {"messages": []}}
        assert diff_prompts([record], [dict(record)]).ok

    def test_a_single_stray_newline_fails(self):
        """Exactly the failure mode a paraphrase-tolerant check would miss."""
        want = {"request_digest": "a", "request": {"system": "You are Grace.\nTools:"}}
        got = {"request_digest": "b", "request": {"system": "You are Grace.\n\nTools:"}}
        diff = diff_prompts([want], [got])
        assert not diff.ok
        assert "first difference at char" in diff.mismatches[0]


class TestTranscriptDiff:
    def _records(self, texts):
        return [{"pcm_digest": str(i), "text": t} for i, t in enumerate(texts)]

    def test_casing_and_terminal_punctuation_are_not_regressions(self):
        want = self._records(["Open Notepad."])
        got = self._records(["open notepad"])
        assert diff_transcripts(want, got).ok

    def test_rate_above_the_floor_passes(self):
        want = self._records([f"utterance {i}" for i in range(100)])
        got = self._records([f"utterance {i}" for i in range(99)] + ["different"])
        assert diff_transcripts(want, got).ok

    def test_rate_below_the_floor_fails(self):
        want = self._records([f"utterance {i}" for i in range(100)])
        got = self._records([f"utterance {i}" for i in range(90)] + ["x"] * 10)
        diff = diff_transcripts(want, got)
        assert not diff.ok
        assert "below the 98% floor" in diff.mismatches[0]

    def test_empty_tape_is_vacuously_ok(self):
        assert diff_transcripts([], []).ok


class TestTimingDiff:
    def test_faster_than_budget_passes(self):
        assert diff_stage_timings({"whisper": 800.0}, {"whisper": 200.0}).ok

    def test_within_headroom_passes(self):
        assert diff_stage_timings({"whisper": 800.0}, {"whisper": 950.0}).ok

    def test_beyond_headroom_fails(self):
        diff = diff_stage_timings({"whisper": 800.0}, {"whisper": 1500.0})
        assert not diff.ok
        assert "exceeds budget" in diff.mismatches[0]

    def test_a_stage_the_port_does_not_have_is_not_a_timing_failure(self):
        """Absence is an event/dispatch concern; this check only grades speed."""
        assert diff_stage_timings({"uia_legacy": 40.0}, {}).ok


def test_summarise_folds_results():
    ok, report = summarise([
        diff_event_streams([{"type": "Idle"}], [{"type": "Idle"}]),
        diff_dispatches([], [{"tool": "x", "params": {}}]),
    ])
    assert not ok
    assert "events: OK" in report
    assert "dispatch: 1 mismatch" in report
