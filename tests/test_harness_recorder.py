"""Tests for the session tape recorder.

The recorder is graded on two properties that matter more than its features:

* it never breaks a turn, whatever goes wrong with the tape, and
* it records chunk boundaries and raw PCM faithfully, because those are the
  two things the Rust port is most likely to get subtly wrong and the two
  things a summarised recording would erase the evidence of.
"""

import json
import os

import pytest

from grace.harness.recorder import Recorder, get_recorder, reset_recorder_for_tests


@pytest.fixture
def recorder(tmp_path):
    rec = Recorder(str(tmp_path), session_id="test-session")
    yield rec
    rec.close()


def _read(rec: Recorder, stream: str) -> list[dict]:
    path = os.path.join(rec.dir, f"{stream}.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_events_are_recorded_in_order_with_offsets(recorder):
    recorder.record_event({"type": "WakeWordDetected"})
    recorder.record_event({"type": "FinalTranscript", "text": "open notepad"})

    records = _read(recorder, "events")
    assert [r["i"] for r in records] == [0, 1]
    assert [r["event"]["type"] for r in records] == ["WakeWordDetected", "FinalTranscript"]
    assert all(isinstance(r["t_ms"], float) for r in records)
    # Offsets, never wall clock - tapes have to diff across runs and machines.
    assert records[0]["t_ms"] <= records[1]["t_ms"]


def test_llm_chunks_are_recorded_individually_not_joined(recorder):
    """Chunk boundaries drive ResponseChunk/SpeechChunk and thus TTS splits.

    A recording that stored only the joined text would let a Rust SSE reader
    with different buffering pass the tape while producing a different UI
    stream and different sentence boundaries.
    """
    recorder.record_llm(
        kind="chat",
        request={"messages": [{"role": "user", "content": "hi"}]},
        chunks=["Sure", ", opening", " that now."],
    )

    record = _read(recorder, "llm")[0]
    assert record["chunks"] == ["Sure", ", opening", " that now."]
    assert "response" not in record


def test_llm_request_is_recorded_verbatim(recorder):
    """Prompt-string byte equality is a migration gate; a paraphrase is useless."""
    prompt = "You are Grace.\n\nTools:\n  - open_app\n"
    recorder.record_llm(kind="chat", request={"messages": [{"role": "system", "content": prompt}]})

    record = _read(recorder, "llm")[0]
    assert record["request"]["messages"][0]["content"] == prompt
    assert record["request_digest"]


def test_identical_requests_share_a_digest(recorder):
    request = {"messages": [{"role": "user", "content": "hello"}], "temperature": 0.1}
    recorder.record_llm(kind="chat", request=request, chunks=["a"])
    recorder.record_llm(kind="chat", request=dict(reversed(list(request.items()))), chunks=["b"])

    first, second = _read(recorder, "llm")
    # Key order must not change the digest, or replay lookups miss.
    assert first["request_digest"] == second["request_digest"]


def test_stt_writes_raw_pcm_and_transcript(recorder):
    pcm = bytes(range(256)) * 4
    recorder.record_stt(pcm, "open notepad")

    record = _read(recorder, "stt")[0]
    assert record["text"] == "open notepad"
    assert record["pcm_bytes"] == len(pcm)

    path = os.path.join(recorder.pcm_dir, f"{record['pcm_digest']}.raw")
    with open(path, "rb") as fh:
        assert fh.read() == pcm


def test_repeated_pcm_is_stored_once(recorder):
    pcm = b"\x01\x02" * 100
    recorder.record_stt(pcm, "first")
    recorder.record_stt(pcm, "second")

    records = _read(recorder, "stt")
    assert records[0]["pcm_digest"] == records[1]["pcm_digest"]
    assert len(os.listdir(recorder.pcm_dir)) == 1


def test_dispatch_records_tool_params_and_result(recorder):
    recorder.record_dispatch("open_app", {"name": "notepad"}, {"status": "ok"})

    record = _read(recorder, "dispatch")[0]
    assert record["tool"] == "open_app"
    assert record["params"] == {"name": "notepad"}
    assert record["result"] == {"status": "ok"}


def test_meta_redacts_api_keys(recorder):
    class FakeConfig:
        gemini_api_key = "AIza-super-secret"
        use_cloud_llm = True
        llama_port = 8080

    recorder.write_meta(FakeConfig())

    with open(os.path.join(recorder.dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    assert meta["config"]["gemini_api_key"] == "<redacted>"
    assert meta["config"]["use_cloud_llm"] is True
    assert meta["config"]["llama_port"] == 8080


def test_an_unserialisable_value_costs_only_that_value(recorder):
    """A recorder fault must cost the tape, never the user's turn.

    A value whose to_dict() and repr() both raise degrades to an empty object
    and the surrounding record still lands - a partially-legible tape is worth
    more than a dropped one, and far more than a failed turn.
    """
    class Exploding:
        def to_dict(self):
            raise RuntimeError("boom")

        def __repr__(self):
            raise RuntimeError("boom")

    recorder.record_dispatch("open_app", {"bad": Exploding(), "name": "notepad"}, {"status": "ok"})

    record = _read(recorder, "dispatch")[0]
    assert record["tool"] == "open_app"
    assert record["params"]["name"] == "notepad"
    assert record["params"]["bad"] == {}
    assert record["result"] == {"status": "ok"}


def test_an_unwritable_stream_never_propagates(recorder):
    """If the tape file itself cannot be written, the turn still proceeds."""
    class Unwritable:
        def write(self, _data):
            raise OSError("disk full")

        def flush(self):
            raise OSError("disk full")

        def close(self):
            pass

    recorder._handles["events"] = Unwritable()
    recorder.record_event({"type": "Idle"})  # must not raise
    recorder.close()


def test_bytes_are_digested_not_inlined(recorder):
    recorder.record_dispatch("read_pdf", {"blob": b"\x00" * 4096}, {"status": "ok"})

    record = _read(recorder, "dispatch")[0]
    assert record["params"]["blob"]["__bytes__"] == 4096
    assert "digest" in record["params"]["blob"]


class TestRecorderResolution:
    """get_recorder() reads the environment exactly once, by design."""

    def setup_method(self):
        reset_recorder_for_tests()

    def teardown_method(self):
        reset_recorder_for_tests()

    def test_disabled_without_the_env_var(self, monkeypatch):
        monkeypatch.delenv("GRACE_RECORD_DIR", raising=False)
        assert get_recorder() is None

    def test_enabled_by_the_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GRACE_RECORD_DIR", str(tmp_path))
        rec = get_recorder()
        assert rec is not None
        assert os.path.isdir(rec.dir)

    def test_resolution_is_memoized(self, monkeypatch, tmp_path):
        """Flipping mid-run would leave a hole in the tape - worse than no tape."""
        monkeypatch.setenv("GRACE_RECORD_DIR", str(tmp_path))
        first = get_recorder()
        monkeypatch.delenv("GRACE_RECORD_DIR", raising=False)
        assert get_recorder() is first

    def test_unwritable_root_degrades_to_no_recording(self, monkeypatch, tmp_path):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        monkeypatch.setenv("GRACE_RECORD_DIR", str(blocker))
        assert get_recorder() is None
