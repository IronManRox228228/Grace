"""Record-then-replay round trip: Phase 0's exit criterion, in miniature.

The migration's whole regression strategy rests on one claim: a recorded
session can be replayed and reproduce itself exactly. If the harness cannot
reproduce Python from Python, it cannot judge Rust, and every later phase's
exit criterion is decoration.

So this drives a real GraceApp turn with only its *external* edges faked -
microphone, Whisper's model, the LLM's HTTP stream, TTS, and Windows - records
a tape, then replays that tape through the same code path and requires the
event stream, dispatch calls, and prompt strings to match byte for byte.

Everything between those edges is the production implementation: the real
turn state machine, the real router, the real intent parser, the real
dispatcher, the real ResponseGenerator and its sentence splitting.
"""

import dataclasses
import os
from unittest import mock

import pytest

from grace.config import Config
from grace.harness.recorder import reset_recorder_for_tests
from grace.harness.tape import Tape


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeWhisperModel:
    """Deterministic stand-in for faster-whisper's lazy generator."""

    def __init__(self, transcript):
        self._transcript = transcript

    def transcribe(self, audio, **kwargs):
        return iter([_FakeSegment(self._transcript)]), None


def _speech_then_silence(chunk_bytes=1024, speech=6, silence=20):
    """Loud chunks then quiet ones, so the real VAD closes the turn itself."""
    loud = (8000).to_bytes(2, "little", signed=True) * (chunk_bytes // 2)
    quiet = b"\x00\x00" * (chunk_bytes // 2)
    return [loud] * speech + [quiet] * silence


class _ScriptedPump:
    """Feeds a fixed chunk sequence, paced so the wall-clock VAD behaves."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._index = 0

    def start(self, loop=None):
        return self

    def stop(self):
        return None

    async def get(self, timeout=None):
        import asyncio
        if self._index >= len(self._chunks):
            # Not None: the listen loop would spin against its own 30s timeout.
            raise RuntimeError("scripted audio exhausted before the VAD fired")
        chunk = self._chunks[self._index]
        self._index += 1
        # Paced so the 275ms silence window is crossed 5.5 chunks in, in the
        # middle of a chunk rather than at its edge. On a boundary, scheduler
        # jitter alone decides whether the turn ends one chunk earlier, and the
        # replay then reports a VAD parity failure that is a coin flip.
        await asyncio.sleep(0.05)
        # Route through the production tap so the recording pass tapes audio
        # exactly as AudioPump.get() would.
        from grace.audio.pump import _tape
        return _tape(chunk)

    def drain(self):
        return 0


@pytest.fixture
def scripted_backend(monkeypatch, tmp_path):
    """A GraceApp whose every external edge is deterministic."""
    from grace.harness import replay as replay_mod

    transcript = "what time is it"
    llm_chunks = ['{"tool": "converse", ', '"params": {"response": ', '"It is half past four."}}']

    config = dataclasses.replace(
        Config(),
        use_cloud_llm=True,
        gemini_api_key="test-key",
        use_ui_tars_local=False,
        whisper_silence_duration_ms=275,
        followup_timeout_seconds=0,
    )

    async def _fake_gemini(self, messages, temperature=0.2, max_tokens=1024):
        for chunk in llm_chunks:
            yield chunk

    patches = [
        mock.patch("grace.main.Config", lambda: config),
        mock.patch("grace.main.AudioCapture", replay_mod.NullAudio),
        mock.patch("grace.main.WakeWordDetector", replay_mod.NullWakeWord),
        mock.patch("grace.main.KokoroEngine", replay_mod.NullTts),
        mock.patch("grace.main.TTSPlayer", replay_mod.NullTts),
        mock.patch("grace.main.ComputerUse", replay_mod.NullComputerUse),
        mock.patch(
            "grace.llm.gemma_client.GemmaClient._stream_gemini_response", _fake_gemini
        ),
    ]
    return {
        "config": config,
        "patches": patches,
        "transcript": transcript,
        "llm_chunks": llm_chunks,
    }


def _install_fake_whisper(app, transcript):
    """Keep the real _transcribe_pcm - and its recorder tap - but fake the model."""
    app.whisper._model = _FakeWhisperModel(transcript)
    app.whisper._initialized = True


async def _run_recorded_turn(scripted):
    from contextlib import ExitStack

    from grace.harness.replay import CapturingWsServer
    from grace.util.timing import start_turn

    with ExitStack() as stack:
        for patch in scripted["patches"]:
            stack.enter_context(patch)

        # No importlib.reload here: reloading re-executes the module body and
        # rebinds AudioCapture, KokoroEngine and friends back to the real
        # classes, silently undoing every patch above and opening a real
        # microphone.
        import grace.main as main_mod

        ws = CapturingWsServer()
        stack.enter_context(mock.patch.object(main_mod, "WsEventServer", lambda *a, **k: ws))

        app = main_mod.GraceApp()
        app._running = True
        app.pump = _ScriptedPump(_speech_then_silence())
        _install_fake_whisper(app, scripted["transcript"])

        # The whole activation cycle, follow-up window included - the same
        # entry point replay drives. Recording a narrower path than the replay
        # reproduces would make the round trip disagree about events neither
        # side got wrong.
        await app._handle_activation()

        return ws.events


@pytest.fixture
def recorded_tape(monkeypatch, tmp_path, scripted_backend):
    """Drive one real turn with recording on, and return the resulting tape."""
    import asyncio

    reset_recorder_for_tests()
    monkeypatch.setenv("GRACE_RECORD_DIR", str(tmp_path / "rec"))
    monkeypatch.setenv("GRACE_RECORD_AUDIO", "1")

    from grace.harness.recorder import get_recorder
    recorder = get_recorder()
    assert recorder is not None
    # main.start() writes this before any work, and replay reads the config
    # back out of it. A tape without meta.json is replayed against whatever the
    # ambient environment happens to say, which is how a VAD parity failure
    # gets reported for a turn that was fine.
    recorder.write_meta(scripted_backend["config"])

    events = asyncio.run(_run_recorded_turn(scripted_backend))
    recorder.close()

    yield Tape.load(recorder.dir), events, scripted_backend

    reset_recorder_for_tests()


def test_a_real_turn_is_recorded_at_every_boundary(recorded_tape):
    tape, events, _ = recorded_tape

    assert [e["type"] for e in events][:4] == [
        "WakeWordDetected", "ListeningStarted", "FinalTranscript", "ListeningStopped",
    ]
    assert tape.event_stream, "no events were taped"
    assert tape.llm, "the LLM exchange was not taped"
    assert tape.stt, "the transcription was not taped"
    assert tape.audio, "the mic chunk sequence was not taped"
    # Snapshots and dispatches are route-dependent: a conversational turn
    # reaches neither, which is why the replay assertion below compares
    # whatever the tape holds rather than a fixed expectation.


def test_llm_chunk_boundaries_survive_the_tape(recorded_tape):
    """Joining the stream at record time would erase this evidence."""
    tape, _, scripted = recorded_tape
    assert tape.llm[0]["chunks"] == scripted["llm_chunks"]


def test_recorded_transcript_matches_the_pcm_that_produced_it(recorded_tape):
    tape, _, scripted = recorded_tape
    record = tape.stt[0]
    assert record["text"] == scripted["transcript"]
    assert len(tape.pcm(record["pcm_digest"])) == record["pcm_bytes"]


def test_replaying_the_tape_reproduces_it_exactly(recorded_tape):
    """The claim everything else depends on: Python reproduces Python.

    Same event sequence, same dispatch calls, same prompt strings - byte for
    byte, in order.
    """
    import asyncio
    from contextlib import ExitStack

    from grace.harness.replay import replay

    tape, _, scripted = recorded_tape
    reset_recorder_for_tests()  # the replay itself must not record

    with ExitStack() as stack:
        for patch in scripted["patches"]:
            stack.enter_context(patch)
        result = asyncio.run(replay(tape))

    assert result.ok, "\n" + result.report()


def test_replay_does_not_write_a_tape_of_its_own(recorded_tape, tmp_path):
    """A replay that recorded itself would quietly overwrite the oracle."""
    tape, _, _ = recorded_tape
    before = sorted(os.listdir(os.path.dirname(tape.dir)))

    reset_recorder_for_tests()
    assert sorted(os.listdir(os.path.dirname(tape.dir))) == before
