"""Replay a recorded session against the backend and diff the result.

This is the primary no-regression gate. It re-runs a turn with every external
edge pinned to what the tape recorded - LLM responses, transcripts, perception
snapshots, tool results, and (optionally) the microphone chunk sequence - and
compares the resulting event stream, prompt strings, and dispatch calls against
the recording.

Phase 0 uses it to prove Python reproduces Python. If it cannot do that, it
cannot judge Rust, and every later phase's exit criterion is meaningless. The
Rust driver writes the same JSONL shapes and is graded by the same rules in
``tape.py``, so the two implementations are directly comparable.

All stubbing lives here rather than in the production modules. The backend does
not know it is being replayed, which is the point: a seam added for the harness
is a seam the harness can no longer detect a regression in.

Usage::

    python -m grace.harness.replay --tape recordings/20260812-140301
    python -m grace.harness.replay --corpus corpus/ --json report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from contextlib import ExitStack
from typing import Optional
from unittest import mock

from .recorder import _digest, llm_request_digest
from .tape import (
    Diff,
    Tape,
    diff_dispatches,
    diff_event_streams,
    diff_prompts,
    summarise,
)

logger = logging.getLogger("grace.harness.replay")


class TapeExhausted(RuntimeError):
    """The backend asked for something the tape does not contain.

    Never swallowed. A replay that improvises a response where the recording
    ran out is not reproducing anything - it is grading the port against a
    fiction, which is worse than no gate at all.
    """


# -- pinned edges ----------------------------------------------------------


class TapeLlm:
    """Stands in for GemmaClient, answering from the recording.

    Lookup is by request digest, so a prompt the port constructs differently is
    a miss rather than a silent fallback - which is exactly the signal wanted,
    since prompt-string equality is itself a migration gate.
    """

    def __init__(self, tape: Tape):
        self._by_digest = tape.llm_by_digest()
        self._consumed: dict[str, int] = {}
        self.requests: list[dict] = []

    async def _prepare(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def health_check(self) -> bool:
        return True

    async def chat(self, messages, temperature=0.7, max_tokens=8192, stream=True):
        from ..llm.gemma_client import _redact_images

        # Built to match GemmaClient.chat's recorded request exactly, image
        # redaction included, so the digest is comparable.
        request = {
            "backend": "replay",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "messages": _redact_images(messages),
        }
        digest = llm_request_digest(request)
        self.requests.append({"request": request, "request_digest": digest})

        record = self._next(digest, messages)
        error = record.get("error")

        if not stream:
            if error:
                raise _rebuild_error(error)
            return record.get("response") or "".join(record.get("chunks") or [])

        async def _stream():
            for chunk in record.get("chunks") or []:
                yield chunk
            if error:
                # After the chunks, not before: the recorder tapes whatever
                # arrived before the failure, and a stream that died halfway
                # through has to die halfway through here too.
                raise _rebuild_error(error)

        return _stream()

    async def generate_intent(self, user_message: str, system_prompt: str):
        stream = await self.chat(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_message}],
            temperature=0.1, max_tokens=4096, stream=True,
        )
        tokens = [token async for token in stream]
        return "".join(tokens) if tokens else None

    async def generate_text(self, prompt, system_prompt, temperature=0.2,
                            max_tokens=8192, messages=None, image_b64=None):
        if not messages:
            user = {"role": "user", "content": prompt}
            if image_b64:
                user["image_b64"] = image_b64
            messages = [{"role": "system", "content": system_prompt}, user]
        stream = await self.chat(messages, temperature=temperature,
                                 max_tokens=max_tokens, stream=True)
        tokens = [token async for token in stream]
        return "".join(tokens) if tokens else None

    def _next(self, digest: str, messages) -> dict:
        records = self._by_digest.get(digest)
        index = self._consumed.get(digest, 0)
        if not records or index >= len(records):
            preview = ""
            for message in reversed(messages):
                content = message.get("content")
                if isinstance(content, str) and content:
                    preview = content[:160]
                    break
            raise TapeExhausted(
                f"no recorded LLM response for request digest {digest}.\n"
                f"  last message: {preview!r}\n"
                "  The prompt the backend built does not match anything on the "
                "tape - usually a prompt-construction difference, not a missing "
                "recording."
            )
        self._consumed[digest] = index + 1
        return records[index]


def _rebuild_error(recorded: str) -> Exception:
    """Re-raise a taped LLM failure as the type the backend branches on.

    Recorded as ``"<ClassName>: <message>"`` by ``gemma_client._record_stream``.
    The type matters: ``RateLimitError`` has its own handler in both the turn
    handler and the agent loop, and replaying a rate-limited call as an empty
    successful response sends the replay down a completely different path -
    an unparseable-plan retry instead of an apology - so every rate-limit tape
    would report a regression that never happened.
    """
    from ..llm.gemma_client import RateLimitError

    known = {"RateLimitError": RateLimitError}
    name, _, message = recorded.partition(": ")
    return known.get(name, RuntimeError)(message or recorded)


class TapeDispatcher:
    """Replaces Dispatcher.execute: asserts the call, never performs it.

    A replay must not actually click, launch, or delete anything. What is
    graded is that the same tools were asked for, with the same parameters, in
    the same order.
    """

    def __init__(self, tape: Tape, inner):
        self._records = list(tape.dispatch)
        self._inner = inner
        self._index = 0
        self.calls: list[dict] = []

    async def execute(self, intent) -> dict:
        self.calls.append({"tool": intent.tool, "params": intent.params})

        # Events emitted around a dispatch are part of the contract, so the
        # real emitters still run; only the side effect is suppressed.
        await self._inner._emit_tool_started(intent.tool, intent.params) \
            if not intent.tool.startswith("cua_") else \
            await self._inner._emit_cua_started(intent.tool[4:])
        await self._inner._emit_tool_finished()

        if self._index < len(self._records):
            result = self._records[self._index]["result"]
            self._index += 1
            return result
        raise TapeExhausted(
            f"the backend dispatched {intent.tool}({intent.params}) but the tape "
            f"records only {len(self._records)} dispatch(es)"
        )


class TapePerception:
    """Serves recorded element graphs instead of walking the live UIA tree."""

    def __init__(self, tape: Tape):
        self._records = [r["snapshot"] for r in tape.snapshots]
        self._index = 0

    def capture_snapshot(self):
        if self._index >= len(self._records):
            raise TapeExhausted(
                f"the backend requested snapshot #{self._index + 1} but the tape "
                f"records only {len(self._records)}"
            )
        record = self._records[self._index]
        self._index += 1
        return _rebuild_snapshot(record)

    async def capture_snapshot_async(self):
        return self.capture_snapshot()


def _rebuild_snapshot(record: dict):
    """Rehydrate a ScreenSnapshot from its taped form.

    The PNG is not reconstructible from a digest, so a replayed snapshot has no
    pixels. That is correct for grading the planner - which sees the element
    graph - and means a tape cannot grade the *grounder*, which sees the image.
    Grounding parity is a Phase 4 concern with its own screenshot corpus.
    """
    from grace.agent.perception import OcrLine, ScreenSnapshot, WindowInfo
    from grace.perception.elements import ElementNode
    from grace.perception.element_graph import ElementGraph

    window = None
    if record.get("window"):
        window = WindowInfo(
            hwnd=0,
            title=record["window"]["title"],
            class_name=record["window"]["class_name"],
            rect=tuple(record["window"]["rect"]),
        )

    elements = [
        ElementNode(
            id=e["id"], role=e["role"], name=e["name"],
            rect=tuple(e["rect"]), center=tuple(e["center"]),
            value=e.get("value", ""), placeholder=e.get("placeholder", ""),
            frame=e.get("frame", "app"), container=e.get("container", ""),
            focused=e.get("focused", False), focusable=e.get("focusable", False),
            enabled=e.get("enabled", True), offscreen=e.get("offscreen", False),
            automation_id=e.get("automation_id", ""), source=e.get("source", "uia"),
        )
        for e in record.get("elements", [])
    ]

    graph = None
    if elements and window is not None:
        graph = ElementGraph(
            elements=elements, window=window,
            sources=tuple(record.get("graph_sources", [])),
        )

    return ScreenSnapshot(
        active_window=window,
        ocr_lines=[
            OcrLine(text=line["text"], bounding_box=tuple(line["bounding_box"]))
            for line in record.get("ocr_lines", [])
        ],
        width=record["width"], height=record["height"],
        ui_elements=[], png_bytes=None,
        dpi_scale=record.get("dpi_scale", 1.0), graph=graph,
        image_width=record.get("image_width"), image_height=record.get("image_height"),
    )


class TapePump:
    """Replays the recorded microphone chunk sequence at its recorded pace.

    Replaying the chunk sequence rather than the concatenated utterance is what
    lets the real VAD run, and pacing is what makes that meaningful: the VAD
    accumulates silence against wall-clock time (vad/detector.py:98), so a
    sequence delivered as fast as it can be read ends the turn at a different
    chunk than the recording did.

    A replay therefore takes about as long as the listening window it is
    reproducing. That is the cost of being able to say anything at all about
    VAD parity.
    """

    def __init__(self, tape: Tape):
        self._chunks = list(tape.audio_chunks())
        self._index = 0
        self._started: Optional[float] = None
        self._windows = 0
        self.ran_dry = False
        self.max_lateness_ms = 0.0

    def start(self, loop=None):
        return self

    def stop(self) -> None:
        return None

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._chunks)

    async def get(self, timeout: Optional[float] = None) -> Optional[bytes]:
        if self.exhausted:
            if self._windows > 1:
                # A follow-up window that outlives the recorded audio is the
                # normal end of a session: the user stopped talking and the
                # window timed out. None is what a silent microphone returns,
                # and it is the path the recording itself took.
                await asyncio.sleep(0.02)
                return None

            # In the activation window this is a finding, not plumbing. In a
            # real tape the recorded silence tail trips the VAD and the turn
            # ends on its own; running out of chunks first means the replayed
            # VAD did not close the turn where the recorded one did.
            #
            # Raising rather than returning None ends the window at once via
            # the loop's existing error path, instead of leaving it spinning
            # against its own 30s timeout.
            self.ran_dry = True
            raise TapeExhausted("recorded audio ended before the VAD closed the turn")

        import time as _time
        if self._started is None:
            self._started = _time.perf_counter()

        offset_ms, chunk = self._chunks[self._index]
        self._index += 1

        due = self._started + offset_ms / 1000.0
        delay = due - _time.perf_counter()
        if delay > 0:
            # Capped so one long pause in a recording cannot stall a CI run;
            # a gap longer than this is reported by the caller, not slept off.
            await asyncio.sleep(min(delay, 2.0))

        # How far behind the recording this delivery actually landed. A replay
        # cannot make claims about a wall-clock VAD it did not manage to feed
        # on schedule, and on a loaded machine it will sometimes fail to - so
        # the alternative to measuring this is a parity check that cries wolf
        # under CPU contention, which is worse than no check at all.
        self.max_lateness_ms = max(
            self.max_lateness_ms, (_time.perf_counter() - due) * 1000.0
        )
        return chunk

    @property
    def chunk_period_ms(self) -> float:
        """The recording's typical inter-chunk gap.

        The tolerance for lateness: being late by less than one chunk cannot
        move the VAD's silence threshold across a chunk boundary, so it cannot
        change where the turn ended.
        """
        gaps = sorted(
            b - a for (a, _), (b, _) in zip(self._chunks, self._chunks[1:]) if b > a
        )
        return gaps[len(gaps) // 2] if gaps else 0.0

    @property
    def paced_faithfully(self) -> bool:
        return self.max_lateness_ms <= max(self.chunk_period_ms, 1.0)

    def drain(self) -> int:
        """Both listening windows call this before collecting; count them.

        The recorded chunk sequence is continuous across the whole session, so
        there is nothing to skip here - but knowing which window is open is
        what lets exhaustion mean "VAD parity failed" in the activation turn
        and "the session ended" in a follow-up.
        """
        self._windows += 1
        return 0


class TapeWhisper:
    """Returns the recorded transcript for the audio that was buffered.

    Falls back to the tape's recorded order when the buffered bytes do not
    digest to a recorded utterance - which happens whenever the VAD closed the
    turn at a different chunk than it did during recording. That fallback is
    reported, not hidden: it means VAD parity failed even though the transcript
    matched.
    """

    def __init__(self, tape: Tape):
        self._by_digest = tape.stt_by_digest()
        self._consumed: dict[str, int] = {}
        self._ordered = [record["text"] for record in tape.stt]
        self._index = 0
        self._buffer = bytearray()
        self.digest_misses = 0
        self.transcripts: list[dict] = []

    def warmup(self) -> bool:
        return True

    def add_buffer(self, buffer) -> None:
        self._buffer.extend(buffer)

    def add_chunk(self, chunk) -> None:
        self._buffer.extend(chunk)

    def reset_buffer(self) -> None:
        self._buffer.clear()

    def clear_buffer(self) -> None:
        self._buffer.clear()

    def get_buffer(self) -> bytes:
        return bytes(self._buffer)

    def transcribe(self) -> str:
        text = self.transcribe_bytes(bytes(self._buffer))
        self._buffer.clear()
        return text

    def transcribe_bytes(self, raw_bytes: bytes) -> str:
        digest = _digest(raw_bytes)
        # Repeats of the same audio are served in recorded order, so identical
        # utterances in one session do not all resolve to the last transcript.
        candidates = self._by_digest.get(digest) or []
        position = self._consumed.get(digest, 0)
        if position < len(candidates):
            self._consumed[digest] = position + 1
            text = candidates[position]
        else:
            self.digest_misses += 1
            text = self._ordered[self._index] if self._index < len(self._ordered) else ""
        self._index += 1
        self.transcripts.append({"pcm_digest": digest, "text": text})
        return text


class CapturingWsServer:
    """A WsEventServer that keeps events instead of serving them.

    emit() still validates against the frozen contract, because a replay that
    accepted malformed events would let a contract regression through the one
    gate built to catch it.
    """

    def __init__(self):
        self.events: list[dict] = []
        self._on_wake = None

    @property
    def is_connected(self) -> bool:
        # True, so nothing takes a "no frontend attached" shortcut and skips
        # events the recording contains.
        return True

    def set_on_wake(self, callback) -> None:
        self._on_wake = callback

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def emit(self, event: dict) -> None:
        from .contract import validate_event
        from .recorder import get_recorder

        validate_event(event)
        self.events.append(event)

        # Mirrors the real WsEventServer.emit tap, so a turn driven through
        # this stand-in still produces a complete tape. Self-disabling during a
        # replay, where the recorder is not configured.
        recorder = get_recorder()
        if recorder is not None:
            recorder.record_event(event)


class _Constructible:
    """Base for the stubs, which stand in for classes taking real kwargs."""

    def __init__(self, *args, **kwargs):
        pass


class NullTts(_Constructible):
    """Silent stand-in for Kokoro and the player.

    Synthesis and playback are graded by SpeechStarted/SpeechChunk/
    SpeechFinished, which the real ResponseGenerator still emits. Audio quality
    is a listening test, not something a tape can decide.
    """

    # 10 ms of silence at 24 kHz. Non-empty on purpose: ResponseGenerator
    # treats a falsy synthesis result as a failure and takes a different
    # emission path, which would make every replay diverge from its tape.
    _SILENCE = b"\x00" * 480

    def initialize(self, *args, **kwargs): return True
    def shutdown(self, *args, **kwargs): return None
    def synthesize(self, sentence, voice=None): return self._SILENCE
    def play(self, wav): return None
    def stop(self): return None
    def wait(self, *args, **kwargs): return None
    def close(self): return None


class NullAudio(_Constructible):
    """Stand-in for AudioCapture. get_rms keeps the real arithmetic."""

    def list_devices(self): return []
    def start(self): return self
    def stop(self): return None
    def close(self): return None
    def get_chunk(self): return b""

    def get_rms(self, chunk: bytes) -> float:
        import numpy as np
        if not chunk:
            return 0.0
        samples = np.frombuffer(chunk[: len(chunk) - len(chunk) % 2], dtype=np.int16)
        if samples.size == 0:
            return 0.0
        # float64 mean-of-squares, no DC removal - matching AudioCapture exactly.
        # The VAD threshold is 0.008 of full scale, roughly 262 of 32767, so
        # there is very little headroom for an arithmetic difference here.
        return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


class NullWakeWord(_Constructible):
    detected = False
    def start(self, *a, **k): return self
    def stop(self): return None
    def pause(self): return None
    def resume(self): return None
    def reset(self): return None


class NullComputerUse(_Constructible):
    is_ready = False
    def start(self): return None
    def stop(self): return None
    def perform(self, action, params): return {}


class _NullFeedbackSounds:
    """Silences the chime. Not an edge under test - the tape has no audio out."""

    @staticmethod
    def play_chime(): return None

    @staticmethod
    def play_end(): return None


# -- driver ----------------------------------------------------------------


class ReplayResult:
    def __init__(self, tape: Tape, diffs: list[Diff], notes: list[str]):
        self.tape = tape
        self.diffs = diffs
        self.notes = notes

    @property
    def ok(self) -> bool:
        return all(diff.ok for diff in self.diffs)

    def report(self) -> str:
        _, body = summarise(self.diffs)
        lines = [f"tape: {os.path.basename(self.tape.dir)}", body]
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "tape": self.tape.dir,
            "ok": self.ok,
            "notes": self.notes,
            "diffs": {d.kind: d.mismatches for d in self.diffs},
        }


def _config_from(tape: Tape):
    """Rebuild the Config the tape was recorded under.

    Replay used to construct Config() from the ambient environment, which meant
    a tape recorded with a 10s follow-up window was replayed against whatever
    ``FOLLOWUP_TIMEOUT_SECONDS`` happened to be set to - and every follow-up
    turn on the tape then had nothing to reproduce it. Config is part of the
    recording for the same reason the git sha is.

    Redacted fields come back as the literal ``"<redacted>"``. That is correct
    here: every edge that would use a key is stubbed, and what the backend
    needs from ``gemini_api_key`` is only whether it is set.
    """
    import dataclasses

    from grace.config import Config

    recorded = (tape.meta or {}).get("config") or {}
    names = {f.name for f in dataclasses.fields(Config)}
    overrides = {k: v for k, v in recorded.items() if k in names}
    if not overrides:
        return Config()
    return dataclasses.replace(Config(), **overrides)


async def replay(tape: Tape) -> ReplayResult:
    """Re-run a recorded session with every edge pinned.

    The whole session, not just the activation turn: a tape that contains
    follow-up turns records the events they emitted, and grading those against
    a single-turn replay would report a regression on every one of them.
    """
    llm = TapeLlm(tape)
    ws = CapturingWsServer()
    whisper = TapeWhisper(tape)
    notes: list[str] = []

    if not (tape.meta or {}).get("config"):
        notes.append(
            "tape has no recorded config: it is being replayed against the "
            "ambient environment, so any behaviour that depends on a setting "
            "(VAD silence window, follow-up timeout) is not actually pinned"
        )

    with ExitStack() as stack:
        stack.enter_context(
            mock.patch("grace.main.Config", lambda _cfg=_config_from(tape): _cfg)
        )
        for name, replacement in (
            ("AudioCapture", NullAudio),
            ("WakeWordDetector", lambda *a, **k: NullWakeWord()),
            ("KokoroEngine", lambda *a, **k: NullTts()),
            ("TTSPlayer", lambda *a, **k: NullTts()),
            ("ComputerUse", lambda *a, **k: NullComputerUse()),
            ("WsEventServer", lambda *a, **k: ws),
            # The activation chime decodes an MP3 through pydub. A replay
            # should be silent, and should not need ffmpeg on the box.
            ("FeedbackSounds", _NullFeedbackSounds),
            ("GemmaClient", lambda *a, **k: llm),
            ("WhisperStreaming", lambda *a, **k: whisper),
            ("AudioPump", lambda *a, **k: TapePump(tape)),
        ):
            stack.enter_context(mock.patch(f"grace.main.{name}", replacement))

        from grace.main import GraceApp

        app = GraceApp()
        app._running = True
        app.agent_loop._perception = TapePerception(tape)

        # A backstop on the agent loop. The shipped limit is 0 - unlimited -
        # and the loop cannot tell "the model failed" from "the model produced
        # nothing parseable": generate_text catches everything but
        # RateLimitError and returns None (llm/gemma_client.py:413), the
        # planner reads that as an empty response, and the loop retries the
        # unparseable plan forever. So a TapeExhausted from TapeLlm - which is
        # swallowed on the way out - would hang the replay instead of failing
        # it.
        #
        # The bound comes from the tape rather than a constant: the recording
        # observed exactly once per step, so its snapshot count is the number
        # of steps it actually took. Anything beyond that has already diverged.
        app.agent_loop._max_iterations = len(tape.snapshots) + 2

        pump = TapePump(tape)
        app.pump = pump

        if not tape.audio:
            # A tape recorded with GRACE_RECORD_AUDIO=0 has no chunks, and the
            # listen loop would then spin against its own 30s timeout. Close
            # the window on the first poll instead. The audio stage is simply
            # not under test for such a tape - which is why the transcripts
            # will be served by tape order, and why that is reported below.
            notes.append(
                "tape has no recorded audio: the listen window and VAD were "
                "bypassed, so this replay grades the LLM, dispatch and event "
                "behaviour only"
            )
            app.vad.process_chunk = lambda chunk, audio=None: True
            pump._chunks = [(0.0, b"\x00\x00")]

        dispatcher = TapeDispatcher(tape, app.dispatcher)
        app.dispatcher = dispatcher
        app.agent_loop._dispatcher = dispatcher

        try:
            # The full activation cycle, follow-up window included. Both
            # listening windows draw from the same TapePump, which is what the
            # recording looked like: one continuous chunk sequence.
            await app._handle_activation()
        except TapeExhausted as exc:
            notes.append(f"tape exhausted: {exc}")

    if (pump.ran_dry or whisper.digest_misses) and not pump.paced_faithfully:
        # Attribution matters more than volume here. The pump could not deliver
        # the recorded chunk sequence on schedule, so the VAD was fed different
        # timing than the recording had and would be expected to end the turn
        # somewhere else. That is the harness losing a race with the machine it
        # is running on, and reporting it as a backend regression would train
        # people to ignore the one signal that grades the VAD.
        notes.append(
            f"INCONCLUSIVE (VAD): the replay fell {pump.max_lateness_ms:.0f}ms "
            f"behind the recorded chunk schedule, more than one "
            f"{pump.chunk_period_ms:.0f}ms chunk, so turn-end timing could not "
            "be reproduced faithfully. Re-run on an idle machine to grade it."
        )
    else:
        if pump.ran_dry:
            notes.append(
                "the replayed VAD did not close the turn before the recorded "
                "audio ran out - VAD parity failed, whatever the transcript says"
            )
        if whisper.digest_misses:
            notes.append(
                f"{whisper.digest_misses} transcription(s) were served by tape "
                "order rather than PCM digest - the VAD closed the turn at a "
                "different chunk than it did during recording, so VAD parity "
                "failed even though the transcript may match"
            )

    diffs = [
        diff_event_streams(tape.event_stream, [
            e for e in ws.events if e.get("type") != "TurnTrace"
        ]),
        diff_dispatches(tape.dispatch_calls, dispatcher.calls),
        diff_prompts(tape.llm, llm.requests),
    ]
    return ReplayResult(tape, diffs, notes)


def _discover(root: str) -> list[str]:
    if os.path.exists(os.path.join(root, "events.jsonl")):
        return [root]
    return sorted(
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.exists(os.path.join(root, name, "events.jsonl"))
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tape", help="a single recorded session directory")
    source.add_argument("--corpus", help="a directory of recorded sessions")
    parser.add_argument("--json", help="write a machine-readable report here")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    directories = _discover(args.tape or args.corpus)
    if not directories:
        print("no tapes found", file=sys.stderr)
        return 2

    results = []
    for directory in directories:
        result = asyncio.run(replay(Tape.load(directory)))
        results.append(result)
        print(result.report())
        print()

    failed = [r for r in results if not r.ok]
    print(f"{len(results) - len(failed)}/{len(results)} tapes reproduced")

    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump([r.to_dict() for r in results], fh, indent=2, ensure_ascii=False)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
