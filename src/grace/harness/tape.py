"""Reading session tapes, and diffing a run against one.

A tape is the migration's oracle. This module is deliberately implementation-
agnostic: it knows how to read a recording and how to compare two event
streams, and nothing about Python or Rust. The Rust replay driver produces the
same JSONL shapes and is graded by the same comparison rules.

Diffing is not uniform, because "identical" is the wrong bar for some of this:

* **Byte-exact**: the event sequence, LLM prompt strings, and tool dispatches.
  These are the contract and the behaviour. A difference here is a regression.
* **Semantic**: transcripts and OCR text. Two speech models will not agree
  character-for-character, and demanding they do would either block the
  migration or force the wrong model.
* **Budgeted**: timings. Rust should mostly be faster; the check exists to
  catch a port that accidentally serialised something that used to be
  pipelined, not to pin a millisecond count.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

# Diagnostic-only events. Present in tapes, excluded from event-sequence
# comparison: TurnTrace carries wall-clock durations that legitimately differ
# on every run, and the renderer ignores it.
NON_BEHAVIORAL_EVENTS = frozenset({"TurnTrace"})


@dataclass
class Tape:
    """One recorded session, loaded from a directory written by Recorder."""

    dir: str
    meta: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    llm: list[dict] = field(default_factory=list)
    stt: list[dict] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)
    dispatch: list[dict] = field(default_factory=list)
    audio: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, directory: str) -> "Tape":
        tape = cls(dir=directory)
        meta_path = os.path.join(directory, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                tape.meta = json.load(fh)

        for stream in ("events", "llm", "stt", "snapshots", "dispatch", "audio"):
            setattr(tape, stream, list(_read_jsonl(os.path.join(directory, f"{stream}.jsonl"))))
        return tape

    # -- views used by a replay driver -------------------------------------

    @property
    def event_stream(self) -> list[dict]:
        """The recorded events, in order, excluding diagnostics."""
        return [
            record["event"]
            for record in self.events
            if record["event"].get("type") not in NON_BEHAVIORAL_EVENTS
        ]

    @property
    def dispatch_calls(self) -> list[dict]:
        return [{"tool": r["tool"], "params": r["params"]} for r in self.dispatch]

    def llm_by_digest(self) -> dict[str, list[dict]]:
        """Recorded LLM exchanges keyed by request digest.

        A list per digest, because the same prompt legitimately recurs within a
        turn (the agent loop re-plans against an unchanged screen) and replay
        must hand back the responses in the order they were recorded.
        """
        index: dict[str, list[dict]] = {}
        for record in self.llm:
            index.setdefault(record["request_digest"], []).append(record)
        return index

    def stt_by_digest(self) -> dict[str, list[str]]:
        """Recorded transcripts keyed by PCM digest, in order.

        A list per digest, for the same reason ``llm_by_digest`` returns one:
        the same audio can legitimately be transcribed twice in a session, and
        a flat dict silently keeps only the last transcript - so the *first*
        lookup then returns the *last* turn's text. That is a wrong-answer
        failure, not a missing-data one, which makes it exactly the kind a
        replay would otherwise report as a backend regression.
        """
        index: dict[str, list[str]] = {}
        for record in self.stt:
            index.setdefault(record["pcm_digest"], []).append(record["text"])
        return index

    def pcm(self, digest: str) -> bytes:
        with open(os.path.join(self.dir, "pcm", f"{digest}.raw"), "rb") as fh:
            return fh.read()

    def audio_chunks(self) -> Iterator[tuple[float, bytes]]:
        """The mic chunk sequence with each chunk's arrival offset in ms.

        The offsets are not decoration. VadDetector accumulates silence against
        ``time.time()`` (detector.py:98), not against a sample count, so
        turn-end depends on how fast chunks actually arrived. Replaying the
        sequence as fast as it can be read would end the turn at a different
        chunk than the recording did, and any VAD comparison built on that
        would be meaningless.

        This is also a live constraint on the Rust port: a wall-clock VAD has
        to stay wall-clock, or turn-end timing shifts for every user.
        """
        path = os.path.join(self.dir, "audio.raw")
        if not self.audio or not os.path.exists(path):
            return
        with open(path, "rb") as fh:
            for record in self.audio:
                fh.seek(record["offset"])
                yield record["t_ms"], fh.read(record["bytes"])

    def stage_budget(self, percentile: float = 0.95) -> dict[str, float]:
        """Per-stage duration budget derived from the tape's own TurnTraces."""
        durations: dict[str, list[float]] = {}
        for record in self.events:
            event = record["event"]
            if event.get("type") != "TurnTrace":
                continue
            for stage in event["trace"]["stages"]:
                if stage["ms"] is not None:
                    durations.setdefault(stage["name"], []).append(stage["ms"])

        budget = {}
        for name, values in durations.items():
            values.sort()
            index = min(len(values) - 1, int(len(values) * percentile))
            budget[name] = values[index]
        return budget


def _read_jsonl(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# -- diffing ---------------------------------------------------------------


@dataclass
class Diff:
    """The result of comparing one run against a tape."""

    kind: str
    mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def report(self) -> str:
        if self.ok:
            return f"{self.kind}: OK"
        lines = [f"{self.kind}: {len(self.mismatches)} mismatch(es)"]
        lines.extend(f"  {m}" for m in self.mismatches)
        return "\n".join(lines)


def diff_event_streams(expected: list[dict], actual: list[dict]) -> Diff:
    """Byte-exact comparison of two event sequences.

    Order matters and is the point: the UI is a state machine driven by this
    sequence, and main.py's deliberate pauses mean "same events, different
    order" is a different user experience, not a rounding error.
    """
    diff = Diff(kind="events")
    for index in range(max(len(expected), len(actual))):
        want = expected[index] if index < len(expected) else None
        got = actual[index] if index < len(actual) else None
        if want == got:
            continue
        if want is None:
            diff.mismatches.append(f"[{index}] unexpected extra event: {_brief(got)}")
        elif got is None:
            diff.mismatches.append(f"[{index}] missing event: {_brief(want)}")
        elif want.get("type") != got.get("type"):
            diff.mismatches.append(
                f"[{index}] type: expected {want.get('type')!r}, got {got.get('type')!r}"
            )
        else:
            diff.mismatches.append(
                f"[{index}] {want.get('type')} payload: expected {_brief(want)}, got {_brief(got)}"
            )
    return diff


def diff_dispatches(expected: list[dict], actual: list[dict]) -> Diff:
    """Byte-exact comparison of the outbound side-effect sequence.

    This is what catches "Grace ended up in the right place on screen but took
    a different route to get there" - a run that produces the correct final
    state through different actions has not reproduced the behaviour.
    """
    diff = Diff(kind="dispatch")
    for index in range(max(len(expected), len(actual))):
        want = expected[index] if index < len(expected) else None
        got = actual[index] if index < len(actual) else None
        if want == got:
            continue
        if want is None:
            diff.mismatches.append(f"[{index}] unexpected extra call: {_brief(got)}")
        elif got is None:
            diff.mismatches.append(f"[{index}] missing call: {_brief(want)}")
        else:
            diff.mismatches.append(
                f"[{index}] expected {want['tool']}({want['params']}), "
                f"got {got['tool']}({got['params']})"
            )
    return diff


def diff_prompts(expected: list[dict], actual: list[dict]) -> Diff:
    """Byte-exact comparison of LLM request payloads.

    Prompt-string equality across two string and JSON stacks is one of the
    migration's hardest gates, and it has to be exact: a stray newline or a
    reordered tool list changes what the model does, silently and only
    sometimes.
    """
    diff = Diff(kind="prompts")
    for index in range(max(len(expected), len(actual))):
        want = expected[index] if index < len(expected) else None
        got = actual[index] if index < len(actual) else None
        if want is None:
            diff.mismatches.append(f"[{index}] unexpected extra LLM request")
            continue
        if got is None:
            diff.mismatches.append(f"[{index}] missing LLM request")
            continue
        if want.get("request_digest") == got.get("request_digest"):
            continue
        diff.mismatches.append(
            f"[{index}] prompt differs:\n"
            + _first_difference(
                _prompt_fields(want.get("request")), _prompt_fields(got.get("request"))
            )
        )
    return diff


def _prompt_fields(request: Any) -> Any:
    """Project a request onto the fields the digest actually covers.

    Rendering the whole request here pointed readers at ``backend`` and
    ``model`` - which differ by construction between a Python recording and any
    other driver, and which the digest deliberately excludes. The first
    reported difference has to be one that mattered, or the diagnostic sends
    people chasing a field that was never compared.
    """
    from .recorder import LLM_REQUEST_KEY_FIELDS

    if not isinstance(request, dict):
        return request
    return {field: request.get(field) for field in LLM_REQUEST_KEY_FIELDS}


def diff_transcripts(
    expected: list[dict], actual: list[dict], *, tolerance: float = 0.98
) -> Diff:
    """Semantic comparison of transcripts, with an exact-match rate floor.

    Two speech models will not agree character-for-character and demanding they
    do would block the migration for no user-visible benefit. What is NOT
    negotiable is the subset of utterances that change routing: see
    ``routing_relevant`` in the replay driver - a transcript diff that flips
    CapabilityRouter.classify is a blocker, not a rounding error.
    """
    diff = Diff(kind="transcripts")
    if not expected:
        return diff

    exact = 0
    for index, want in enumerate(expected):
        got = actual[index] if index < len(actual) else None
        if got is None:
            diff.mismatches.append(f"[{index}] missing transcript for {want['pcm_digest']}")
            continue
        if _normalise_text(want["text"]) == _normalise_text(got["text"]):
            exact += 1
        else:
            diff.mismatches.append(
                f"[{index}] transcript: expected {want['text']!r}, got {got['text']!r}"
            )

    rate = exact / len(expected)
    if rate >= tolerance:
        # Under the floor the individual diffs are informational, not failures.
        diff.mismatches = []
    else:
        diff.mismatches.insert(
            0, f"exact-match rate {rate:.1%} is below the {tolerance:.0%} floor"
        )
    return diff


def diff_stage_timings(
    budget: dict[str, float], observed: dict[str, float], *, headroom: float = 1.2
) -> Diff:
    """Budget check, not an equality check.

    Rust should mostly be faster. This exists to catch a port that accidentally
    serialised something that used to be pipelined - losing the one-deep TTS
    lookahead, for instance, is invisible in an event diff but audible as a gap
    between every pair of sentences.
    """
    diff = Diff(kind="timings")
    for name, ceiling in budget.items():
        actual = observed.get(name)
        if actual is None:
            continue
        limit = ceiling * headroom
        if actual > limit:
            diff.mismatches.append(
                f"{name}: {actual:.0f}ms exceeds budget {ceiling:.0f}ms +{headroom:.0%} "
                f"({limit:.0f}ms)"
            )
    return diff


def _normalise_text(text: str) -> str:
    """Casing, surrounding space, and terminal punctuation are not behaviour."""
    return " ".join(text.lower().strip().strip(".!?,").split())


def _brief(value: Any, limit: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _first_difference(want: Any, got: Any) -> str:
    """Point at the first differing character of two prompts.

    A prompt is thousands of characters; a full diff buries the one that
    matters. This prints the offset and a window of context on each side.
    """
    left = json.dumps(want, ensure_ascii=False, sort_keys=True, indent=2)
    right = json.dumps(got, ensure_ascii=False, sort_keys=True, indent=2)

    limit = min(len(left), len(right))
    offset = next((i for i in range(limit) if left[i] != right[i]), limit)
    start = max(0, offset - 60)

    return (
        f"    first difference at char {offset}\n"
        f"    expected: …{left[start:offset + 60]!r}\n"
        f"    actual:   …{right[start:offset + 60]!r}"
    )


def summarise(diffs: list[Diff]) -> tuple[bool, str]:
    """Fold per-stream diffs into an overall pass/fail plus a printable report."""
    ok = all(diff.ok for diff in diffs)
    report = "\n".join(diff.report() for diff in diffs)
    return ok, report
