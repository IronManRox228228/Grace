"""Per-turn latency instrumentation for Grace.

Grace handles exactly one user turn at a time, so a single module-level
"current trace" is enough to let deeply nested code (perception, TTS, the
agent loop) record timings without threading a trace object through every
call signature. Work dispatched to worker threads via ``asyncio.to_thread``
sees the same trace, which is what we want.

Usage::

    with start_turn("wake") as trace:
        with trace.stage("whisper"):
            transcript = whisper.transcribe()
        async with trace.stage("planner"):
            step = await planner.next_step()
        trace.mark_event("first_audio_out")
    log.info(trace.summary())

Every stage is cheap (two ``perf_counter`` calls) and the whole module is a
no-op when no turn is active, so instrumentation left in the hot path costs
nothing measurable.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("grace.timing")

# Grace processes one turn at a time; a guarded module global is simpler and
# more thread-friendly here than a ContextVar (which would not propagate into
# to_thread workers).
_current_lock = threading.Lock()
_current: Optional["TurnTrace"] = None


@dataclass
class StageRecord:
    """A single completed (or in-flight) timed stage."""

    name: str
    start: float
    depth: int
    duration_ms: Optional[float] = None
    detail: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.duration_ms is None


class Stage:
    """Context manager timing one stage. Works with both ``with`` and ``async with``."""

    __slots__ = ("_trace", "_record")

    def __init__(self, trace: "TurnTrace", name: str):
        self._trace = trace
        self._record = trace._open_stage(name)

    def detail(self, text: str) -> "Stage":
        """Attach a short note to this stage (e.g. element counts, token counts)."""
        self._record.detail = text
        return self

    def _close(self) -> None:
        self._trace._close_stage(self._record)

    def __enter__(self) -> "Stage":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._close()
        return False

    async def __aenter__(self) -> "Stage":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self._close()
        return False


@dataclass
class TurnTrace:
    """Timing record for a single user turn."""

    label: str = "turn"
    start: float = field(default_factory=time.perf_counter)
    stages: list[StageRecord] = field(default_factory=list)
    events: dict[str, float] = field(default_factory=dict)
    _depth: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- recording -------------------------------------------------------

    def stage(self, name: str) -> Stage:
        """Time a block of work. Nesting is tracked and shown in the summary."""
        return Stage(self, name)

    def mark(self, name: str, duration_ms: float, detail: Optional[str] = None) -> None:
        """Record a stage whose duration was measured elsewhere."""
        with self._lock:
            self.stages.append(
                StageRecord(name=name, start=time.perf_counter(), depth=self._depth,
                            duration_ms=duration_ms, detail=detail)
            )

    def mark_event(self, name: str) -> float:
        """Record a point-in-time milestone as ms since the turn started.

        Used for things like time-to-first-audio, where what matters is the
        offset from the start of the turn rather than a span.
        """
        offset_ms = (time.perf_counter() - self.start) * 1000.0
        with self._lock:
            # First occurrence wins - "first audio out" should not be
            # overwritten by the second sentence.
            self.events.setdefault(name, offset_ms)
        return offset_ms

    def _open_stage(self, name: str) -> StageRecord:
        with self._lock:
            record = StageRecord(name=name, start=time.perf_counter(), depth=self._depth)
            self._depth += 1
            self.stages.append(record)
            return record

    def _close_stage(self, record: StageRecord) -> None:
        with self._lock:
            if record.duration_ms is None:
                record.duration_ms = (time.perf_counter() - record.start) * 1000.0
            self._depth = max(0, self._depth - 1)

    # -- reporting -------------------------------------------------------

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "label": self.label,
                "total_ms": round(self.total_ms, 1),
                "stages": [
                    {
                        "name": s.name,
                        "depth": s.depth,
                        "ms": round(s.duration_ms, 1) if s.duration_ms is not None else None,
                        "detail": s.detail,
                    }
                    for s in self.stages
                ],
                "events": {k: round(v, 1) for k, v in self.events.items()},
            }

    def summary(self) -> str:
        """One multi-line block suitable for a single INFO log call."""
        lines = [f"--- turn trace '{self.label}' total={self.total_ms:.0f}ms ---"]
        with self._lock:
            stages = list(self.stages)
            events = dict(self.events)
        for s in stages:
            indent = "  " * s.depth
            ms = "  (open)" if s.duration_ms is None else f"{s.duration_ms:8.1f}ms"
            detail = f"  [{s.detail}]" if s.detail else ""
            lines.append(f"  {ms}  {indent}{s.name}{detail}")
        for name, offset in events.items():
            lines.append(f"  @{offset:7.1f}ms  {name}")
        return "\n".join(lines)


class _NullStage:
    """Zero-cost stand-in used when no turn is active."""

    __slots__ = ()

    def detail(self, text: str) -> "_NullStage":
        return self

    def __enter__(self) -> "_NullStage":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    async def __aenter__(self) -> "_NullStage":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


_NULL_STAGE = _NullStage()


class _TurnContext:
    """Context manager returned by :func:`start_turn`."""

    __slots__ = ("trace",)

    def __init__(self, trace: TurnTrace):
        self.trace = trace

    def __enter__(self) -> TurnTrace:
        return self.trace

    def __exit__(self, exc_type, exc, tb) -> bool:
        end_turn()
        return False


def start_turn(label: str = "turn") -> _TurnContext:
    """Begin a new turn trace and install it as the current one."""
    global _current
    trace = TurnTrace(label=label)
    with _current_lock:
        _current = trace
    return _TurnContext(trace)


def current_trace() -> Optional[TurnTrace]:
    """Return the in-flight trace, or None when no turn is active."""
    with _current_lock:
        return _current


def end_turn() -> Optional[TurnTrace]:
    """Clear the current trace and return it."""
    global _current
    with _current_lock:
        trace = _current
        _current = None
    return trace


def stage(name: str):
    """Time a block against the current turn, or do nothing if there isn't one.

    This is the form used inside library code that may run either inside a
    turn or standalone (tests, warmup, the setup scripts).
    """
    trace = current_trace()
    return trace.stage(name) if trace is not None else _NULL_STAGE


def mark(name: str, duration_ms: float, detail: Optional[str] = None) -> None:
    """Record an externally measured duration against the current turn."""
    trace = current_trace()
    if trace is not None:
        trace.mark(name, duration_ms, detail)


def mark_event(name: str) -> None:
    """Record a milestone against the current turn."""
    trace = current_trace()
    if trace is not None:
        trace.mark_event(name)
