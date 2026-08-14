"""Migration harness: contract enforcement and record/replay.

Everything in this package is inert unless explicitly switched on by an
environment variable, so it costs nothing in normal operation:

``GRACE_CONTRACT_STRICT=1``
    Raise instead of log when an emitted event violates the frozen contract in
    ``contract/grace-events.schema.json``. On in dev and CI; off in production,
    where a contract bug must never take down a turn.

``GRACE_RECORD_DIR=<path>``
    Record a session tape - every event, LLM exchange, transcription, snapshot
    and tool dispatch - for later replay against the Rust port.

The point of both is to make "no regressions" a mechanically checkable claim
rather than an aspiration. See the migration plan, Phase 0.

Two entry points, both CLIs::

    python -m grace.harness.generate --out corpus/   # scripted tapes
    python -m grace.harness.replay --corpus corpus/  # the regression gate

``generate`` is deliberately not imported here: it patches production classes
and is a tool, not a runtime dependency.
"""

from .contract import ContractViolation, validate_event
from .recorder import Recorder, get_recorder, install_recorder
from .tape import (
    Diff,
    Tape,
    diff_dispatches,
    diff_event_streams,
    diff_prompts,
    diff_stage_timings,
    diff_transcripts,
    summarise,
)

__all__ = [
    "ContractViolation",
    "validate_event",
    "Recorder",
    "get_recorder",
    "install_recorder",
    "Tape",
    "Diff",
    "diff_event_streams",
    "diff_dispatches",
    "diff_prompts",
    "diff_transcripts",
    "diff_stage_timings",
    "summarise",
]
