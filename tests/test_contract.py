"""Conformance tests for the frozen backend -> frontend event contract.

These are the tests that make the migration checkable. They pin three things:

1. Every event shape the backend actually emits validates against the schema.
2. Malformed variants are rejected - especially closed-object violations,
   because an unnoticed extra field is exactly how the contract drifted before.
3. The generated types.ts is in sync with the schema.

Rule for the whole migration: the contract is additive-only, and only for
genuinely new UI features. If a port cannot reproduce an existing event at the
right moment, that is a bug in the port, not a reason to edit the schema.
"""

import json
import os
import subprocess
import sys

import pytest

from grace.harness.contract import (
    SCHEMA_PATH,
    ContractViolation,
    contract_variants,
    validate_event,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Every distinct (type, key-set) shape emitted by the backend today, with the
# call site that produces it. Adding an emit() call to the backend without
# adding its shape here should fail review; adding it here without adding it to
# the schema fails this file.
EMITTED_SHAPES = [
    ("main.py:586,593,738", {"type": "Idle"}),
    ("main.py:531", {"type": "WakeWordDetected"}),
    ("main.py:534", {"type": "ListeningStarted"}),
    ("main.py:697", {"type": "FollowupListeningStarted", "timeout": 10}),
    ("main.py:597,743", {"type": "FinalTranscript", "text": "open notepad"}),
    ("main.py:598", {"type": "ListeningStopped"}),
    ("main.py:607", {"type": "UnderstandingStarted", "label": "Understanding request…"}),
    ("main.py:617,635", {"type": "UnderstandingFinished"}),
    ("dispatcher.py:182,209", {"type": "ToolExecutionStarted", "label": "Opening Notepad…"}),
    (
        "loop.py:211",
        {"type": "ToolExecutionStarted", "label": "Clicking Send…", "tool": "cua_click", "step": 3},
    ),
    ("dispatcher.py:186", {"type": "ToolExecutionFinished"}),
    ("loop.py:237", {"type": "ToolExecutionFinished", "tool": "cua_click", "status": "ok"}),
    ("generator.py:127,142,218", {"type": "ResponseChunk", "text": " Sure, opening that now."}),
    ("generator.py:95,190", {"type": "SpeechStarted"}),
    ("generator.py:128,143,219", {"type": "SpeechChunk"}),
    ("generator.py:150,228", {"type": "SpeechFinished"}),
    ("main.py:585,591,603,682", {"type": "ConversationFinished"}),
    (
        "main.py:520",
        {
            "type": "TurnTrace",
            "trace": {
                "label": "wake",
                "total_ms": 4210.5,
                "stages": [
                    {"name": "whisper", "depth": 0, "ms": 812.4, "detail": None},
                    {"name": "ground#1", "depth": 1, "ms": None, "detail": "24 elements"},
                ],
                "events": {"first_audio_out": 1904.2},
            },
        },
    ),
    ("reserved for the Rust port", {"type": "Error", "message": "TTS sidecar unavailable"}),
]


@pytest.mark.parametrize(
    "event", [shape for _, shape in EMITTED_SHAPES], ids=[site for site, _ in EMITTED_SHAPES]
)
def test_emitted_shape_satisfies_contract(event):
    validate_event(event)


def test_contract_has_no_variant_the_backend_never_emits():
    """Dead contract surface is where drift hides.

    The previous hand-written union declared PartialTranscript, which nothing
    ever sent, and the renderer carried a reducer case for it for two releases.
    Error is the single sanctioned exception: it is reserved for the Rust
    port's sidecar-failure paths and the renderer already handles it.
    """
    emitted = {shape["type"] for _, shape in EMITTED_SHAPES}
    assert contract_variants() == emitted


@pytest.mark.parametrize(
    "event, reason",
    [
        ({"type": "PartialTranscript", "text": "x"}, "variant was removed from the contract"),
        ({"type": "FinalTranscript"}, "missing required 'text'"),
        ({"type": "UnderstandingStarted"}, "missing required 'label'"),
        ({"type": "FollowupListeningStarted", "timeout": "10"}, "timeout must be a number"),
        ({"type": "ToolExecutionStarted", "label": "x", "bogus": 1}, "closed object"),
        ({"type": "ToolExecutionStarted", "label": "x", "step": 0}, "step is 1-based"),
        ({"type": "ResponseChunk", "text": None}, "text must be a string"),
        ({"type": "Idle", "extra": True}, "closed object"),
        ({"type": "Error", "message": 5}, "message must be a string"),
        ({"type": "Nope"}, "unknown discriminator"),
        ({"no_type": 1}, "no discriminator at all"),
        ("Idle", "not an object"),
    ],
)
def test_malformed_events_are_rejected(event, reason):
    with pytest.raises(ContractViolation):
        validate_event(event)


def test_nested_turn_trace_is_validated_not_waved_through():
    """TurnTrace carries a nested object; a closed-object bug there must fail."""
    bad = {
        "type": "TurnTrace",
        "trace": {"label": "wake", "total_ms": 1.0, "stages": [], "events": {}, "extra": 1},
    }
    with pytest.raises(ContractViolation):
        validate_event(bad)

    bad_stage = {
        "type": "TurnTrace",
        "trace": {
            "label": "wake",
            "total_ms": 1.0,
            "stages": [{"name": "whisper", "depth": 0, "ms": 1.0}],
            "events": {},
        },
    }
    with pytest.raises(ContractViolation):
        validate_event(bad_stage)


def test_boolean_is_not_accepted_as_a_number():
    """bool subclasses int in Python; the contract never means True by 'number'."""
    with pytest.raises(ContractViolation):
        validate_event({"type": "FollowupListeningStarted", "timeout": True})


def test_schema_variant_names_match_their_type_const():
    """The definition key and the 'type' const must agree, or codegen ordering
    silently produces a union that does not match runtime validation."""
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        schema = json.load(fh)
    for name, definition in schema["definitions"].items():
        assert definition["properties"]["type"]["const"] == name


def test_generated_types_ts_is_up_to_date():
    """CI gate: types.ts is generated, never hand-edited."""
    result = subprocess.run(
        [sys.executable, os.path.join("contract", "codegen_types.py"), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "frontend/renderer/src/state/types.ts is out of date with "
        "contract/grace-events.schema.json.\n"
        "Run: python contract/codegen_types.py\n" + result.stdout + result.stderr
    )
