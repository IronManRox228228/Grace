"""Every `emit()` in the backend must have a contract entry, found by reading the code.

`tests/test_contract.py` validates a hand-maintained list, `EMITTED_SHAPES`,
whose keys are *source line numbers* - "main.py:586,593,738". Those went stale
the first time anything above them moved, and more importantly the list can only
be wrong in the silent direction: an `emit()` added to the backend and not added
to the list produces no failure anywhere. The contract is then out of date and
every test still passes, which is the exact condition the contract exists to
prevent.

This walks the AST of the backend instead and finds the emit call sites itself,
so a new event cannot be introduced without either appearing in the schema or
failing here. It is the enforcement `contract/README.md` claims and nothing
performed.

Only the `type` is checked, not the full key set: keys are frequently built
from variables that no static read can resolve, and a test that pretends
otherwise would either be wrong or be a parser. The full shapes stay in
test_contract.py, which validates them against the schema.
"""

import ast
import json
import os

import pytest

from grace.harness.contract import SCHEMA_PATH

pytestmark = pytest.mark.integration

BACKEND = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "grace")
)

# Files whose emit() calls are the harness re-emitting a recording rather than
# the backend producing an event of its own.
SKIP = {os.path.join("harness", "replay.py"), os.path.join("harness", "generate.py")}

# Declared in the schema, deliberately not emitted by Python. `contract/README.md`
# gives the reason: the renderer has always handled `Error`, and the Rust port
# needs it for the sidecar-unavailable and supervisor-failure paths that Python
# has no equivalent of. Listed here rather than allowed silently, so that a
# *second* unemitted event has to be justified rather than blending in.
RESERVED = {"Error"}


def schema_event_types() -> set[str]:
    with open(SCHEMA_PATH, encoding="utf-8") as handle:
        schema = json.load(handle)

    types: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            const = node.get("const")
            if isinstance(const, str) and node is not schema:
                types.add(const)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema.get("definitions") or schema.get("$defs") or schema)
    return types


def emitted_types() -> dict[str, list[str]]:
    """{event type: [where it is emitted]}, read out of the source."""
    found: dict[str, list[str]] = {}

    for root, _, files in os.walk(BACKEND):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            relative = os.path.relpath(path, BACKEND)
            if relative in SKIP:
                continue

            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            if ".emit(" not in source:
                continue

            for node in ast.walk(ast.parse(source, filename=path)):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr == "emit"):
                    continue
                if not node.args:
                    continue
                event_type = _literal_type(node.args[0])
                if event_type:
                    found.setdefault(event_type, []).append(f"{relative}:{node.lineno}")

    return found


def _literal_type(node) -> str:
    """The `type` value of a dict literal argument, if it is a plain string."""
    if not isinstance(node, ast.Dict):
        return ""
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and key.value == "type":
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
    return ""


class TestEmitCallSites:
    def test_the_ast_walk_finds_the_events_we_know_exist(self):
        # A guard on the guard. If the walk silently stopped finding anything -
        # a moved directory, a renamed method - every other assertion here
        # would pass vacuously, which is the failure mode this whole file
        # exists to replace.
        found = emitted_types()
        assert {"WakeWordDetected", "FinalTranscript", "Idle"} <= set(found), (
            f"the AST walk found only {sorted(found)}"
        )

    def test_every_emitted_event_is_in_the_schema(self):
        declared = schema_event_types()
        undeclared = {
            event: sites for event, sites in emitted_types().items()
            if event not in declared
        }
        assert undeclared == {}, (
            "these events are emitted by the backend but are not in the "
            f"contract schema: {undeclared}. The renderer's reducer will not "
            "know what to do with them."
        )

    def test_the_schema_declares_nothing_the_backend_never_sends(self):
        # The drift Phase 0 found ran in this direction too: types.ts declared
        # PartialTranscript and Error, which Python has never emitted. A
        # renderer branch for an event that cannot arrive is dead code that
        # reads like a supported feature.
        emitted = set(emitted_types())
        phantom = schema_event_types() - emitted - RESERVED
        assert phantom == set(), (
            f"the schema declares events nothing emits: {sorted(phantom)}"
        )
