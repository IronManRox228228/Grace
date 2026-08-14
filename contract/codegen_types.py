"""Generate frontend/renderer/src/state/types.ts from grace-events.schema.json.

The hand-written TypeScript union drifted from what the backend actually
emitted (it declared PartialTranscript and Error, which were never sent, and
was missing FollowupListeningStarted and TurnTrace, which were). Generating it
removes the only place that drift could live.

Usage::

    python contract/codegen_types.py           # write the file
    python contract/codegen_types.py --check   # exit 1 if it would change

``--check`` is the CI gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, "grace-events.schema.json")
TYPES_PATH = os.path.join(
    HERE, "..", "frontend", "renderer", "src", "state", "types.ts"
)

BANNER = """// GENERATED FILE - DO NOT EDIT.
// Source: contract/grace-events.schema.json
// Regenerate: python contract/codegen_types.py
//
// The backend (Python today, Rust after the migration) and this file are two
// views of one frozen contract. Adding a variant here without adding it to the
// schema will be reverted by the next codegen run; CI runs --check.
"""

# Hand-maintained tail: application state derived FROM the contract, which the
# schema deliberately says nothing about. Kept verbatim from the original file.
TAIL = """
export type GraceState =
  | 'idle'
  | 'listening'
  | 'understanding'
  | 'executing'
  | 'speaking'
  | 'completed'
  | 'error';

export interface GraceSnapshot {
  state: GraceState;
  userTranscript: string; // live/partial + final user speech
  statusLabel: string; // "Understanding request…", "Opening Microsoft Edge…"
  responseText: string; // streamed assistant response
  errorMessage?: string;
}

export const INITIAL_SNAPSHOT: GraceSnapshot = {
  state: 'idle',
  userTranscript: '',
  statusLabel: '',
  responseText: '',
};
"""


def _ts_type(node: dict) -> str:
    """Map a JSON Schema property node onto a TypeScript type expression."""
    if "const" in node:
        return f"'{node['const']}'"

    kind = node.get("type")
    if isinstance(kind, list):
        return " | ".join(_ts_type({**node, "type": k}) for k in kind)

    if kind == "string":
        return "string"
    if kind in ("number", "integer"):
        return "number"
    if kind == "boolean":
        return "boolean"
    if kind == "null":
        return "null"
    if kind == "array":
        return f"{_ts_type(node['items'])}[]"
    if kind == "object":
        props = node.get("properties")
        if props:
            required = set(node.get("required", []))
            fields = ", ".join(
                f"{name}{'' if name in required else '?'}: {_ts_type(sub)}"
                for name, sub in props.items()
            )
            return "{ " + fields + " }"
        extra = node.get("additionalProperties")
        if isinstance(extra, dict):
            return f"Record<string, {_ts_type(extra)}>"
        return "Record<string, unknown>"

    return "unknown"


def _first_line(text: str, limit: int = 96) -> str:
    """The first sentence of a description, for a one-line TS comment."""
    head = text.split("\n", 1)[0].strip()
    if ". " in head:
        head = head.split(". ", 1)[0] + "."
    if len(head) > limit:
        head = head[: limit - 1].rstrip() + "…"
    return head


def _render_variant(definition: dict) -> tuple[str, str]:
    """Return ``(type expression, trailing comment)`` for one variant."""
    props: dict = definition["properties"]
    required = set(definition.get("required", []))

    fields = []
    for prop_name, node in props.items():
        optional = "" if prop_name in required else "?"
        fields.append(f"{prop_name}{optional}: {_ts_type(node)}")

    body = "{ " + "; ".join(fields) + " }"
    description = definition.get("description", "")
    comment = f" // {_first_line(description)}" if description else ""
    return body, comment


def render() -> str:
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        schema = json.load(fh)

    order = [ref["$ref"].rsplit("/", 1)[-1] for ref in schema["oneOf"]]
    definitions = schema["definitions"]
    variants = [_render_variant(definitions[name]) for name in order]

    lines = [BANNER, "export type GraceEvent ="]
    for index, (body, comment) in enumerate(variants):
        # The terminating semicolon belongs to the type expression, not the
        # line - putting it last would bury it inside the trailing comment.
        terminator = ";" if index == len(variants) - 1 else ""
        lines.append(f"  | {body}{terminator}{comment}")

    return "\n".join(lines) + "\n" + TAIL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the generated output differs from the file on disk",
    )
    args = parser.parse_args()

    generated = render()
    path = os.path.abspath(TYPES_PATH)

    if args.check:
        try:
            with open(path, encoding="utf-8") as fh:
                current = fh.read()
        except FileNotFoundError:
            print(f"MISSING: {path}", file=sys.stderr)
            return 1
        if current != generated:
            print(
                f"OUT OF DATE: {path}\n"
                "Run: python contract/codegen_types.py",
                file=sys.stderr,
            )
            return 1
        print(f"up to date: {path}")
        return 0

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(generated)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
