"""Runtime validation of emitted events against the frozen event contract.

This is a deliberately small, dependency-free JSON Schema subset validator
rather than a `jsonschema` dependency, for three reasons:

1. It runs on every emitted event, on the turn's hot path.
2. It must keep working after the Python backend is deleted - the same ~120
   lines port directly to Rust, where the alternative is a much heavier
   schema crate for a schema we control.
3. The subset actually used by contract/grace-events.schema.json is tiny:
   a top-level ``oneOf`` of closed objects discriminated by a ``const`` on
   ``type``, with primitive, array, and nested-object properties.

Anything outside that subset raises at load time rather than silently passing,
so the validator can never drift into rubber-stamping the schema.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

SCHEMA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "contract", "grace-events.schema.json",
    )
)

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    # bool is a subclass of int in Python; the contract never means "true" when
    # it says number, so exclude it explicitly.
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}

_SUPPORTED_KEYWORDS = {
    "type", "const", "properties", "required", "additionalProperties",
    "items", "minimum", "description", "$ref",
}


class ContractViolation(ValueError):
    """An event does not match contract/grace-events.schema.json."""


class _Schema:
    """The parsed contract: variant name -> closed object schema."""

    def __init__(self, document: dict):
        self._definitions: dict[str, dict] = document["definitions"]
        self._variants: dict[str, dict] = {}

        for ref in document["oneOf"]:
            name = ref["$ref"].rsplit("/", 1)[-1]
            definition = self._definitions[name]
            const = definition["properties"]["type"]["const"]
            if const != name:
                raise ValueError(
                    f"contract: variant {name!r} discriminates on {const!r}; "
                    "definition key and type const must match"
                )
            _assert_supported(definition, name)
            self._variants[const] = definition

    @property
    def variant_names(self) -> frozenset[str]:
        return frozenset(self._variants)

    def validate(self, event: Any) -> None:
        if not isinstance(event, dict):
            raise ContractViolation(f"event must be an object, got {type(event).__name__}")

        kind = event.get("type")
        if kind is None:
            raise ContractViolation("event has no 'type' discriminator")
        if kind not in self._variants:
            raise ContractViolation(
                f"unknown event type {kind!r}; the contract declares "
                f"{sorted(self._variants)}"
            )
        _check(event, self._variants[kind], path=kind)


def _assert_supported(node: dict, path: str) -> None:
    """Fail loudly on schema features this validator would silently ignore."""
    unsupported = set(node) - _SUPPORTED_KEYWORDS
    if unsupported:
        raise ValueError(
            f"contract at {path}: unsupported schema keyword(s) {sorted(unsupported)}. "
            "Either extend harness/contract.py or avoid the keyword."
        )
    if node.get("type") == "object" and "properties" in node:
        if node.get("additionalProperties") is not False:
            raise ValueError(
                f"contract at {path}: objects with properties must set "
                "additionalProperties:false, or unknown fields pass silently"
            )
        for name, sub in node["properties"].items():
            _assert_supported(sub, f"{path}.{name}")
    if "items" in node:
        _assert_supported(node["items"], f"{path}[]")


def _type_matches(value: Any, expected: Any) -> bool:
    kinds = expected if isinstance(expected, list) else [expected]
    return any(_TYPE_CHECKS[kind](value) for kind in kinds)


def _check(value: Any, node: dict, path: str) -> None:
    if "const" in node:
        if value != node["const"]:
            raise ContractViolation(f"{path}: expected {node['const']!r}, got {value!r}")
        return

    expected = node.get("type")
    if expected is not None and not _type_matches(value, expected):
        raise ContractViolation(
            f"{path}: expected type {expected}, got {type(value).__name__}"
        )

    if "minimum" in node and value < node["minimum"]:
        raise ContractViolation(f"{path}: {value} is below minimum {node['minimum']}")

    if isinstance(value, dict):
        properties: dict = node.get("properties", {})
        for name in node.get("required", []):
            if name not in value:
                raise ContractViolation(f"{path}: missing required field {name!r}")

        extra_schema = node.get("additionalProperties")
        for name, sub_value in value.items():
            if name in properties:
                _check(sub_value, properties[name], f"{path}.{name}")
            elif extra_schema is False:
                raise ContractViolation(
                    f"{path}: unexpected field {name!r}; the contract allows "
                    f"{sorted(properties)}"
                )
            elif isinstance(extra_schema, dict):
                _check(sub_value, extra_schema, f"{path}.{name}")

    elif isinstance(value, list) and "items" in node:
        for index, item in enumerate(value):
            _check(item, node["items"], f"{path}[{index}]")


_schema: Optional[_Schema] = None


def _load() -> _Schema:
    global _schema
    if _schema is None:
        with open(SCHEMA_PATH, encoding="utf-8") as fh:
            _schema = _Schema(json.load(fh))
    return _schema


def validate_event(event: Any) -> None:
    """Raise :class:`ContractViolation` if *event* is not a valid GraceEvent."""
    _load().validate(event)


def contract_variants() -> frozenset[str]:
    """Every event type the contract declares. Used by the conformance tests."""
    return _load().variant_names
