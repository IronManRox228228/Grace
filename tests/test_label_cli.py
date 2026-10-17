"""Tests for snapshot labelling CLI tool."""

import json
from pathlib import Path
import pytest

from bench.dataset_builder import ElementSummary, load_element_selection_dataset
from bench.label_cli import (
    format_element_row,
    label_snapshot_interactive,
    run_labeller,
)


def test_format_element_row():
    el = ElementSummary(id=1, name="Submit", control_type="button", parent_path="Form", window_title="Test")
    row = format_element_row(el, is_target=True)
    assert ">>>" in row
    assert "#1" in row
    assert "[button    ]" in row
    assert '"Submit"' in row


def test_label_snapshot_interactive_input():
    elements = [
        ElementSummary(id=1, name="Cancel", control_type="button", parent_path="", window_title="Test"),
        ElementSummary(id=2, name="OK", control_type="button", parent_path="", window_title="Test"),
    ]

    # Scripted input: user types "2"
    item = label_snapshot_interactive(
        sample_id="test_01",
        window_title="Test Window",
        default_instruction="click OK",
        elements=elements,
        input_fn=lambda prompt: "2",
    )
    assert item is not None
    assert item.correct_id == 2
    assert item.instruction == "click OK"
    # Verify abstain option is present
    assert any(e.id == -1 for e in item.elements)


def test_label_snapshot_interactive_abstain():
    elements = [
        ElementSummary(id=1, name="Cancel", control_type="button", parent_path="", window_title="Test"),
    ]

    # Scripted input: user types "-1" (none of these)
    item = label_snapshot_interactive(
        sample_id="test_02",
        window_title="Test Window",
        default_instruction="find missing",
        elements=elements,
        input_fn=lambda prompt: "-1",
    )
    assert item is not None
    assert item.correct_id == -1


def test_label_snapshot_interactive_skip():
    elements = [
        ElementSummary(id=1, name="Cancel", control_type="button", parent_path="", window_title="Test"),
    ]

    # User types "s" to skip
    item = label_snapshot_interactive(
        sample_id="test_03",
        window_title="Test Window",
        default_instruction="skip this",
        elements=elements,
        input_fn=lambda prompt: "s",
    )
    assert item is None


def test_run_labeller_session(tmp_path):
    snap_file = tmp_path / "snapshots.jsonl"
    out_file = tmp_path / "labelled.jsonl"

    snap = {
        "snapshot": {
            "window": {"title": "App - Calculator"},
            "elements": [
                {"id": 10, "role": "button", "name": "One"},
                {"id": 20, "role": "button", "name": "Two"},
            ],
        }
    }
    snap_file.write_text(json.dumps(snap) + "\n", encoding="utf-8")

    # Scripted input: selects element 10
    added = run_labeller(
        snapshots_path=snap_file,
        out_path=out_file,
        resume=False,
        input_fn=lambda p: "10",
        max_items=1,
    )
    assert added == 1
    assert out_file.is_file()

    items = load_element_selection_dataset(out_file)
    assert len(items) == 1
    assert items[0].correct_id == 10
