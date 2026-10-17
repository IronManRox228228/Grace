"""Tests for dataset builder and extraction logic."""

import json
from pathlib import Path
import pytest

from bench.dataset_builder import (
    ElementSelectionItem,
    ElementSummary,
    YesNoItem,
    build_element_summary,
    extract_app_name,
    extract_element_selection_dataset,
    extract_tool_calling_dataset,
    extract_yes_no_dataset,
    load_element_selection_dataset,
    load_tool_calling_dataset,
    load_yes_no_dataset,
    save_dataset_jsonl,
)


def test_element_summary_formatting():
    el = {
        "id": 2,
        "role": "edit",
        "name": "Search",
        "container": "Main",
    }
    all_els = [
        {"id": 1, "name": "Back"},
        el,
        {"id": 3, "name": "Submit"},
    ]
    summary = build_element_summary(el, all_els, "Edge Window")

    assert summary.id == 2
    assert summary.name == "Search"
    assert summary.control_type == "edit"
    assert summary.parent_path == "Main"
    assert summary.window_title == "Edge Window"
    assert "Submit" in summary.neighbouring_labels or "Back" in summary.neighbouring_labels

    text = summary.to_text()
    assert "[#2] Search (edit)" in text
    assert "Edge Window" in text


def test_extract_app_name():
    assert extract_app_name("Grace test page - Edge") == "Edge"
    assert extract_app_name("Untitled - Notepad") == "Notepad"
    assert extract_app_name("Calculator") == "Calculator"


def test_extract_element_selection_from_fake_tape(tmp_path):
    tape_dir = tmp_path / "tape_01"
    tape_dir.mkdir()

    # Create fake snapshots.jsonl
    snap = {
        "snapshot": {
            "window": {"title": "Test - Settings"},
            "elements": [
                {"id": 1, "role": "button", "name": "System"},
                {"id": 2, "role": "button", "name": "Bluetooth"},
            ],
        }
    }
    (tape_dir / "snapshots.jsonl").write_text(json.dumps(snap) + "\n", encoding="utf-8")
    (tape_dir / "dispatch.jsonl").write_text(
        json.dumps({"tool": "cua_click", "params": {"element_id": 2}}) + "\n", encoding="utf-8"
    )
    (tape_dir / "stt.jsonl").write_text(
        json.dumps({"text": "click bluetooth"}) + "\n", encoding="utf-8"
    )

    items = extract_element_selection_dataset(tmp_path, held_out_apps=["Settings"])
    assert len(items) >= 1

    pos_item = next(it for it in items if it.correct_id == 2)
    assert pos_item.correct_id == 2
    assert pos_item.is_held_out is True
    assert pos_item.app_name == "Settings"
    assert any(e.id == -1 for e in pos_item.elements)  # abstain candidate included


def test_extract_yes_no_dataset_covers_categories():
    items = extract_yes_no_dataset()
    categories = {it.category for it in items}
    expected = {"clear_yes", "clear_no", "hedged", "greeting_yes", "greeting_no", "repeat", "stop", "abstain"}
    assert expected.issubset(categories)


def test_extract_tool_calling_dataset():
    items = extract_tool_calling_dataset()
    assert len(items) >= 20
    tools = {it.expected_tool for it in items}
    assert "open_app" in tools
    assert "cua_click" in tools
    assert "cua_type_text" in tools


def test_dataset_jsonl_serialization(tmp_path):
    target = tmp_path / "test_yn.jsonl"
    items = [
        YesNoItem(sample_id="s1", utterance="yes", category="clear_yes", expected_label="YES"),
        YesNoItem(sample_id="s2", utterance="no", category="clear_no", expected_label="NO"),
    ]
    save_dataset_jsonl(items, target)

    loaded = load_yes_no_dataset(target)
    assert len(loaded) == 2
    assert loaded[0].sample_id == "s1"
    assert loaded[0].expected_label == "YES"
