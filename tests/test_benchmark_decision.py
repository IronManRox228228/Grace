"""Tests for Benchmark 2 decision modeling evaluators and runners."""

import pytest

from bench.dataset_builder import ElementSelectionItem, ElementSummary, YesNoItem
from bench.run_decision_benchmark import (
    AnyJevLFMModel,
    ContrastiveHeadsModel,
    GLiNERDecideModel,
    HoloReferenceModel,
    Julia1Model,
    LayaModel,
    evaluate_decision_model,
)


def test_julia_shortlist_top_20():
    julia = Julia1Model()
    elements = [
        ElementSummary(id=i, name=f"Button {i}", control_type="button", parent_path="Form", window_title="Win")
        for i in range(30)
    ]
    # Add abstain candidate
    elements.append(ElementSummary(id=-1, name="none of these", control_type="action", parent_path="", window_title="Win"))

    shortlisted = julia._shortlist_top_20("click Button 5", elements)
    assert len(shortlisted) == 20
    # Must retain abstain option #-1
    assert any(e.id == -1 for e in shortlisted)
    # The relevant button should be shortlisted
    assert any(e.name == "Button 5" for e in shortlisted)


def test_calibrated_abstain_threshold():
    model = Julia1Model()
    # Ambiguous/unmatched elements where user instruction is completely unrelated
    elements = [
        ElementSummary(id=1, name="Alpha", control_type="button", parent_path="", window_title="Win"),
        ElementSummary(id=2, name="Beta", control_type="button", parent_path="", window_title="Win"),
        ElementSummary(id=-1, name="none of these", control_type="action", parent_path="", window_title="Win"),
    ]
    ranked_ids, probs = model.predict_element_selection(
        instruction="unrelated zebra goal",
        elements=elements,
        abstain_threshold=0.50,
    )
    # Due to low score / abstain threshold, -1 must be ranked first
    assert ranked_ids[0] == -1


def test_decision_evaluators_scoring():
    elements = [
        ElementSummary(id=1, name="Submit", control_type="button", parent_path="", window_title="Win"),
        ElementSummary(id=2, name="Cancel", control_type="button", parent_path="", window_title="Win"),
        ElementSummary(id=-1, name="none of these", control_type="action", parent_path="", window_title="Win"),
    ]

    models = [
        Julia1Model(),
        GLiNERDecideModel(),
        LayaModel(),
        AnyJevLFMModel(mode="L0"),
        AnyJevLFMModel(mode="L1"),
        ContrastiveHeadsModel(),
        HoloReferenceModel(),
    ]

    for m in models:
        ranked_ids, probs = m.predict_element_selection("click the submit button", elements)
        assert len(ranked_ids) == len(elements)
        assert ranked_ids[0] == 1  # Top-1 is Submit

        label, conf = m.predict_yes_no("yes please")
        assert label == "YES"
        assert conf > 0.5


def test_evaluate_decision_model_metrics():
    model = Julia1Model()
    elements = [
        ElementSummary(id=1, name="Submit", control_type="button", parent_path="", window_title="Win"),
        ElementSummary(id=-1, name="none of these", control_type="action", parent_path="", window_title="Win"),
    ]
    elem_items = [
        ElementSelectionItem(sample_id="s1", instruction="submit form", elements=elements, correct_id=1, app_name="App"),
    ]
    yn_items = [
        YesNoItem(sample_id="y1", utterance="confirm", category="clear_yes", expected_label="YES"),
        YesNoItem(sample_id="y2", utterance="what did you say", category="repeat", expected_label="REPEAT"),
    ]

    metrics = evaluate_decision_model(model, elem_items, yn_items)
    assert metrics["element_selection"]["top_1_accuracy"] == 1.0
    assert metrics["yes_no"]["overall_accuracy"] == 1.0
    assert "latency_p50_ms" in metrics["element_selection"]
