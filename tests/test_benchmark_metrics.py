"""Tests for benchmark metrics (top-k, abstain precision, ECE, tool calling)."""

import pytest

from bench.metrics import (
    compute_abstain_precision,
    compute_expected_calibration_error,
    compute_latency_stats,
    compute_tool_calling_metrics,
    compute_top_k_accuracy,
)


def test_top_k_accuracy():
    predictions = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ground_truth = [1, 5, 10]

    # top-1: hit on sample 0 only
    assert compute_top_k_accuracy(predictions, ground_truth, k=1) == pytest.approx(1.0 / 3.0)

    # top-2: hit on sample 0 and 1
    assert compute_top_k_accuracy(predictions, ground_truth, k=2) == pytest.approx(2.0 / 3.0)

    # empty inputs
    assert compute_top_k_accuracy([], [], k=1) == 0.0


def test_abstain_precision():
    # ground truth: sample 0 is normal, sample 1 is abstain (-1), sample 2 is normal
    ground_truth = [5, -1, 3]
    # predictions: sample 0 predicted normal, sample 1 predicted abstain, sample 2 predicted abstain (FP)
    predictions = [5, -1, -1]

    stats = compute_abstain_precision(predictions, ground_truth, abstain_id=-1)
    # TP = 1 (sample 1), FP = 1 (sample 2), FN = 0, TN = 1 (sample 0)
    assert stats["tp"] == 1
    assert stats["fp"] == 1
    assert stats["fn"] == 0
    assert stats["tn"] == 1
    assert stats["precision"] == pytest.approx(0.5)
    assert stats["recall"] == pytest.approx(1.0)
    assert stats["f1"] == pytest.approx(2.0 * 0.5 * 1.0 / 1.5)


def test_expected_calibration_error():
    # Perfectly calibrated predictions: confidence 1.0 with 100% accuracy
    confidences = [1.0, 1.0, 1.0]
    accuracies = [True, True, True]
    assert compute_expected_calibration_error(confidences, accuracies, num_bins=10) == pytest.approx(0.0)

    # Severely overconfident: confidence 0.9 with 0% accuracy
    confidences = [0.9, 0.9]
    accuracies = [False, False]
    ece = compute_expected_calibration_error(confidences, accuracies, num_bins=10)
    assert ece == pytest.approx(0.9)


def test_tool_calling_metrics():
    predictions = [
        {"tool": "open_app", "args": {"app_name": "chrome"}, "valid_json": True},
        {"tool": "close_app", "args": {"app_name": "notepad"}, "valid_json": True},
        {"tool": "open_app", "args": {"app_name": "spotify"}, "valid_json": False},
    ]
    references = [
        {"tool": "open_app", "args": {"app_name": "chrome"}},
        {"tool": "close_app", "args": {"app_name": "spotify"}},  # tool matches, args mismatch
        {"tool": "open_app", "args": {"app_name": "spotify"}},
    ]

    metrics = compute_tool_calling_metrics(predictions, references)
    assert metrics["exact_tool_accuracy"] == pytest.approx(1.0)  # all 3 tool names match
    assert metrics["args_match_accuracy"] == pytest.approx(2.0 / 3.0)  # samples 0 and 2 match
    assert metrics["json_validity_rate"] == pytest.approx(2.0 / 3.0)  # samples 0 and 1 are valid JSON
