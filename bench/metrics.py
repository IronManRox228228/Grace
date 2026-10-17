"""Benchmark evaluation metrics: WER and RTF (PLAN.md §8, Task 09)."""

from __future__ import annotations

from typing import List, Tuple


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) using Levenshtein distance on word tokens.

    WER = (Substitutions + Deletions + Insertions) / Number of Reference Words.
    Returns 0.0 if both are empty. If reference is empty and hypothesis is not, returns 1.0.
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    r_len = len(ref_words)
    h_len = len(hyp_words)

    # Dynamic programming distance matrix
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1],  # Substitution
                )

    return dp[r_len][h_len] / float(r_len)


def real_time_factor(audio_duration_seconds: float, processing_time_seconds: float) -> float:
    """Calculate Real-Time Factor (RTF = processing_time / audio_duration).

    An RTF < 1.0 means processing is faster than real-time (e.g. 0.2 means 5x faster).
    """
    if audio_duration_seconds <= 0:
        return 0.0
    return processing_time_seconds / audio_duration_seconds


def compute_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Compute summary statistics (p50, p90, p95, p99, mean, min, max) for latencies in ms."""
    if not latencies_ms:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    sorted_vals = sorted(latencies_ms)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        if n == 1:
            return sorted_vals[0]
        rank = (p / 100.0) * (n - 1)
        k = int(rank)
        d = rank - k
        if k + 1 < n:
            return sorted_vals[k] + d * (sorted_vals[k + 1] - sorted_vals[k])
        return sorted_vals[-1]

    return {
        "count": n,
        "mean": sum(sorted_vals) / float(n),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "p50": percentile(50.0),
        "p90": percentile(90.0),
        "p95": percentile(95.0),
        "p99": percentile(99.0),
    }


def compute_top_k_accuracy(predictions: list[list[int]], ground_truth: list[int], k: int = 1) -> float:
    """Compute top-k accuracy for candidate rankings or predictions.

    Args:
        predictions: List of ranked candidate IDs per sample (highest confidence first).
        ground_truth: List of true target IDs per sample.
        k: The cutoff rank (1 for top-1, 3 for top-3, etc.).

    Returns:
        Fraction of samples where the ground truth ID is within the top-k predictions.
    """
    if not ground_truth or not predictions or len(predictions) != len(ground_truth):
        return 0.0

    hits = 0
    for preds, true_id in zip(predictions, ground_truth):
        top_k = preds[:k]
        if true_id in top_k:
            hits += 1

    return hits / float(len(ground_truth))


def compute_abstain_precision(predictions: list[int], ground_truth: list[int], abstain_id: int = -1) -> dict[str, float]:
    """Compute precision, recall, and F1 for abstain / 'none of these' decisions.

    Args:
        predictions: Top-1 predicted element ID per sample.
        ground_truth: Ground truth element ID per sample.
        abstain_id: Sentinel ID indicating abstain / 'none of these' (default -1).

    Returns:
        Dict with precision, recall, f1, and count tallies.
    """
    if not ground_truth or not predictions or len(predictions) != len(ground_truth):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for pred, truth in zip(predictions, ground_truth):
        pred_abstain = (pred == abstain_id)
        true_abstain = (truth == abstain_id)

        if pred_abstain and true_abstain:
            tp += 1
        elif pred_abstain and not true_abstain:
            fp += 1
        elif not pred_abstain and true_abstain:
            fn += 1
        else:
            tn += 1

    precision = (tp / float(tp + fp)) if (tp + fp) > 0 else (1.0 if tp == 0 and fp == 0 and fn == 0 else 0.0)
    recall = (tp / float(tp + fn)) if (tp + fn) > 0 else (1.0 if tp == 0 and fn == 0 and fp == 0 else 0.0)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_expected_calibration_error(
    confidences: list[float],
    accuracies: list[bool],
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions into `num_bins` uniform intervals over [0, 1].
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Args:
        confidences: Output probabilities / confidence scores in [0, 1].
        accuracies: Whether each sample's top-1 prediction was correct.
        num_bins: Number of confidence bins (default 10).

    Returns:
        ECE value in [0, 1].
    """
    if not confidences or not accuracies or len(confidences) != len(accuracies):
        return 0.0

    n = len(confidences)
    bin_size = 1.0 / num_bins
    ece = 0.0

    for b in range(num_bins):
        bin_lower = b * bin_size
        bin_upper = (b + 1) * bin_size

        # Include 1.0 in the last bin
        bin_indices = [
            i for i, c in enumerate(confidences)
            if (bin_lower <= c < bin_upper) or (b == num_bins - 1 and c == 1.0)
        ]

        if not bin_indices:
            continue

        bin_conf = sum(confidences[i] for i in bin_indices) / float(len(bin_indices))
        bin_acc = sum(1.0 for i in bin_indices if accuracies[i]) / float(len(bin_indices))
        bin_weight = len(bin_indices) / float(n)

        ece += bin_weight * abs(bin_acc - bin_conf)

    return ece


def compute_tool_calling_metrics(
    predictions: list[dict],
    references: list[dict],
) -> dict[str, float]:
    """Compute exact tool accuracy, argument match accuracy, and JSON validity.

    Args:
        predictions: List of dicts with {"tool": str, "args": dict, "valid_json": bool}
        references: List of dicts with {"tool": str, "args": dict}

    Returns:
        Dict with exact_tool_accuracy, args_match_accuracy, and json_validity_rate.
    """
    if not references or not predictions or len(predictions) != len(references):
        return {
            "exact_tool_accuracy": 0.0,
            "args_match_accuracy": 0.0,
            "json_validity_rate": 0.0,
            "total_samples": 0,
        }

    n = len(references)
    valid_json_count = 0
    exact_tool_count = 0
    exact_args_count = 0

    for pred, ref in zip(predictions, references):
        is_valid_json = pred.get("valid_json", True)
        if is_valid_json:
            valid_json_count += 1

        pred_tool = (pred.get("tool") or "").strip().lower()
        ref_tool = (ref.get("tool") or "").strip().lower()

        if pred_tool == ref_tool and pred_tool != "":
            exact_tool_count += 1

            pred_args = pred.get("args") or {}
            ref_args = ref.get("args") or {}
            if _normalize_args(pred_args) == _normalize_args(ref_args):
                exact_args_count += 1

    return {
        "exact_tool_accuracy": exact_tool_count / float(n),
        "args_match_accuracy": exact_args_count / float(n),
        "json_validity_rate": valid_json_count / float(n),
        "total_samples": n,
    }


def _normalize_args(args: dict) -> dict:
    """Normalize argument keys and string values for fair dictionary comparison."""
    norm = {}
    for k, v in args.items():
        clean_k = str(k).strip().lower()
        if isinstance(v, str):
            clean_v = v.strip().lower()
        elif isinstance(v, (int, float, bool)):
            clean_v = v
        elif isinstance(v, dict):
            clean_v = _normalize_args(v)
        elif isinstance(v, list):
            clean_v = [x.strip().lower() if isinstance(x, str) else x for x in v]
        else:
            clean_v = str(v)
        norm[clean_k] = clean_v
    return norm


