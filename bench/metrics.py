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

