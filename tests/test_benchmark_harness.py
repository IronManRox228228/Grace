"""Unit tests for generic benchmark harness, metrics, normalizer, and load generator (Task 09)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time

import pytest

from bench.harness import (
    append_result,
    compute_file_sha256,
    create_benchmark_record,
    get_git_commit,
    get_machine_profile,
    get_runtime_info,
    get_thread_settings,
    load_results,
)
from bench.load_generator import BackgroundLoadSimulator
from bench.metrics import compute_latency_stats, real_time_factor, word_error_rate
from bench.normalizer import normalize_text
from bench.report import generate_markdown_report


# ============================================================================
# Metrics Tests
# ============================================================================

def test_word_error_rate_identical():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_word_error_rate_empty():
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "some hypothesis") == 1.0
    assert word_error_rate("reference text", "") == 1.0


def test_word_error_rate_substitutions_insertions_deletions():
    # 1 substitution in 4 words = 0.25
    assert word_error_rate("the quick brown fox", "the fast brown fox") == 0.25
    # 1 insertion: "the quick brown red fox" = 1/4 = 0.25
    assert word_error_rate("the quick brown fox", "the quick brown red fox") == 0.25
    # 1 deletion: "the brown fox" = 1/4 = 0.25
    assert word_error_rate("the quick brown fox", "the brown fox") == 0.25


def test_real_time_factor():
    assert real_time_factor(10.0, 2.0) == 0.2
    assert real_time_factor(0.0, 1.0) == 0.0
    assert real_time_factor(5.0, 5.0) == 1.0


def test_compute_latency_stats():
    empty_stats = compute_latency_stats([])
    assert empty_stats["count"] == 0
    assert empty_stats["p50"] == 0.0

    single_stats = compute_latency_stats([100.0])
    assert single_stats["count"] == 1
    assert single_stats["mean"] == 100.0
    assert single_stats["p50"] == 100.0
    assert single_stats["p95"] == 100.0

    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_latency_stats(latencies)
    assert stats["count"] == 5
    assert stats["mean"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["p50"] == 30.0
    assert stats["p95"] == pytest.approx(48.0, 0.1)


# ============================================================================
# Text Normalizer Tests
# ============================================================================

def test_normalize_text_basic():
    raw = "Hello, World!"
    assert normalize_text(raw) == "hello world"


def test_normalize_text_bracketed_tokens():
    raw = "[cough] open <laughter> Chrome (sigh) browser"
    assert normalize_text(raw) == "open chrome browser"


def test_normalize_text_number_expansion():
    raw = "I have 2 cats and 10 dogs on street 42"
    # 2 -> two, 10 -> ten, numbers > 20 remain as digits
    assert normalize_text(raw) == "i have two cats and ten dogs on street 42"


def test_normalize_text_quotes_and_hyphens():
    raw = "“Don’t” touch twenty-one buttons!"
    assert normalize_text(raw) == "dont touch twenty one buttons"


# ============================================================================
# Background Load Simulator Tests
# ============================================================================

def test_background_load_simulator():
    # Small test simulation: 2 tabs, 5MB each, 1% CPU
    sim = BackgroundLoadSimulator(num_tabs=2, mb_per_tab=5.0, cpu_load_pct=1.0)
    assert not sim.is_running

    with sim:
        assert sim.is_running
        assert sim.allocated_mb == 10.0
        time.sleep(0.05)

    assert not sim.is_running
    assert sim.allocated_mb == 0.0


# ============================================================================
# Generic Harness & Reporting Tests
# ============================================================================

def test_harness_environment_detection():
    profile = get_machine_profile(background_load="test_load")
    assert "cpu" in profile
    assert profile["ram_total_gb"] > 0
    assert profile["tier_chosen"] in ["cpu", "igpu", "npu", "dgpu"]
    assert profile["background_load"] == "test_load"

    runtimes = get_runtime_info()
    assert "python" in runtimes

    threads = get_thread_settings(profile["tier_chosen"])
    assert threads["active_worker_threads"] > 0


def test_harness_file_sha256(tmp_path: Path):
    dummy_file = tmp_path / "model.bin"
    dummy_file.write_bytes(b"grace-benchmark-dummy-model-bytes")

    sha = compute_file_sha256(dummy_file)
    assert sha is not None
    assert len(sha) == 64

    # Non-existent file
    assert compute_file_sha256(tmp_path / "nonexistent.bin") is None


def test_harness_record_and_append(tmp_path: Path):
    record = create_benchmark_record(
        benchmark_name="test_bench",
        model_id="test-whisper",
        metrics={
            "overall_wer": 0.12,
            "mean_rtf": 0.25,
            "latency_p50_ms": 150.0,
            "latency_p95_ms": 300.0,
        },
        background_load="idle",
    )

    out_file = append_result("test_bench", record, results_dir=tmp_path)
    assert out_file.is_file()

    loaded = load_results("test_bench", results_dir=tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["model_id"] == "test-whisper"
    assert loaded[0]["metrics"]["overall_wer"] == 0.12


def test_report_generation():
    records = [
        {
            "benchmark_name": "asr_torgo",
            "model_id": "test-model-1",
            "timestamp": "2026-10-17T15:00:00Z",
            "git_commit": "abc1234",
            "machine_profile": {
                "os": "Windows 11",
                "cpu": "Intel Test CPU",
                "cpu_count_physical": 8,
                "cpu_count_logical": 16,
                "ram_total_gb": 16.0,
                "ram_available_gb": 8.0,
                "tier_chosen": "cpu",
                "tier_rationale": "Test CPU floor",
                "background_load": "idle",
            },
            "thread_settings": {
                "model_threads": 4,
                "rationale": "Test threads",
            },
            "runtime": {"python": "3.10.11"},
            "metrics": {
                "overall_wer": 0.15,
                "dysarthric_wer": 0.22,
                "control_wer": 0.05,
                "mean_rtf": 0.18,
                "latency_p50_ms": 120.0,
                "latency_p95_ms": 250.0,
                "wer_by_severity": {"Control": 0.05, "Mild": 0.10, "Moderate": 0.20, "Severe": 0.35},
                "wer_by_speaker": {
                    "F01": {"wer": 0.35, "is_dysarthric": True, "severity": "Severe", "utterance_count": 10},
                    "FC01": {"wer": 0.05, "is_dysarthric": False, "severity": "Control", "utterance_count": 10},
                },
            },
        }
    ]

    report = generate_markdown_report(records)
    assert "# Benchmark Report: ASR on TORGO" in report
    assert "test-model-1" in report
    assert "15.00%" in report  # Overall WER formatted
    assert "F01" in report
    assert "FC01" in report
