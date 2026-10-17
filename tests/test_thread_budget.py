"""Tests for thread budgeting and CPU allocation (PLAN.md §4, Task 08)."""

import pytest
from unittest.mock import patch, MagicMock

from grace.tier import HardwareTier
from grace.thread_budget import (
    calculate_thread_budget,
    get_llama_thread_args,
    configure_ort_session_options,
)


def test_cpu_floor_disables_tts_lookahead():
    with patch("os.cpu_count", return_value=8):
        budget = calculate_thread_budget(HardwareTier.FLOOR_CPU)
        assert budget.allow_tts_lookahead is False
        assert budget.active_worker_threads == 6  # 8 - 2 reserved
        assert budget.llama_threads == 4  # capped at 4 on floor
        assert budget.ort_intra_op_threads == 4


def test_accelerated_tiers_enable_tts_lookahead():
    with patch("os.cpu_count", return_value=8):
        budget_npu = calculate_thread_budget(HardwareTier.BEST_NPU)
        assert budget_npu.allow_tts_lookahead is True
        assert budget_npu.llama_threads == 6

        budget_igpu = calculate_thread_budget(HardwareTier.MIDDLE_IGPU)
        assert budget_igpu.allow_tts_lookahead is True


def test_thread_budget_override_via_config():
    class DummyConfig:
        thread_budget = 3

    with patch("os.cpu_count", return_value=16):
        budget = calculate_thread_budget(HardwareTier.FLOOR_CPU, DummyConfig())
        assert budget.active_worker_threads == 3
        assert budget.llama_threads == 3
        assert "override" in budget.rationale.lower()


def test_thread_budget_override_via_env(monkeypatch):
    monkeypatch.setenv("THREAD_BUDGET", "2")
    with patch("os.cpu_count", return_value=12):
        budget = calculate_thread_budget(HardwareTier.MIDDLE_IGPU)
        assert budget.active_worker_threads == 2


def test_get_llama_thread_args():
    with patch("os.cpu_count", return_value=8):
        budget = calculate_thread_budget(HardwareTier.FLOOR_CPU)
        args = get_llama_thread_args(budget)
        assert args == ["-t", str(budget.llama_threads)]


def test_configure_ort_session_options():
    mock_opts = MagicMock()
    with patch("os.cpu_count", return_value=8):
        budget = calculate_thread_budget(HardwareTier.FLOOR_CPU)
        configure_ort_session_options(mock_opts, budget)
        assert mock_opts.intra_op_num_threads == budget.ort_intra_op_threads
        assert mock_opts.inter_op_num_threads == budget.ort_inter_op_threads
