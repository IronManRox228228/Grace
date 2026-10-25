"""Tests for hardware tier detection (PLAN.md §2, Task 08)."""

import pytest
from unittest.mock import patch

from grace.tier import (
    HardwareTier,
    TierDiagnostics,
    detect_hardware_tier,
    get_memory_info,
)

from grace.fleet import get_fleet_config


def test_hardware_tier_enum_values():
    assert HardwareTier.BEST_NPU.value == "npu"
    assert HardwareTier.MIDDLE_IGPU.value == "igpu"
    assert HardwareTier.FLOOR_CPU.value == "cpu"


def test_manual_override_via_config():
    class DummyConfig:
        hardware_tier = "npu"

    diag = detect_hardware_tier(DummyConfig())
    assert diag.tier == HardwareTier.BEST_NPU
    assert "override" in diag.rationale.lower()


def test_manual_override_via_env(monkeypatch):
    monkeypatch.setenv("HARDWARE_TIER", "igpu")
    diag = detect_hardware_tier()
    assert diag.tier == HardwareTier.MIDDLE_IGPU


def test_low_available_ram_forces_cpu_floor(monkeypatch):
    monkeypatch.delenv("HARDWARE_TIER", raising=False)
    # Total 16 GB, but available is only 2.1 GB (< 3.5 GB threshold)
    with patch("grace.tier.get_memory_info", return_value=(16.0, 2.1)):
        with patch("grace.tier.check_npu_available", return_value=True):
            diag = detect_hardware_tier()
            assert diag.tier == HardwareTier.FLOOR_CPU
            assert "low available ram" in diag.rationale.lower()


def test_npu_selected_when_available_and_ram_sufficient(monkeypatch):
    monkeypatch.delenv("HARDWARE_TIER", raising=False)
    with patch("grace.tier.get_memory_info", return_value=(16.0, 8.0)):
        with patch("grace.tier.check_npu_available", return_value=True):
            with patch("grace.tier.check_igpu_available", return_value=True):
                diag = detect_hardware_tier()
                assert diag.tier == HardwareTier.BEST_NPU


def test_igpu_selected_when_no_npu_and_ram_sufficient(monkeypatch):
    monkeypatch.delenv("HARDWARE_TIER", raising=False)
    with patch("grace.tier.get_memory_info", return_value=(16.0, 8.0)):
        with patch("grace.tier.check_npu_available", return_value=False):
            with patch("grace.tier.check_dgpu_available", return_value=False):
                with patch("grace.tier.check_igpu_available", return_value=True):
                    diag = detect_hardware_tier()
                    assert diag.tier == HardwareTier.MIDDLE_IGPU


def test_cpu_floor_fallback_when_no_accelerator(monkeypatch):
    monkeypatch.delenv("HARDWARE_TIER", raising=False)
    with patch("grace.tier.get_memory_info", return_value=(16.0, 8.0)):
        with patch("grace.tier.check_npu_available", return_value=False):
            with patch("grace.tier.check_dgpu_available", return_value=False):
                with patch("grace.tier.check_igpu_available", return_value=False):
                    diag = detect_hardware_tier()
                    assert diag.tier == HardwareTier.FLOOR_CPU


def test_get_memory_info_returns_positive_numbers():
    total, avail = get_memory_info()
    assert total > 0.0
    assert avail > 0.0
    assert avail <= total


def test_dgpu_selected_when_available(monkeypatch):
    monkeypatch.delenv("HARDWARE_TIER", raising=False)
    with patch("grace.tier.get_memory_info", return_value=(16.0, 8.0)):
        with patch("grace.tier.check_npu_available", return_value=False):
            with patch("grace.tier.check_igpu_available", return_value=False):
                with patch("grace.tier.check_dgpu_available", return_value=True):
                    diag = detect_hardware_tier()
                    assert diag.tier == HardwareTier.BEST_DGPU
                    assert diag.has_dgpu is True
                    assert "cuda" in diag.rationale.lower()


def test_dgpu_takes_precedence_over_igpu(monkeypatch):
    monkeypatch.delenv("HARDWARE_TIER", raising=False)
    with patch("grace.tier.get_memory_info", return_value=(16.0, 8.0)):
        with patch("grace.tier.check_npu_available", return_value=False):
            with patch("grace.tier.check_igpu_available", return_value=True):
                with patch("grace.tier.check_dgpu_available", return_value=True):
                    diag = detect_hardware_tier()
                    assert diag.tier == HardwareTier.BEST_DGPU


def test_low_ram_forces_floor_even_with_dgpu(monkeypatch):
    monkeypatch.delenv("HARDWARE_TIER", raising=False)
    with patch("grace.tier.get_memory_info", return_value=(16.0, 2.1)):
        with patch("grace.tier.check_dgpu_available", return_value=True):
            diag = detect_hardware_tier()
            assert diag.tier == HardwareTier.FLOOR_CPU


def test_no_dgpu_flag_when_absent(monkeypatch):
    monkeypatch.delenv("HARDWARE_TIER", raising=False)
    with patch("grace.tier.get_memory_info", return_value=(16.0, 8.0)):
        with patch("grace.tier.check_npu_available", return_value=False):
            with patch("grace.tier.check_igpu_available", return_value=False):
                with patch("grace.tier.check_dgpu_available", return_value=False):
                    diag = detect_hardware_tier()
                    assert diag.tier == HardwareTier.FLOOR_CPU
                    assert diag.has_dgpu is False


def test_dgpu_tier_triggers_gpu_single_fleet(monkeypatch):
    """A discrete NVIDIA GPU runs the single-GPU preset (Holo on CUDA)."""
    diag = TierDiagnostics(
        tier=HardwareTier.BEST_DGPU,
        available_ram_gb=16.0,
        total_ram_gb=16.0,
        has_npu=False,
        has_dgpu=True,
        has_igpu=False,
        override=None,
        rationale="dGPU",
    )
    with patch("grace.fleet.detect_hardware_tier", return_value=diag):
        cfg = get_fleet_config(None, None)
        assert cfg.preset == "gpu-single"
