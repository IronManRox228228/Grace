"""An unset GRACE_PRESET means "pick by hardware tier"; an explicit one wins.

Run 7 evidence: Config defaulted `preset` to "cpu-fleet", so on an RTX 4060
the tier auto-selection in get_fleet_config was never reached and Holo ran
blind (hide_tree) in the fleet preset.
"""

from unittest.mock import patch

import pytest

from grace.config import Config
from grace.fleet import get_fleet_config
from grace.tier import HardwareTier, TierDiagnostics


def _diag(tier):
    return TierDiagnostics(tier=tier, available_ram_gb=16.0, total_ram_gb=16.0, has_npu=False,
                           has_dgpu=tier == HardwareTier.BEST_DGPU, has_igpu=False,
                           override=None, rationale="t")


@pytest.mark.parametrize("tier,expected", [
    (HardwareTier.BEST_DGPU, "gpu-single"),
    (HardwareTier.BEST_NPU, "gpu-single"),
    (HardwareTier.MIDDLE_IGPU, "cpu-fleet"),
    (HardwareTier.FLOOR_CPU, "cpu-fleet"),
])
def test_unset_preset_follows_tier(monkeypatch, tier, expected):
    monkeypatch.delenv("GRACE_PRESET", raising=False)
    cfg = Config()
    assert not cfg.preset
    with patch("grace.fleet.detect_hardware_tier", return_value=_diag(tier)):
        assert get_fleet_config(cfg.preset, cfg).preset == expected


def test_explicit_env_preset_wins_over_tier(monkeypatch):
    monkeypatch.setenv("GRACE_PRESET", "cpu-fleet")
    cfg = Config()
    assert cfg.preset == "cpu-fleet"
    with patch("grace.fleet.detect_hardware_tier", return_value=_diag(HardwareTier.BEST_DGPU)):
        assert get_fleet_config(cfg.preset, cfg).preset == "cpu-fleet"
