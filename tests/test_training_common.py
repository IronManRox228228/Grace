"""Unit tests for training common governance, checkpointing, and replacement gate."""

from pathlib import Path
import pytest

from training.common import (
    CheckpointManager,
    ModelReplacementGate,
    ResourceGovernor,
    TrainingConfig,
)


def test_resource_governor_user_active_hook():
    gov = ResourceGovernor(
        allow_on_battery=True,
        max_cpu_percent=100.0,
        power_provider=lambda: (True, 80),
        thermal_provider=lambda: (45.0, False),
        cpu_percent_provider=lambda: 10.0,
        turn_active_provider=lambda: False,
    )
    should_pause, reason = gov.should_pause()
    assert should_pause is False

    # Simulate user activity hook
    gov.register_user_active_hook(lambda: True)
    should_pause, reason = gov.should_pause()
    assert should_pause is True
    assert "User activity" in reason


def test_checkpoint_manager(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(checkpoint_dir=ckpt_dir, interval_seconds=1)

    # Save checkpoint
    saved_path = mgr.save_checkpoint(step=10, epoch=1, state_dict={"w": "tensor"}, metrics={"loss": 0.42})
    assert saved_path.is_dir()
    assert (saved_path / "meta.json").is_file()

    # Find latest checkpoint
    latest = mgr.find_latest_checkpoint()
    assert latest == saved_path

    meta = mgr.load_latest_checkpoint_meta()
    assert meta["step"] == 10
    assert meta["metrics"]["loss"] == 0.42


def test_model_replacement_gate():
    gate = ModelReplacementGate(slot_name="element_selection", baseline_metric_key="top1")

    # Better score: approve
    approved, msg = gate.should_replace(new_metric=0.92, current_baseline_metric=0.88)
    assert approved is True
    assert "APPROVED" in msg

    # Worse score: reject
    rejected, msg = gate.should_replace(new_metric=0.85, current_baseline_metric=0.88)
    assert rejected is False
    assert "REJECTED" in msg
