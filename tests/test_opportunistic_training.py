"""Unit tests for opportunistic on-device training scheduler (PLAN §6, Task 15).

Tests:
- Scheduler state machine with fake clocks, fake power, fake idle, and fake thermal sources.
- Never triggers on a timer; triggers only on accuracy drop or enough new samples.
- Preconditions: runs only when plugged in (AC) and idle (no user input, no active turn, thermals normal).
- Immediate pause on battery, user activity, or thermal throttling.
- Checkpointing and crash/power-cut resumption on a toy model.
- Promotion gate: replaces old adapter only if candidate strictly beats baseline on held-out recordings.
- One-step rollback ("Grace, go back to the old voice model").
- Voice status query ("How's my voice training going?") and dispatcher integration.
- Strictly no real training or real mic/speaker input in tests.
"""

from pathlib import Path
import shutil
import pytest

from grace.agent.decision_ladder import DecisionLadder, DecisionRung
from grace.enrollment.ongoing import OngoingTracker
from grace.enrollment.storage import EnrollmentStorage
from grace.intent.parser import Intent
from grace.tools.dispatcher import Dispatcher
from grace.training.scheduler import (
    AcousticAdapterTrainer,
    ToyAdapterTrainer,
    TrainingScheduler,
    TrainingSchedulerConfig,
    TrainingSchedulerState,
)
from training.common import (
    CheckpointManager,
    ModelReplacementGate,
    ResourceGovernor,
)


class FakeClock:
    def __init__(self, start_time: float = 1000.0) -> None:
        self.current_time = start_time

    def now(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


class MockPowerSource:
    def __init__(self, plugged: bool = True, percent: int = 100) -> None:
        self.plugged = plugged
        self.percent = percent

    def __call__(self):
        return self.plugged, self.percent


class MockIdleSource:
    def __init__(self, idle_seconds: float = 600.0) -> None:
        self.idle_seconds = idle_seconds

    def __call__(self) -> float:
        return self.idle_seconds


class MockThermalSource:
    def __init__(self, temp: float = 45.0, throttled: bool = False) -> None:
        self.temp = temp
        self.throttled = throttled

    def __call__(self):
        return self.temp, self.throttled


@pytest.fixture
def test_env(tmp_path):
    clock = FakeClock()
    power = MockPowerSource(plugged=True, percent=100)
    idle = MockIdleSource(idle_seconds=600.0)  # 10 min idle
    thermal = MockThermalSource(temp=45.0, throttled=False)
    turn_active = lambda: False

    base_dir = tmp_path / "training"
    storage_dir = tmp_path / "enrollment"

    storage = EnrollmentStorage(base_dir=storage_dir, use_dpapi=False)
    tracker = OngoingTracker(storage=storage, state_file=storage_dir / "tracker.json")

    governor = ResourceGovernor(
        allow_on_battery=False,
        idle_threshold_seconds=300.0,
        max_temp_celsius=80.0,
        power_provider=power,
        idle_provider=idle,
        thermal_provider=thermal,
        turn_active_provider=turn_active,
        clock_provider=clock.now,
        cpu_percent_provider=lambda: 15.0,
    )

    config = TrainingSchedulerConfig(
        min_new_samples=5,
        accuracy_drop_threshold=0.08,
        chunk_duration_seconds=10.0,
        idle_threshold_seconds=300.0,
        total_training_steps=20,
        steps_per_chunk=5,
        base_dir=base_dir,
    )

    toy_trainer = ToyAdapterTrainer(held_out_accuracy=0.97, target_steps=20)

    scheduler = TrainingScheduler(
        config=config,
        ongoing_tracker=tracker,
        enrollment_storage=storage,
        resource_governor=governor,
        trainer=toy_trainer,
        clock_provider=clock.now,
    )

    return {
        "clock": clock,
        "power": power,
        "idle": idle,
        "thermal": thermal,
        "turn_active": turn_active,
        "storage": storage,
        "tracker": tracker,
        "governor": governor,
        "config": config,
        "trainer": toy_trainer,
        "scheduler": scheduler,
        "base_dir": base_dir,
    }


def test_scheduler_never_triggers_on_timer(test_env):
    """Hard Rule: Training must NEVER be triggered by a timer or clock advance alone."""
    scheduler = test_env["scheduler"]
    clock = test_env["clock"]

    # Initially in IDLE
    assert scheduler.state == TrainingSchedulerState.IDLE

    # Advance clock by hours and days
    clock.advance(3600.0 * 24.0 * 7.0)
    triggered, reason = scheduler.check_trigger()
    assert triggered is False
    assert "timer" in reason.lower()
    assert scheduler.state == TrainingSchedulerState.IDLE

    # Run chunk while not triggered
    result = scheduler.run_chunk()
    assert result["status"] == "idle"
    assert scheduler.current_step == 0


def test_scheduler_triggers_on_accuracy_drop(test_env):
    """Trigger 1: Accuracy drop from Task 14 OngoingTracker triggers ELIGIBLE state."""
    scheduler = test_env["scheduler"]
    tracker = test_env["tracker"]

    # Simulate 20 turns with low accuracy (drop > 8%)
    for i in range(20):
        # 12 correct out of 20 = 60% accuracy (well below 95% baseline)
        tracker.record_turn(verified_correct=(i < 12))

    triggered, reason = scheduler.check_trigger()
    assert triggered is True
    assert "accuracy drop detected" in reason.lower()
    assert scheduler.state == TrainingSchedulerState.ELIGIBLE


def test_scheduler_triggers_on_sample_count(test_env):
    """Trigger 2: Accumulating enough new confirmed samples triggers ELIGIBLE state."""
    scheduler = test_env["scheduler"]
    storage = test_env["storage"]

    # Add 5 samples (min_new_samples is 5 in fixture)
    for i in range(5):
        storage.save_utterance(
            session_id="test_session",
            utterance_id=f"utt_{i}",
            phrase_id=f"phrase_{i}",
            phrase_text=f"Sample text {i}",
            category="commands",
            pcm_bytes=b"\x00\x01" * 1600,
        )

    triggered, reason = scheduler.check_trigger()
    assert triggered is True
    assert "collected 5 new confirmed samples" in reason.lower()
    assert scheduler.state == TrainingSchedulerState.ELIGIBLE


def test_scheduler_environment_safeguards(test_env):
    """Scheduler enforces AC power, user idle, and thermal safety before and during training."""
    scheduler = test_env["scheduler"]
    power = test_env["power"]
    idle = test_env["idle"]
    thermal = test_env["thermal"]

    # Set eligible
    scheduler.state = TrainingSchedulerState.ELIGIBLE

    # 1. Battery switch prevents starting
    power.plugged = False
    power.percent = 65
    res = scheduler.run_chunk()
    assert res["status"] == "paused"
    assert "battery" in res["reason"].lower()
    assert scheduler.state == TrainingSchedulerState.PAUSED
    power.plugged = True

    # 2. User active prevents starting (e.g. active 10s ago < 300s threshold)
    idle.idle_seconds = 10.0
    res = scheduler.run_chunk()
    assert res["status"] == "paused"
    assert "user active recently" in res["reason"].lower()
    assert scheduler.state == TrainingSchedulerState.PAUSED
    idle.idle_seconds = 600.0

    # 3. High temperature prevents starting
    thermal.temp = 88.0
    res = scheduler.run_chunk()
    assert res["status"] == "paused"
    assert "temperature high" in res["reason"].lower()
    assert scheduler.state == TrainingSchedulerState.PAUSED
    thermal.temp = 45.0

    # 4. Throttling prevents starting
    thermal.throttled = True
    res = scheduler.run_chunk()
    assert res["status"] == "paused"
    assert "throttling" in res["reason"].lower()
    assert scheduler.state == TrainingSchedulerState.PAUSED
    thermal.throttled = False


def test_mid_chunk_immediate_pause(test_env):
    """When running a chunk, interruption immediately pauses and writes checkpoint."""
    scheduler = test_env["scheduler"]
    power = test_env["power"]

    # Eligible to run
    scheduler.state = TrainingSchedulerState.ELIGIBLE

    # Run first chunk normally (5 steps)
    res = scheduler.run_chunk()
    assert res["status"] == "chunk_complete"
    assert res["step"] == 5
    assert res["chunks_completed"] == 1

    # Simulate battery unplug during step 7
    call_count = 0
    original_should_pause = scheduler.resource_governor.should_pause

    def should_pause_mid_chunk():
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            return True, "Machine running on battery (50% remaining)"
        return original_should_pause()

    scheduler.resource_governor.should_pause = should_pause_mid_chunk

    res2 = scheduler.run_chunk()
    assert res2["status"] == "paused"
    assert "battery" in res2["reason"].lower()
    assert scheduler.state == TrainingSchedulerState.PAUSED
    assert scheduler.current_step == 6  # Executed 1 step before pausing on 2nd check

    # Checkpoint was saved on pause
    latest_ckpt = scheduler.checkpoint_manager.find_latest_checkpoint()
    assert latest_ckpt is not None
    meta = scheduler.checkpoint_manager.load_latest_checkpoint_meta()
    assert meta["step"] == 6


def test_checkpoint_atomic_resumption_after_power_cut(test_env, tmp_path):
    """Simulates sudden power cut/crash; a new scheduler instance resumes cleanly from checkpoint."""
    scheduler = test_env["scheduler"]
    config = test_env["config"]
    storage = test_env["storage"]
    tracker = test_env["tracker"]
    governor = test_env["governor"]
    clock = test_env["clock"]

    scheduler.state = TrainingSchedulerState.ELIGIBLE

    # Run 2 chunks (10 steps)
    scheduler.run_chunk(max_steps=5)
    scheduler.run_chunk(max_steps=5)
    assert scheduler.current_step == 10
    assert scheduler.chunks_completed == 2

    # Simulate abrupt process crash / power cut: instantiate new scheduler pointing to same directory
    new_trainer = ToyAdapterTrainer(held_out_accuracy=0.97, target_steps=20)
    resumed_scheduler = TrainingScheduler(
        config=config,
        ongoing_tracker=tracker,
        enrollment_storage=storage,
        resource_governor=governor,
        trainer=new_trainer,
        clock_provider=clock.now,
    )

    # Verifies step 10 was seamlessly restored from disk
    assert resumed_scheduler.current_step == 10
    assert new_trainer.step == 10
    assert resumed_scheduler.chunks_completed == 2

    # Resume and finish remaining steps
    res = resumed_scheduler.run_chunk(max_steps=10)
    assert res["status"] == "completed"
    assert resumed_scheduler.current_step == 20
    assert resumed_scheduler.state == TrainingSchedulerState.COMPLETED


def test_promotion_gate_and_rejection(test_env):
    """A new adapter replaces active model ONLY if held-out accuracy improves; otherwise discarded."""
    scheduler = test_env["scheduler"]
    trainer = test_env["trainer"]

    scheduler.state = TrainingSchedulerState.ELIGIBLE
    scheduler.current_accuracy = 0.950  # Baseline active accuracy

    # 1. Candidate with BETTER score (0.975 > 0.950) -> PROMOTED
    trainer.held_out_accuracy = 0.975
    scheduler.run_chunk(max_steps=20)

    assert scheduler.state == TrainingSchedulerState.COMPLETED
    assert scheduler.last_promotion_status == "promoted"
    assert scheduler.current_accuracy == 0.975
    assert (scheduler.active_adapter_dir / "adapter_model.json").is_file()

    # 2. Candidate with WORSE score (0.940 <= 0.975) -> DISCARDED
    trainer.held_out_accuracy = 0.940
    trainer.step = 0
    scheduler.current_step = 0
    scheduler.state = TrainingSchedulerState.ELIGIBLE

    scheduler.run_chunk(max_steps=20)
    assert scheduler.state == TrainingSchedulerState.COMPLETED
    assert scheduler.last_promotion_status == "discarded"
    # Active accuracy remains 0.975! Not degraded.
    assert scheduler.current_accuracy == 0.975


def test_one_step_rollback(test_env):
    """Users can roll back to the previous voice model if desired."""
    scheduler = test_env["scheduler"]
    trainer = test_env["trainer"]

    # Run 1: promote first model (accuracy 0.95)
    scheduler.state = TrainingSchedulerState.ELIGIBLE
    scheduler.current_accuracy = 0.900
    trainer.held_out_accuracy = 0.950
    scheduler.run_chunk(max_steps=20)
    assert scheduler.last_promotion_status == "promoted"

    # Run 2: promote second model (accuracy 0.98), backing up first model
    scheduler.state = TrainingSchedulerState.ELIGIBLE
    trainer.held_out_accuracy = 0.980
    trainer.step = 0
    scheduler.current_step = 0
    scheduler.run_chunk(max_steps=20)
    assert scheduler.last_promotion_status == "promoted"
    assert scheduler.previous_adapter_dir.exists()

    # Roll back to first model
    ok, msg = scheduler.rollback_adapter()
    assert ok is True
    assert "successfully rolled back" in msg.lower()
    assert scheduler.last_promotion_status == "rolled_back"


def test_voice_status_and_dispatcher_integration(test_env):
    """Voice status query returns natural spoken responses and fast-path routes properly."""
    scheduler = test_env["scheduler"]
    ladder = DecisionLadder(hardware_tier="cpu")
    dispatcher = Dispatcher()
    dispatcher.set_training_scheduler(scheduler)

    # 1. Query pattern matching in DecisionLadder
    res1 = ladder.check_pattern_fast_path("How's my voice training going?")
    assert res1 is not None
    assert res1.rung == DecisionRung.RUNG_1_PATTERN
    assert res1.tool == "voice_training_status"

    res2 = ladder.check_pattern_fast_path("Grace, go back to the old voice model")
    assert res2 is not None
    assert res2.rung == DecisionRung.RUNG_1_PATTERN
    assert res2.tool == "rollback_voice_model"

    # 2. Voice summary strings
    scheduler.state = TrainingSchedulerState.TRAINING
    scheduler.current_step = 10
    scheduler.total_steps = 20
    scheduler.chunks_completed = 2
    msg = scheduler.get_voice_status_summary()
    assert "50%" in msg
    assert "2 idle chunks" in msg

    scheduler.state = TrainingSchedulerState.PAUSED
    scheduler.pause_reason = "Machine running on battery (42% remaining)"
    msg = scheduler.get_voice_status_summary()
    assert "battery" in msg.lower()

    # 3. Dispatch execution
    import asyncio
    dispatch_res = asyncio.run(dispatcher.execute(Intent(tool="voice_training_status", params={})))
    assert dispatch_res["status"] == "ok"
    assert "battery" in dispatch_res["text"].lower()
    assert "data" in dispatch_res
    assert dispatch_res["data"]["state"] == "paused"
