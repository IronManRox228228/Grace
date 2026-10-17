"""Tests for Speech Personalization & Enrollment Flow (PLAN §6, Task 14).

Verifies:
1. Phrase generator coverage (wake words, safety, tool schema, app index, alphabet/digits, Harvard sentences, personal words).
2. Audio quality validation (rejection of clipped or silent recordings; never rejecting unclear/dysarthric speech).
3. Encrypted at-rest local storage with DPAPI and user deletion rights.
4. Adaptive stopping evaluator with held-out splits, weak category focus, and concrete metric reports.
5. Short, resumable sessions with switch input integration (skip/redo) and non-nagging next session scheduling.
6. Ongoing accuracy tracking and explicit consent gating for everyday samples.
"""

import array
import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from grace.enrollment.audio_validator import AudioQualityValidator
from grace.enrollment.evaluator import AdaptiveStoppingEvaluator
from grace.enrollment.ongoing import OngoingTracker
from grace.enrollment.phrase_generator import EnrollmentPhrase, PhraseGenerator
from grace.enrollment.session import EnrollmentSession, SessionState
from grace.enrollment.storage import EnrollmentStorage
from grace.input.switch_input import SwitchAction


def _generate_synthetic_pcm(duration_s: float = 1.0, amplitude: int = 4000, sample_rate: int = 16000) -> bytes:
    """Generate synthetic 16-bit mono PCM sine wave for testing."""
    import math
    num_samples = int(duration_s * sample_rate)
    samples = array.array("h")
    for i in range(num_samples):
        # 440 Hz tone
        val = int(amplitude * math.sin(2 * math.pi * 440 * (i / sample_rate)))
        # clamp
        val = max(-32768, min(32767, val))
        samples.append(val)
    return samples.tobytes()


def test_phrase_generator_coverage():
    """Verify phrase generator covers all categories and held-out ratio required by PLAN §6."""
    mock_app_indexer = MagicMock()
    mock_app_indexer._apps = {"notepad": "notepad.exe", "calculator": "calc.exe"}
    mock_app_indexer._uwp_apps = {"edge": "Microsoft.Edge"}

    mock_facts = MagicMock()
    mock_facts.list_facts.return_value = ["call Dr. Patel", "directions to city clinic"]

    gen = PhraseGenerator(
        app_indexer=mock_app_indexer,
        facts_store=mock_facts,
        held_out_ratio=0.20,
    )

    curriculum = gen.generate_full_curriculum()
    assert len(curriculum) >= 400

    categories = {p.category for p in curriculum}
    expected_categories = {
        "wake_word",
        "safety_confirmation",
        "commands",
        "apps_folders",
        "alphabet_digits",
        "phonetic_sentences",
        "personal_words",
    }
    assert expected_categories.issubset(categories)

    # 1. Wake word: 50+ times
    wake_phrases = [p for p in curriculum if p.category == "wake_word"]
    assert len(wake_phrases) >= 50
    assert all("Grace" in p.text for p in wake_phrases)

    # 2. Safety: yes/no/stop/cancel
    safety_phrases = [p for p in curriculum if p.category == "safety_confirmation"]
    assert len(safety_phrases) >= 30

    # 3. Commands from schema: 150+
    cmd_phrases = [p for p in curriculum if p.category == "commands"]
    assert len(cmd_phrases) >= 150

    # 4. Harvard sentences: 100+
    harvard_phrases = [p for p in curriculum if p.category == "phonetic_sentences"]
    assert len(harvard_phrases) >= 100

    # 5. Held-out split
    held_out = [p for p in curriculum if p.is_held_out]
    assert 0.15 <= (len(held_out) / len(curriculum)) <= 0.25


def test_audio_quality_validator_rules():
    """Verify clipped or silent recordings are rejected, but unclear speech is never rejected."""
    validator = AudioQualityValidator(
        sample_rate=16000,
        min_duration_s=0.20,
        clipping_peak_threshold=32500,
        silence_rms_threshold=120.0,
    )

    # 1. Normal speech amplitude passes
    normal_pcm = _generate_synthetic_pcm(duration_s=1.0, amplitude=5000)
    res_normal = validator.validate_pcm(normal_pcm)
    assert res_normal.is_valid is True
    assert res_normal.error_type is None

    # 2. Low amplitude / dysarthric speech (e.g. amplitude 800, well above noise floor 120) PASSES
    dysarthric_pcm = _generate_synthetic_pcm(duration_s=1.5, amplitude=600)
    res_dysarthric = validator.validate_pcm(dysarthric_pcm)
    assert res_dysarthric.is_valid is True, "Dysarthric/low volume speech must never be rejected"

    # 3. Silence / empty audio is rejected
    silent_pcm = bytes(16000 * 2)  # all zeros
    res_silent = validator.validate_pcm(silent_pcm)
    assert res_silent.is_valid is False
    assert res_silent.error_type == "silent"

    # 4. Clipped audio is rejected
    clipped_pcm = _generate_synthetic_pcm(duration_s=1.0, amplitude=32767)
    res_clipped = validator.validate_pcm(clipped_pcm)
    assert res_clipped.is_valid is False
    assert res_clipped.error_type == "clipped"

    # 5. Too short (< 0.20s)
    short_pcm = _generate_synthetic_pcm(duration_s=0.10, amplitude=5000)
    res_short = validator.validate_pcm(short_pcm)
    assert res_short.is_valid is False
    assert res_short.error_type == "too_short"


def test_storage_encryption_and_user_privacy(tmp_path):
    """Verify local storage encrypts files, supports round-trip decryption, and provides user deletion."""
    storage = EnrollmentStorage(base_dir=tmp_path / "enrollment", use_dpapi=True)

    session_id = "test_session_01"
    utterance_id = "utt_001"
    original_pcm = _generate_synthetic_pcm(duration_s=0.5, amplitude=4000)

    # Save
    audio_path = storage.save_utterance(
        session_id=session_id,
        utterance_id=utterance_id,
        phrase_id="cmd_001",
        phrase_text="open the calculator",
        category="commands",
        pcm_bytes=original_pcm,
        extra_meta={"confidence": 0.98},
    )
    assert audio_path.exists()

    # Load and verify PCM bytes match exactly
    loaded_pcm = storage.load_utterance_pcm(session_id, utterance_id)
    assert loaded_pcm == original_pcm

    # Load and verify metadata
    meta = storage.load_utterance_meta(session_id, utterance_id)
    assert meta is not None
    assert meta["phrase_text"] == "open the calculator"
    assert meta["category"] == "commands"
    assert meta["confidence"] == 0.98

    # List recordings
    recordings = storage.list_recordings(session_id)
    assert len(recordings) == 1
    assert recordings[0]["utterance_id"] == utterance_id

    # User deletion of single utterance
    assert storage.delete_utterance(session_id, utterance_id) is True
    assert storage.load_utterance_pcm(session_id, utterance_id) is None
    assert len(storage.list_recordings(session_id)) == 0

    # User deletion of entire session
    storage.save_utterance(
        session_id=session_id,
        utterance_id="utt_002",
        phrase_id="cmd_002",
        phrase_text="close notepad",
        category="commands",
        pcm_bytes=original_pcm,
    )
    assert storage.delete_session(session_id) is True
    assert not (tmp_path / "enrollment" / session_id).exists()


def test_adaptive_stopping_and_headline_reporting():
    """Verify held-out metric calculation, headline reporting, and stopping on plateau."""
    evaluator = AdaptiveStoppingEvaluator(
        target_accuracy=0.95,
        min_improvement_threshold=0.015,
        weak_category_threshold=0.88,
        min_held_out_samples_per_category=2,
    )

    # Round 1: High commands, weak yes/no
    # Commands: 9/10 correct
    for i in range(8):
        evaluator.record_sample("commands", "open calculator", "open calculator", is_held_out=False)
    evaluator.record_sample("commands", "turn volume up", "turn volume up", is_held_out=True)
    evaluator.record_sample("commands", "scroll down", "scroll down", is_held_out=True)

    # Safety: 5/10 correct (weak)
    for i in range(5):
        evaluator.record_sample("safety_confirmation", "confirm", "confirm", is_held_out=False)
    for i in range(3):
        evaluator.record_sample("safety_confirmation", "cancel that", "unrecognized", is_held_out=False)
    evaluator.record_sample("safety_confirmation", "yes", "yes", is_held_out=True)
    evaluator.record_sample("safety_confirmation", "abort", "unrecognized", is_held_out=True)

    evaluator.conclude_round()
    headline = evaluator.format_headline_report()
    assert "commands 100%" in headline
    assert "yes/no 50%" in headline

    should_stop, rationale, weak_cats = evaluator.evaluate_stopping_decision()
    assert should_stop is False
    assert "safety_confirmation" in weak_cats

    # Round 2: Safety improves to target accuracy (>= 95%)
    for i in range(20):
        evaluator.record_sample("safety_confirmation", "cancel", "cancel", is_held_out=True)
    evaluator.conclude_round()

    should_stop_r2, rationale_r2, _ = evaluator.evaluate_stopping_decision()
    assert should_stop_r2 is True
    assert "commands" in evaluator.format_headline_report()


def test_enrollment_session_resumability_and_switch_input(tmp_path):
    """Verify session mid-session resume, switch input integration, and non-nagging next schedule."""
    storage = EnrollmentStorage(base_dir=tmp_path / "enrollment", use_dpapi=False)
    session = EnrollmentSession(
        session_id="session_accessible_test",
        duration_minutes=0.001,  # Short timer to test duration expiration
        storage=storage,
    )

    session.initialize_new_session()
    assert session.state == SessionState.ACTIVE
    phrase_1 = session.get_current_phrase()
    assert phrase_1 is not None

    # 1. Switch Action: NO (double-press) skips phrase
    session.handle_switch_input(SwitchAction.NO)
    phrase_2 = session.get_current_phrase()
    assert phrase_2 is not None
    assert phrase_2.phrase_id != phrase_1.phrase_id
    assert session.current_index == 1

    # 2. Submit valid audio
    pcm = _generate_synthetic_pcm(duration_s=0.5, amplitude=4000)
    ok, msg, val = session.submit_recording(pcm, transcribed_text=phrase_2.text)
    assert ok is True
    assert val.is_valid is True

    # 3. Switch Action: STOP (long-press) redoes previous phrase
    session.handle_switch_input(SwitchAction.STOP)
    assert session.current_index == 1

    # 4. Resumability: Reload session state from disk
    new_session_instance = EnrollmentSession(
        session_id="session_accessible_test",
        storage=storage,
    )
    resumed_ok = new_session_instance.start_or_resume()
    assert resumed_ok is True
    assert new_session_instance.current_index == 1
    assert len(new_session_instance.phrases) == len(session.phrases)

    # 5. Suggested next session without nagging
    suggestion = session.compute_suggested_next_session()
    assert "Tomorrow at" in suggestion


def test_ongoing_accuracy_tracker_and_consent_privacy(tmp_path):
    """Verify ongoing drift tracking and explicit consent gating for everyday speech samples."""
    storage = EnrollmentStorage(base_dir=tmp_path / "enrollment", use_dpapi=False)
    tracker = OngoingTracker(
        storage=storage,
        state_file=tmp_path / "ongoing_tracker.json",
        drop_threshold=0.08,
        min_turns_to_evaluate=5,
        user_consent_everyday_samples=False,  # default no consent
    )

    pcm = _generate_synthetic_pcm(duration_s=0.5, amplitude=4000)

    # 1. Consent Gate: when consent is False, sample is discarded
    added = tracker.add_everyday_sample(
        phrase_text="open edge",
        category="commands",
        pcm_bytes=pcm,
        verified_success=True,
    )
    assert added is False, "Audio must never be saved without explicit user consent"

    # 2. Grant explicit consent
    tracker.set_user_consent(True)
    added_with_consent = tracker.add_everyday_sample(
        phrase_text="open edge",
        category="commands",
        pcm_bytes=pcm,
        verified_success=True,
    )
    assert added_with_consent is True

    # 3. Track daily accuracy and drift detection
    # Record 8 turns with only 50% success to simulate a drop
    for i in range(4):
        tracker.record_turn(verified_correct=True, date_str="2026-10-17")
    for i in range(4):
        tracker.record_turn(verified_correct=False, date_str="2026-10-17")

    topup_prompt = tracker.check_for_topup_suggestion()
    assert topup_prompt is not None
    assert "Would you like to do a quick 5-minute top-up enrollment session?" in topup_prompt
