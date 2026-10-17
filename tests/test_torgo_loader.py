"""Unit tests for TORGO speech dataset loader, preprocessor, and evaluator (Task 09)."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import pytest

from bench.run_asr_torgo import MockEvaluator, evaluate_model_on_torgo
from bench.torgo import (
    TorgoDataset,
    TorgoUtterance,
    is_image_description_prompt,
    is_non_speech_prompt,
    load_torgo_dataset,
    resample_wav_to_16k_mono,
)


def create_synthetic_wav(
    path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    duration_sec: float = 0.5,
) -> Path:
    """Generate a valid PCM WAV file for synthetic testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(sample_rate * duration_sec)

    # 440Hz sine tone
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    sine = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    if channels > 1:
        samples = np.column_stack([sine] * channels).flatten()
    else:
        samples = sine

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    return path


@pytest.fixture
def fake_torgo_tree(tmp_path: Path) -> Path:
    """Create a synthetic TORGO dataset directory tree for testing."""
    torgo_root = tmp_path / "TORGO"

    # 1. Dysarthric speaker F01 (Severe)
    f01_sess = torgo_root / "F01" / "Session1"
    f01_prompts = f01_sess / "prompts"
    f01_wav = f01_sess / "wav_headMic"
    f01_prompts.mkdir(parents=True, exist_ok=True)
    f01_wav.mkdir(parents=True, exist_ok=True)

    # 0001: Valid speech, 16kHz mono
    (f01_prompts / "0001.txt").write_text("The quick brown fox", encoding="utf-8")
    create_synthetic_wav(f01_wav / "0001.wav", sample_rate=16000, channels=1)

    # 0002: Non-speech acoustic tokens -> excluded
    (f01_prompts / "0002.txt").write_text("[cough] [laughter]", encoding="utf-8")
    create_synthetic_wav(f01_wav / "0002.wav", sample_rate=16000, channels=1)

    # 0003: Image description prompt -> excluded
    (f01_prompts / "0003.txt").write_text("Describe the picture MC_001.jpg", encoding="utf-8")
    create_synthetic_wav(f01_wav / "0003.wav", sample_rate=16000, channels=1)

    # 0004: Empty prompt -> excluded
    (f01_prompts / "0004.txt").write_text("   \n", encoding="utf-8")
    create_synthetic_wav(f01_wav / "0004.wav", sample_rate=16000, channels=1)

    # 0005: Valid speech, 44.1kHz stereo (tests resampling)
    (f01_prompts / "0005.txt").write_text("feed the bird", encoding="utf-8")
    create_synthetic_wav(f01_wav / "0005.wav", sample_rate=44100, channels=2)

    # 0006: Missing WAV file -> excluded
    (f01_prompts / "0006.txt").write_text("orphan prompt text", encoding="utf-8")

    # 2. Control speaker FC01 (Control)
    fc01_sess = torgo_root / "FC01" / "Session1"
    fc01_prompts = fc01_sess / "prompts"
    fc01_wav = fc01_sess / "wav_headMic"
    fc01_prompts.mkdir(parents=True, exist_ok=True)
    fc01_wav.mkdir(parents=True, exist_ok=True)

    # 0001: Valid speech, 16kHz mono
    (fc01_prompts / "0001.txt").write_text("volume up please", encoding="utf-8")
    create_synthetic_wav(fc01_wav / "0001.wav", sample_rate=16000, channels=1)

    return torgo_root


def test_exclusion_heuristics():
    assert is_image_description_prompt("picture showing dog.jpg")
    assert is_image_description_prompt("[image: 0102.png]")
    assert is_image_description_prompt("Describe the picture on screen")
    assert not is_image_description_prompt("The quick brown fox")

    assert is_non_speech_prompt("[cough]")
    assert is_non_speech_prompt("[laughter] [sigh]")
    assert is_non_speech_prompt("xxx")
    assert is_non_speech_prompt("   ")
    assert not is_non_speech_prompt("feed the bird")


def test_load_synthetic_torgo(fake_torgo_tree: Path):
    dataset = load_torgo_dataset(fake_torgo_tree)

    # Total valid: F01 has 0001 & 0005; FC01 has 0001 -> 3 total
    assert len(dataset.utterances) == 3

    # Exclusions properly tallied
    assert dataset.exclusions["non_speech"] == 1       # 0002
    assert dataset.exclusions["image_description"] == 1 # 0003
    assert dataset.exclusions["empty_prompt"] == 1      # 0004
    assert dataset.exclusions["missing_wav"] == 1       # 0006

    # Classification & severity checks
    dysarthric_utts = dataset.get_dysarthric()
    assert len(dysarthric_utts) == 2
    for u in dysarthric_utts:
        assert u.speaker_id == "F01"
        assert u.is_dysarthric is True
        assert u.severity == "Severe"

    control_utts = dataset.get_controls()
    assert len(control_utts) == 1
    assert control_utts[0].speaker_id == "FC01"
    assert control_utts[0].is_dysarthric is False
    assert control_utts[0].severity == "Control"

    # Query methods
    assert len(dataset.get_by_speaker("F01")) == 2
    assert len(dataset.get_by_severity("Severe")) == 2
    assert len(dataset.get_by_severity("Control")) == 1
    assert len(dataset.get_by_severity("Mild")) == 0


def test_audio_resampling_and_caching(fake_torgo_tree: Path, tmp_path: Path):
    dataset = load_torgo_dataset(fake_torgo_tree)
    # Utterance 0005 is 44.1kHz stereo
    utt_stereo = next(u for u in dataset.utterances if u.prompt_id == "0005")

    cache_dir = tmp_path / "cache_16k"
    wav_16k = utt_stereo.get_wav_16k(cache_dir=cache_dir)

    assert wav_16k.is_file()
    assert wav_16k.stat().st_size > 0

    # Verify converted WAV header
    with wave.open(str(wav_16k), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2

    # Verify caching (file modification time unchanged on second access)
    mtime_before = wav_16k.stat().st_mtime_ns
    wav_16k_second = utt_stereo.get_wav_16k(cache_dir=cache_dir)
    mtime_after = wav_16k_second.stat().st_mtime_ns
    assert mtime_before == mtime_after


def test_evaluate_mock_on_torgo(fake_torgo_tree: Path, tmp_path: Path):
    dataset = load_torgo_dataset(fake_torgo_tree)
    cache_dir = tmp_path / "cache"

    evaluator = MockEvaluator("mock-test-engine")
    metrics = evaluate_model_on_torgo(evaluator, dataset, cache_dir=cache_dir)

    assert metrics["evaluated_utterances"] == 3
    assert "overall_wer" in metrics
    assert "dysarthric_wer" in metrics
    assert "control_wer" in metrics
    assert metrics["mean_rtf"] > 0.0
    assert metrics["latency_p50_ms"] > 0.0
    assert "Severe" in metrics["wer_by_severity"]
    assert "Control" in metrics["wer_by_severity"]
    assert "F01" in metrics["wer_by_speaker"]
    assert "FC01" in metrics["wer_by_speaker"]
