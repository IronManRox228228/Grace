"""Tests for Sherpa-ONNX engine adapters (PLAN.md §4, Task 08)."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from grace.audio.sherpa_engine import (
    SherpaASR,
    SherpaTTS,
    SherpaVAD,
    SherpaKWS,
    _require_sherpa,
    _verify_model_file,
)


def test_missing_model_file_raises_with_models_md_reference():
    with pytest.raises(FileNotFoundError) as exc_info:
        _verify_model_file("nonexistent/path/to/model.onnx", "test component")
    assert "docs/MODELS.md" in str(exc_info.value)
    assert "test component" in str(exc_info.value)


def test_missing_sherpa_import_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    with pytest.raises(ImportError) as exc_info:
        _require_sherpa()
    assert "pip install sherpa-onnx" in str(exc_info.value)


def test_sherpa_asr_missing_model_raises_at_transcribe(tmp_path):
    asr = SherpaASR(model_path=str(tmp_path / "missing_dir"))
    with pytest.raises(FileNotFoundError) as exc_info:
        asr.transcribe_bytes(b"\x00" * 3200)
    assert "docs/MODELS.md" in str(exc_info.value)


def test_sherpa_tts_missing_model_raises_at_generate(tmp_path):
    tts = SherpaTTS(model_path=str(tmp_path / "missing_dir"))
    with pytest.raises(FileNotFoundError) as exc_info:
        tts.generate_and_speak("Hello world")
    assert "docs/MODELS.md" in str(exc_info.value)


def test_sherpa_vad_missing_model_raises_at_process(tmp_path):
    vad = SherpaVAD(model_path=str(tmp_path / "missing_vad.onnx"))
    with pytest.raises(FileNotFoundError) as exc_info:
        vad.process_chunk(b"\x00" * 1024)
    assert "docs/MODELS.md" in str(exc_info.value)


def test_sherpa_kws_missing_model_raises_at_init(tmp_path):
    kws = SherpaKWS(
        model_path=str(tmp_path / "missing_kws"),
        keywords_file=str(tmp_path / "missing_kws" / "keywords.txt"),
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        kws._initialize()
    assert "docs/MODELS.md" in str(exc_info.value)


def test_sherpa_asr_mocked_transcription(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.int8.onnx").write_bytes(b"dummy")
    (model_dir / "tokens.txt").write_text("token")

    fake_sherpa = MagicMock()
    mock_recognizer = MagicMock()
    mock_stream = MagicMock()
    mock_stream.result.text = "open notepad"
    mock_recognizer.create_stream.return_value = mock_stream
    fake_sherpa.OfflineRecognizer.from_nemo_ctc.return_value = mock_recognizer

    with patch("grace.audio.sherpa_engine._require_sherpa", return_value=fake_sherpa):
        asr = SherpaASR(model_path=str(model_dir), model_type="parakeet")
        res = asr.transcribe_bytes(b"\x00" * 3200)
        assert res == "open notepad"
        mock_recognizer.decode_stream.assert_called_once_with(mock_stream)
