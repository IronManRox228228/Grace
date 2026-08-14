"""Whisper model resolution.

scripts/download_models.py fetches the model with cache_dir=<repo>/models, but
WhisperStreaming used to construct WhisperModel with no download_root - so
faster-whisper looked in ~/.cache/huggingface, found nothing, and with
HF_HUB_OFFLINE=1 the first transcription raised LocalEntryNotFoundError and
took the whole process down.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.stt import whisper_stream
from grace.stt.whisper_stream import WhisperStreaming

REPO_MODELS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models")
)


class FakeWhisperModel:
    """Records how faster-whisper was constructed."""

    calls = []

    def __init__(self, model_path, **kwargs):
        type(self).calls.append({"model_path": model_path, **kwargs})
        if kwargs.get("_explode"):
            raise RuntimeError("boom")


@pytest.fixture
def fake_backend(monkeypatch):
    FakeWhisperModel.calls = []

    fake_module = type(sys)("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    fake_torch = type(sys)("torch")
    fake_torch.cuda = type("cuda", (), {"is_available": staticmethod(lambda: False)})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    return FakeWhisperModel


class TestDownloadRoot:
    def test_defaults_to_the_repo_models_directory(self):
        assert WhisperStreaming("small")._download_root == REPO_MODELS

    def test_explicit_root_is_respected(self):
        assert WhisperStreaming("small", download_root="D:/elsewhere")._download_root == "D:/elsewhere"

    def test_repo_id_model_gets_the_download_root(self, fake_backend):
        WhisperStreaming("small")._initialize()
        assert fake_backend.calls[0]["download_root"] == REPO_MODELS

    def test_explicit_directory_path_is_used_as_is(self, fake_backend, tmp_path):
        # A real directory is already a model; pointing HF's cache logic at it
        # would be wrong.
        WhisperStreaming(str(tmp_path))._initialize()
        assert "download_root" not in fake_backend.calls[0]

    def test_missing_models_directory_falls_back_to_default_cache(self, monkeypatch, fake_backend):
        monkeypatch.setattr(whisper_stream, "_default_download_root", lambda: None)
        WhisperStreaming("small")._initialize()
        assert "download_root" not in fake_backend.calls[0]

    def test_load_failure_retries_against_the_default_cache(self, monkeypatch, fake_backend):
        attempts = []

        class FailingFirst(FakeWhisperModel):
            def __init__(self, model_path, **kwargs):
                attempts.append(kwargs)
                if "download_root" in kwargs:
                    raise RuntimeError("not in repo models dir")

        sys.modules["faster_whisper"].WhisperModel = FailingFirst
        WhisperStreaming("small")._initialize()

        assert len(attempts) == 2
        assert "download_root" in attempts[0]
        assert "download_root" not in attempts[1]

    def test_initialize_is_idempotent(self, fake_backend):
        stream = WhisperStreaming("small")
        stream._initialize()
        stream._initialize()
        assert len(fake_backend.calls) == 1


class TestWarmup:
    def test_warmup_failure_is_reported_not_raised(self, monkeypatch, fake_backend):
        def explode(self):
            raise RuntimeError("no model anywhere")

        monkeypatch.setattr(WhisperStreaming, "_initialize", explode)
        # Warmup runs on a background thread at startup; raising there would be
        # invisible, and it must never block Grace from starting.
        assert WhisperStreaming("small").warmup() is False
