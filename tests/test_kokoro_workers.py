"""Test Kokoro TTS engine."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.tts.kokoro_engine import KokoroEngine, KokoroWorker


class TestKokoroEngine:
    """Test suite for KokoroEngine."""

    def test_init(self):
        """Test KokoroEngine initialization."""
        engine = KokoroEngine(
            model_path="/dummy/model",
            voices_path="/dummy/voices",
            num_workers=3,
            device="cuda",
            dtype="float16",
        )
        assert engine._num_workers == 3
        assert not engine.is_initialized

    def test_init_one_worker(self):
        """Test with single worker."""
        engine = KokoroEngine(
            model_path="/dummy",
            voices_path="/dummy",
            num_workers=1,
        )
        assert engine._num_workers == 1

    def test_synthesize_sentences_empty(self):
        """Test splitting empty text."""
        engine = KokoroEngine(
            model_path="/dummy",
            voices_path="/dummy",
        )
        sentences = engine._split_sentences("")
        assert sentences == []

    def test_split_sentences_single(self):
        """Test splitting a single sentence."""
        engine = KokoroEngine(
            model_path="/dummy",
            voices_path="/dummy",
        )
        result = engine._split_sentences("Hello world")
        assert result == ["Hello world"]

    def test_split_sentences_multiple(self):
        """Test splitting multiple sentences."""
        engine = KokoroEngine(
            model_path="/dummy",
            voices_path="/dummy",
        )
        result = engine._split_sentences("First sentence. Second sentence! Third sentence?")
        assert len(result) == 3
        assert any(s.startswith("First sentence") for s in result)
        assert any(s.startswith("Second sentence") for s in result)
        assert any(s.startswith("Third sentence") for s in result)

    def test_split_sentences_with_newlines(self):
        """Test splitting text with mixed whitespace."""
        engine = KokoroEngine(
            model_path="/dummy",
            voices_path="/dummy",
        )
        result = engine._split_sentences("Hello.  World!")
        assert len(result) == 2


class TestKokoroWorker:
    """Test suite for KokoroWorker."""

    def test_init(self):
        """Test worker initialization."""
        worker = KokoroWorker(
            model_path="/dummy/model",
            voices_path="/dummy/voices",
            device="cuda",
            dtype="float16",
        )
        assert not worker._initialized
        assert worker._pipeline is None
