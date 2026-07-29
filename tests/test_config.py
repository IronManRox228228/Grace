"""Test config module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.config import Config


class TestConfig:
    """Test suite for Config."""

    def test_init_with_defaults(self):
        """Test config initializes with default values."""
        config = Config()
        assert config.llama_port == 8080
        assert config.llama_host == "127.0.0.1"
        assert config.llama_context_window == 8192


        assert config.llama_ngl == 999
        assert config.llama_cache_type_k == "f16"
        assert config.llama_cache_type_v == "f16"

        assert config.mic_sample_rate == 16000
        assert config.mic_chunk == 512
        assert config.mic_channels == 1
        assert config.mic_width == 2
        assert config.vosk_keyword == "grace"
        assert config.vosk_threshold == 0.4
        assert config.whisper_vad_threshold == 0.008
        assert config.whisper_silence_duration_ms == 700
        assert config.kokoro_workers == 3
        assert config.kokoro_device == "cuda"
        assert config.kokoro_dtype == "float16"

    def test_llama_server_url(self):
        """Test derived server URL property."""
        config = Config()
        assert config.llama_server_url == "http://127.0.0.1:8080"

    def test_model_paths_exist(self):
        """Test that model paths are non-empty strings."""
        config = Config()
        assert len(config.llama_model_path) > 0
        assert len(config.kokoro_model_path) > 0
        assert len(config.kokoro_voices_path) > 0
