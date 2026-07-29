"""Test TTS player."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.tts.player import TTSPlayer


class TestTTSPlayer:
    """Test suite for TTSPlayer."""

    def test_init(self):
        """Test player initialization."""
        player = TTSPlayer()
        assert player.queue_size == 0

    def test_stop_without_play(self):
        """Test stop() without prior play."""
        player = TTSPlayer()
        player.stop()  # Should not raise

    def test_play_sync_invalid_wav(self):
        """Test play_sync() with invalid WAV data."""
        player = TTSPlayer()
        # Invalid WAV data - should not crash
        player.play_sync(b"not a wav file")
