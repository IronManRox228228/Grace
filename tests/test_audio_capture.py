"""Test audio capture module."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.audio.capture import AudioCapture


class TestAudioCapture:
    """Test suite for AudioCapture."""

    def test_init_defaults(self):
        """Test that AudioCapture initializes with correct defaults."""
        capture = AudioCapture()
        assert capture.is_running is False

    def test_list_devices(self):
        """Test that device listing works (may be empty on CI)."""
        capture = AudioCapture()
        devices = capture.list_devices()
        # Should return a list, even if empty
        assert isinstance(devices, list)

    def test_get_device_count(self):
        """Test device count."""
        capture = AudioCapture()
        count = capture.get_device_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_rms_empty_chunk(self):
        """Test RMS calculation on empty data."""
        capture = AudioCapture()
        assert capture.get_rms(b"") == 0.0

    def test_rms_nonempty_chunk(self):
        """Test RMS calculation on non-empty data."""
        capture = AudioCapture()
        import struct
        # Create a silent chunk (all zeros)
        samples = bytes([0] * 1024)
        rms = capture.get_rms(samples)
        assert rms == 0.0

    def test_rms_with_signal(self):
        """Test RMS calculation with a signal."""
        capture = AudioCapture()
        # Create a chunk with a non-zero signal (max amplitude)
        samples = b"\xff\xff" * 128  # 128 samples at -1 amplitude
        rms = capture.get_rms(samples)
        assert rms > 0.0

    def test_stop_without_start(self):
        """Test that stop() works even without start()."""
        capture = AudioCapture()
        capture.stop()  # Should not raise
        assert not capture.is_running

    def test_close_without_start(self):
        """Test that close() works even without start()."""
        capture = AudioCapture()
        capture.close()  # Should not raise
        assert capture._pyaudio is None
