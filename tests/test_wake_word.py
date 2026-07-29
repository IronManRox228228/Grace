"""Test wake word detector module."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.audio.wake_word import WakeWordDetector


class TestWakeWordDetector:
    """Test suite for WakeWordDetector."""

    def test_init_defaults(self):
        """Test that WakeWordDetector initializes with defaults."""
        detector = WakeWordDetector(model_path="dummy_path")
        assert detector.keyword == "grace"
        assert detector.threshold == 0.8
        assert detector.sample_rate == 16000
        assert not detector.is_running
        assert not detector.detected

    def test_set_keyword(self):
        """Test keyword getter and setter."""
        detector = WakeWordDetector(model_path="dummy_path")
        detector.keyword = "james"
        assert detector.keyword == "james"

    def test_set_threshold(self):
        """Test threshold getter and setter."""
        detector = WakeWordDetector(model_path="dummy_path")
        assert detector.threshold == 0.8
        detector.threshold = 0.95
        assert detector.threshold == 0.95

    def test_set_callback(self):
        """Test callback assignment."""
        detector = WakeWordDetector(model_path="dummy_path")
        called = []

        def on_detect():
            called.append(True)

        detector.set_callback(on_detect)
        assert detector._callback == on_detect

    def test_stop_without_start(self):
        """Test that stop() works even without start()."""
        detector = WakeWordDetector(model_path="dummy_path")
        detector.stop()  # Should not raise
        assert not detector.is_running

    def test_reset(self):
        """Test reset clears detection event."""
        detector = WakeWordDetector(model_path="dummy_path")
        detector._event.set()
        detector.reset()
        assert not detector.detected
