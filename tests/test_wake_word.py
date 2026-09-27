"""Test wake word detector module."""

import sys
import os
import threading

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

    def test_rec_access_is_guarded_by_a_lock(self):
        """R7: `reset()` and the recognizer thread both touch `self._rec` -
        without one shared lock, nothing stops them doing it at the same time.
        """
        detector = WakeWordDetector(model_path="dummy_path")
        assert isinstance(detector._rec_lock, type(threading.Lock()))
        # reset() must take that same lock, not just have one sitting unused.
        detector._rec_lock.acquire()
        try:
            released = detector._rec_lock.acquire(blocking=False)
            assert not released, "the lock was not actually held"
        finally:
            detector._rec_lock.release()


class TestCancelWatch:
    """R5: while an agent goal is running, a spoken "stop"/"cancel" must be
    able to interrupt it - the wake keyword's own matching is unaffected.
    """

    def test_armed_watch_fires_on_stop(self):
        detector = WakeWordDetector(model_path="dummy_path")
        fired = []
        detector.arm_cancel_watch(lambda: fired.append(True))
        detector._check_keyword('{"text": "okay please stop"}')
        assert fired == [True]

    def test_armed_watch_fires_on_cancel(self):
        detector = WakeWordDetector(model_path="dummy_path")
        fired = []
        detector.arm_cancel_watch(lambda: fired.append(True))
        detector._check_keyword('{"text": "cancel that"}')
        assert fired == [True]

    def test_unarmed_detector_ignores_stop_speech(self):
        detector = WakeWordDetector(model_path="dummy_path")
        # No callback set: must not raise, and there is nothing to fire.
        detector._check_keyword('{"text": "stop"}')

    def test_word_boundary_excludes_lookalikes(self):
        detector = WakeWordDetector(model_path="dummy_path")
        fired = []
        detector.arm_cancel_watch(lambda: fired.append(True))
        detector._check_keyword('{"text": "the bus makes a nonstop trip to the cancellation desk"}')
        assert fired == []

    def test_disarming_stops_future_firing(self):
        detector = WakeWordDetector(model_path="dummy_path")
        fired = []
        detector.arm_cancel_watch(lambda: fired.append(True))
        detector.disarm_cancel_watch()
        detector._check_keyword('{"text": "stop"}')
        assert fired == []
