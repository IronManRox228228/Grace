"""Test VAD detector module."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.vad.detector import VadDetector, SilenceState


class TestVadDetector:
    """Test suite for VadDetector."""

    def test_init_defaults(self):
        """Test default initialization."""
        detector = VadDetector()
        assert detector.is_speaking is False

    def test_init_custom_params(self):
        """Test custom threshold and silence duration."""
        detector = VadDetector(threshold=0.3, silence_duration_ms=2000)
        assert detector._threshold == 0.3
        assert detector._silence_duration_ms == 2000

    def test_reset(self):
        """Test that reset clears state."""
        import time
        detector = VadDetector()
        # Manually set some state
        detector._state.is_silent = False
        detector._state.silence_start = 1000.0
        detector._state.total_silence_ms = 500.0
        detector.reset()
        assert detector._state.is_silent is True
        # silence_start should be near current time (from default_factory)
        assert abs(detector._state.silence_start - time.time()) < 2.0
        assert detector._state.total_silence_ms == 0.0

    def test_silence_callback(self):
        """Test silence callback."""
        detector = VadDetector(silence_duration_ms=100)
        called = []

        def on_silence(state):
            called.append(state)

        detector.set_silence_callback(on_silence)

        # Process a silent chunk - should NOT trigger silence callback yet
        # (we need silence_duration_ms worth of silence)
        result = detector.process_chunk(bytes(512))
        # First chunk won't trigger because we need continuous silence

    def test_speech_callback(self):
        """Test speech detection callback."""
        detector = VadDetector(threshold=0.001, silence_duration_ms=10000)
        called = []

        def on_speech(state):
            called.append(state)

        detector.set_speech_callback(on_speech)

        # Create a chunk with a signal above threshold
        import struct
        # Signed int16 samples at ~50% amplitude
        samples_raw = struct.pack("<" + "h" * 128, *([16000] * 128))
        detector.process_chunk(samples_raw)
        assert detector.is_speaking is True
