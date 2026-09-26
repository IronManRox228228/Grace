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
        assert result is False
        assert len(called) == 0

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


class TestAudioClock:
    """Silence measured in audio rather than wall time.

    This is what makes the tape corpus a usable oracle: replay bursts chunks at
    whatever rate the scheduler allows, and under a wall clock the turn then
    ends on a different chunk from run to run.
    """

    RATE = 16000
    WIDTH = 2
    # 512 samples = 32ms at 16kHz. The generator's chunk size.
    CHUNK = 1024

    def _detector(self, silence_ms=275):
        return VadDetector(
            threshold=0.008,
            silence_duration_ms=silence_ms,
            sample_rate=self.RATE,
            sample_width=self.WIDTH,
        )

    def _speech(self):
        import struct
        return struct.pack("<" + "h" * (self.CHUNK // 2), *([6000] * (self.CHUNK // 2)))

    def _silence(self):
        return bytes(self.CHUNK)

    def test_wall_clock_is_still_the_default(self):
        assert VadDetector().uses_audio_clock is False
        assert self._detector().uses_audio_clock is True

    def test_turn_ends_on_a_fixed_chunk_regardless_of_delivery_speed(self):
        # 32ms per chunk, 275ms window. The first silent chunk only marks where
        # silence began - the same convention the wall clock follows - so the
        # 10th is the first to reach the window (9 * 32 = 288ms). No sleeping
        # anywhere, which is the point.
        detector = self._detector()
        detector.process_chunk(self._speech())

        ended_at = None
        for n in range(1, 21):
            if detector.process_chunk(self._silence()):
                ended_at = n
                break
        assert ended_at == 10, f"turn ended on silent chunk {ended_at}, expected 10"

    def test_the_result_does_not_move_when_delivery_is_slow(self):
        # The same sequence with real time passing between chunks. A wall clock
        # would end the turn on the first or second chunk here; the audio clock
        # must not notice.
        import time as _time

        detector = self._detector()
        detector.process_chunk(self._speech())

        ended_at = None
        for n in range(1, 21):
            _time.sleep(0.02)
            if detector.process_chunk(self._silence()):
                ended_at = n
                break
        assert ended_at == 10

    def test_a_generated_turn_closes_with_chunks_to_spare(self):
        # The generator emits 6 speech + 20 silent chunks per turn. The failure
        # this replaces was the audio running out before the turn closed, so the
        # margin is the assertion, not just the fact that it closed.
        detector = self._detector()
        for _ in range(6):
            assert detector.process_chunk(self._speech()) is False

        closed_after = None
        for n in range(1, 21):
            if detector.process_chunk(self._silence()):
                closed_after = n
                break
        assert closed_after is not None, "the turn never closed within the recorded audio"
        assert 20 - closed_after >= 10, (
            f"only {20 - closed_after} chunks of margin; the tail is too tight to "
            f"survive a chunk-size change"
        )

    def test_the_clock_survives_a_reset_between_turns(self):
        # reset() starts a new turn but must not rewind the clock, or the second
        # turn of a session would inherit the first turn's silence.
        detector = self._detector()
        detector.process_chunk(self._speech())
        for _ in range(9):
            detector.process_chunk(self._silence())

        elapsed = detector._audio_seconds
        detector.reset()
        assert detector._audio_seconds == elapsed
        assert detector._state.total_silence_ms == 0.0

        detector.process_chunk(self._speech())
        ended_at = None
        for n in range(1, 21):
            if detector.process_chunk(self._silence()):
                ended_at = n
                break
        assert ended_at == 10, "the second turn did not need the same silence as the first"
