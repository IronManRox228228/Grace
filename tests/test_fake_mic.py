"""The dev-only fake mic plays dropped WAVs, then silence."""

import os
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.audio.fake_mic import FakeMicCapture


def _wav(path, frames: bytes, rate=16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


def test_plays_a_dropped_file_then_silence(tmp_path):
    mic = FakeMicCapture(str(tmp_path), chunk_size=4, lead_in_s=0).start()
    assert mic.get_chunk() == b"\x00" * 8  # nothing dropped yet
    _wav(os.path.join(tmp_path, "a.wav"), b"\x01\x02" * 6)
    assert mic.get_chunk() == b"\x01\x02" * 4
    assert mic.get_chunk() == b"\x01\x02" * 2 + b"\x00" * 4  # padded tail
    assert mic.get_chunk() == b"\x00" * 8
    assert os.path.exists(os.path.join(tmp_path, "played", "a.wav"))


def test_wrong_format_is_skipped_not_played(tmp_path):
    mic = FakeMicCapture(str(tmp_path), chunk_size=4, lead_in_s=0).start()
    _wav(os.path.join(tmp_path, "b.wav"), b"\x05\x05" * 8, rate=24000)
    assert mic.get_chunk() == b"\x00" * 8
    assert os.path.exists(os.path.join(tmp_path, "played", "b.wav"))


def test_each_file_starts_after_a_second_of_silence(tmp_path):
    mic = FakeMicCapture(str(tmp_path), sample_rate=8, chunk_size=4).start()
    _wav(os.path.join(tmp_path, "c.wav"), b"\x01\x02" * 4, rate=8)
    assert mic.get_chunk() == b"\x00" * 8  # 1 s at 8 Hz = 8 frames = 2 chunks
    assert mic.get_chunk() == b"\x00" * 8
    assert mic.get_chunk() == b"\x01\x02" * 4

