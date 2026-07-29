"""Audio feedback for Grace - activation chime and system sounds.

Generates soft, modern chime tones for wake word activation
and other system feedback events. Uses sounddevice for playback.
"""

import logging
import os

import numpy as np
import sounddevice as sd

logger = logging.getLogger("grace.feedback")


_CUSTOM_CHIME_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "frontend", "soundshelfstudio-ui-chime-confirm-567486.mp3"
)
_CACHED_CHIME_SAMPLES = None
_CACHED_CHIME_RATE = 48000


class FeedbackSounds:
    """Generates and plays audio feedback tones."""

    @staticmethod
    def play_chime(duration: float = 0.8, volume: float = 0.4) -> None:
        """Play activation chime from custom MP3 asset file."""
        global _CACHED_CHIME_SAMPLES, _CACHED_CHIME_RATE

        try:
            if _CACHED_CHIME_SAMPLES is None:
                if os.path.exists(_CUSTOM_CHIME_PATH):
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(_CUSTOM_CHIME_PATH)
                    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    # Normalize to [-1.0, 1.0]
                    max_val = float(1 << (8 * audio.sample_width - 1))
                    samples = samples / max_val
                    if audio.channels > 1:
                        samples = samples.reshape((-1, audio.channels))
                    _CACHED_CHIME_SAMPLES = samples * volume
                    _CACHED_CHIME_RATE = audio.frame_rate

            if _CACHED_CHIME_SAMPLES is not None:
                sd.play(_CACHED_CHIME_SAMPLES, _CACHED_CHIME_RATE)
                sd.wait()
                logger.debug("Custom chime MP3 played")
                return
        except Exception as exc:
            logger.warning(f"Custom chime playback failed: {exc}, falling back to tone")

        # Fallback synthetic tone
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        tone1 = np.sin(2 * np.pi * 523.25 * t)
        tone2 = np.sin(2 * np.pi * 659.25 * t) * 0.5
        fade_len = int(sample_rate * 0.05)
        envelope = np.ones(len(t))
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        audio = (tone1 + tone2) * envelope * np.exp(-2 * t) * volume / 2
        sd.play(audio, sample_rate)
        sd.wait()
        logger.debug("Fallback chime played")
