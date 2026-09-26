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
    def play_chime(duration: float = 0.8, volume: float = 0.4, blocking: bool = False) -> None:
        """Play the activation chime.

        Non-blocking by default: this runs on the asyncio thread right before
        listening starts, and waiting out the ~0.8 s chime there delayed the
        start of capture by the full length of the sound.
        """
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
                if blocking:
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
        try:
            sd.play(audio, sample_rate)
            if blocking:
                sd.wait()
        except Exception as e:
            logger.debug(f"Chime playback failed: {e}")
        logger.debug("Fallback chime played")

    @staticmethod
    def play_success(duration: float = 0.18, volume: float = 0.25, blocking: bool = False) -> None:
        """Gentle rising two-tone earcon confirming an action finished (C5 -> E5)."""
        try:
            sample_rate = 24000
            t_half = duration / 2.0
            n1 = int(sample_rate * t_half)
            n2 = int(sample_rate * t_half)
            t1 = np.linspace(0, t_half, n1, endpoint=False)
            t2 = np.linspace(0, t_half, n2, endpoint=False)

            part1 = np.sin(2 * np.pi * 523.25 * t1) * np.exp(-3 * t1)
            part2 = np.sin(2 * np.pi * 659.25 * t2) * np.exp(-3 * t2)
            audio = np.concatenate([part1, part2]) * volume
            sd.play(audio, sample_rate)
            if blocking:
                sd.wait()
        except Exception as exc:
            logger.debug(f"play_success failed: {exc}")

    @staticmethod
    def play_cancel(duration: float = 0.18, volume: float = 0.25, blocking: bool = False) -> None:
        """Gentle descending two-tone earcon confirming cancellation/stop (E5 -> C5)."""
        try:
            sample_rate = 24000
            t_half = duration / 2.0
            n1 = int(sample_rate * t_half)
            n2 = int(sample_rate * t_half)
            t1 = np.linspace(0, t_half, n1, endpoint=False)
            t2 = np.linspace(0, t_half, n2, endpoint=False)

            part1 = np.sin(2 * np.pi * 659.25 * t1) * np.exp(-3 * t1)
            part2 = np.sin(2 * np.pi * 523.25 * t2) * np.exp(-3 * t2)
            audio = np.concatenate([part1, part2]) * volume
            sd.play(audio, sample_rate)
            if blocking:
                sd.wait()
        except Exception as exc:
            logger.debug(f"play_cancel failed: {exc}")

    @staticmethod
    def play_error(duration: float = 0.22, volume: float = 0.25, blocking: bool = False) -> None:
        """Gentle low-frequency double tone indicating an error or no-speech timeout."""
        try:
            sample_rate = 24000
            t_pulse = 0.09
            n_pulse = int(sample_rate * t_pulse)
            n_gap = int(sample_rate * 0.04)
            t1 = np.linspace(0, t_pulse, n_pulse, endpoint=False)

            pulse = np.sin(2 * np.pi * 261.63 * t1) * np.exp(-4 * t1)
            gap = np.zeros(n_gap, dtype=np.float32)
            audio = np.concatenate([pulse, gap, pulse]) * volume
            sd.play(audio, sample_rate)
            if blocking:
                sd.wait()
        except Exception as exc:
            logger.debug(f"play_error failed: {exc}")

    @staticmethod
    def play_listening(duration: float = 0.06, volume: float = 0.18, blocking: bool = False) -> None:
        """Subtle high blip indicating Grace is listening in follow-up mode."""
        try:
            sample_rate = 24000
            n = int(sample_rate * duration)
            t = np.linspace(0, duration, n, endpoint=False)
            audio = np.sin(2 * np.pi * 880.0 * t) * np.exp(-8 * t) * volume
            sd.play(audio, sample_rate)
            if blocking:
                sd.wait()
        except Exception as exc:
            logger.debug(f"play_listening failed: {exc}")
