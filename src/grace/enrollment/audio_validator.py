"""Audio quality validation for speech enrollment (PLAN §6, Task 14).

Hard Rule: Reject clipped or silent recordings only.
Never reject a recording for being unclear; dysarthric/unclear speech is the primary intended input.
"""

from __future__ import annotations

import array
from dataclasses import dataclass
import math
from typing import Optional, Tuple


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a user's enrollment utterance recording."""
    is_valid: bool
    error_type: Optional[str]  # "clipped", "silent", "too_short", None
    message: str
    peak: int
    rms: float
    duration_s: float


class AudioQualityValidator:
    """Validates raw 16kHz 16-bit mono PCM audio for speech enrollment."""

    def __init__(
        self,
        sample_rate: int = 16000,
        min_duration_s: float = 0.20,
        clipping_peak_threshold: int = 32500,
        clipping_ratio_threshold: float = 0.005,
        silence_rms_threshold: float = 120.0,
    ):
        self.sample_rate = sample_rate
        self.min_duration_s = min_duration_s
        self.clipping_peak_threshold = clipping_peak_threshold
        self.clipping_ratio_threshold = clipping_ratio_threshold
        self.silence_rms_threshold = silence_rms_threshold

    def validate_pcm(self, pcm_bytes: bytes) -> ValidationResult:
        """Validate raw 16-bit signed PCM audio.

        Checks:
        1. Length >= min_duration_s
        2. Clipping (peak sample or saturated sample count)
        3. Silence (RMS level)

        Explicitly does NOT check for phoneme clarity or speech intelligibility.
        """
        if not pcm_bytes:
            return ValidationResult(
                is_valid=False,
                error_type="silent",
                message="Recording is empty. Please speak after the prompt.",
                peak=0,
                rms=0.0,
                duration_s=0.0,
            )

        num_samples = len(pcm_bytes) // 2
        duration_s = num_samples / self.sample_rate

        if duration_s < self.min_duration_s:
            return ValidationResult(
                is_valid=False,
                error_type="too_short",
                message="Recording is too brief. Please repeat the phrase.",
                peak=0,
                rms=0.0,
                duration_s=duration_s,
            )

        # Parse 16-bit signed integer samples
        samples = array.array("h")
        samples.frombytes(pcm_bytes[: num_samples * 2])

        # Compute peak, saturated count, and RMS
        peak = 0
        saturated_count = 0
        sum_sq = 0.0

        for s in samples:
            abs_s = abs(s)
            if abs_s > peak:
                peak = abs_s
            if abs_s >= self.clipping_peak_threshold:
                saturated_count += 1
            sum_sq += float(s) * float(s)

        rms = math.sqrt(sum_sq / len(samples))
        clipping_ratio = saturated_count / len(samples)

        # 1. Check for clipping
        if peak >= 32760 or clipping_ratio >= self.clipping_ratio_threshold:
            return ValidationResult(
                is_valid=False,
                error_type="clipped",
                message="Audio was clipped. Please speak slightly further from the microphone.",
                peak=peak,
                rms=rms,
                duration_s=duration_s,
            )

        # 2. Check for silence
        if rms < self.silence_rms_threshold:
            return ValidationResult(
                is_valid=False,
                error_type="silent",
                message="No speech detected. Please repeat a bit louder.",
                peak=peak,
                rms=rms,
                duration_s=duration_s,
            )

        # Passes all quality checks
        return ValidationResult(
            is_valid=True,
            error_type=None,
            message="Recording accepted.",
            peak=peak,
            rms=rms,
            duration_s=duration_s,
        )
