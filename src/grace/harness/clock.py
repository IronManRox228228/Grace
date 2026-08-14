"""Give a harnessed app a VAD that measures silence in audio, not wall time.

The corpus is the migration's oracle, and an oracle that fails 2 of 27 tapes at
random cannot tell a regression from a coin flip. That is what it did.

``VadDetector`` accumulates silence against ``time.time()``. Both harness pumps
deliver a turn's chunks as fast as the scheduler lets them - ``TapePump`` sleeps
to the recording's own offsets, but only when it is *ahead*, and a burst to
catch up after a slow LLM call has no pacing at all. A generated turn carries
about 300ms of trailing silence against a 275ms window, so a burst can put every
silent chunk inside 275ms of wall time; the chunks then run out before the VAD
closes the turn, and the replay reports "recorded audio ended before the VAD
closed the turn". A ~25ms margin against a real clock, decided by CPU load.

Audio time removes the race outright rather than widening the margin: the same
chunk sequence advances the clock by the same amount every run, so the turn ends
on the same chunk on a loaded machine as on an idle one. A generated turn's 20
silent chunks are 640ms of audio against the 275ms window - the turn closes on
the 9th, with 11 chunks to spare, and no scheduler can move that.

This is harness-only. The live listening path keeps its wall clock; see
``VadDetector`` for why that choice is deliberate rather than incidental.
"""

from __future__ import annotations

from ..vad.detector import VadDetector


def install_audio_clock_vad(app) -> VadDetector:
    """Replace ``app.vad`` with an audio-clock detector of the same settings.

    Thresholds come from the app's own config, so a tape still replays against
    the VAD configuration it was recorded with - only the clock changes. Nothing
    registers VAD callbacks (``main.py`` polls ``process_chunk``'s return value),
    so swapping the instance whole is safe.
    """
    config = app.config
    app.vad = VadDetector(
        threshold=config.whisper_vad_threshold,
        silence_duration_ms=config.whisper_silence_duration_ms,
        sample_rate=config.mic_sample_rate,
        sample_width=config.mic_width,
        channels=config.mic_channels,
    )
    return app.vad
