//! Ported from `src/grace/vad/detector.py` (`VadDetector`).
//!
//! **Live constraint, kept deliberately** (see `contract/README.md`, "A
//! constraint the harness surfaced"): by default this accumulates silence
//! against the wall clock, not a sample count. A caller that wants the
//! harness's deterministic replay behaviour passes a `sample_rate` (plus
//! width/channels) to switch the detector onto the *audio* clock instead,
//! exactly like the Python constructor's `sample_rate: Optional[int]`
//! parameter. Changing which clock the *live* path uses is a deliberate
//! behaviour change and must be called out, not something to silently do
//! while porting.

use std::time::Instant;

/// The wall clock vs. audio clock, abstracted so tests never depend on real
/// elapsed time. `WallClock` uses `Instant::now()`; `AudioClock` advances
/// only when `process_chunk` is called, exactly the amount of audio handed
/// to it.
trait Clock {
    fn now_seconds(&mut self, chunk_len_bytes: usize) -> f64;
    fn is_audio_clock(&self) -> bool;
}

struct WallClock {
    start: Instant,
}
impl Clock for WallClock {
    fn now_seconds(&mut self, _chunk_len_bytes: usize) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
    fn is_audio_clock(&self) -> bool {
        false
    }
}

struct AudioClock {
    bytes_per_second: f64,
    accumulated_seconds: f64,
}
impl Clock for AudioClock {
    fn now_seconds(&mut self, chunk_len_bytes: usize) -> f64 {
        self.accumulated_seconds += chunk_len_bytes as f64 / self.bytes_per_second;
        self.accumulated_seconds
    }
    fn is_audio_clock(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub struct SilenceState {
    pub is_silent: bool,
    pub silence_start: f64,
    pub last_speech_time: f64,
    pub total_silence_ms: f64,
}

impl Default for SilenceState {
    /// Matches the Python dataclass default: `is_silent=True` - before any
    /// chunk has been processed, there is nothing to call "speaking" yet.
    fn default() -> Self {
        Self {
            is_silent: true,
            silence_start: 0.0,
            last_speech_time: 0.0,
            total_silence_ms: 0.0,
        }
    }
}

/// Voice Activity Detection using energy thresholding. See the module doc
/// for the wall-clock-vs-audio-clock distinction this is built around.
pub struct VadDetector {
    threshold: f64,
    silence_duration_ms: f64,
    clock: Box<dyn Clock + Send>,
    state: SilenceState,
    has_detected_speech: bool,
}

impl VadDetector {
    /// Wall-clock detector: what the live microphone path uses.
    pub fn new_wall_clock(threshold: f64, silence_duration_ms: u64) -> Self {
        Self {
            threshold,
            silence_duration_ms: silence_duration_ms as f64,
            clock: Box::new(WallClock {
                start: Instant::now(),
            }),
            state: SilenceState::default(),
            has_detected_speech: false,
        }
    }

    /// Audio-clock detector: what the harness's replay uses, so a replay's
    /// turn-end timing does not depend on how fast recorded chunks are
    /// scheduled to arrive.
    pub fn new_audio_clock(
        threshold: f64,
        silence_duration_ms: u64,
        sample_rate: u32,
        sample_width: u32,
        channels: u32,
    ) -> Self {
        Self {
            threshold,
            silence_duration_ms: silence_duration_ms as f64,
            clock: Box::new(AudioClock {
                bytes_per_second: (sample_rate * sample_width * channels) as f64,
                accumulated_seconds: 0.0,
            }),
            state: SilenceState::default(),
            has_detected_speech: false,
        }
    }

    pub fn uses_audio_clock(&self) -> bool {
        self.clock.is_audio_clock()
    }

    pub fn is_speaking(&self) -> bool {
        !self.state.is_silent
    }

    pub fn has_detected_speech(&self) -> bool {
        self.has_detected_speech
    }

    fn normalised_rms(chunk: &[u8]) -> f64 {
        if chunk.len() < 2 || chunk.len() % 2 != 0 {
            return 0.0;
        }
        let samples: Vec<i16> = chunk
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
        let rms = (sum_sq / samples.len() as f64).sqrt();
        rms / 32767.0
    }

    /// Process a single audio chunk of 16-bit little-endian PCM. Returns
    /// `true` if a silence turn-end was detected.
    pub fn process_chunk(&mut self, chunk: &[u8]) -> bool {
        let normalized_rms = Self::normalised_rms(chunk);

        // The clock advances by the chunk's own duration *before* the chunk
        // is judged (audio clock only; the wall clock has already moved on
        // its own) - ported from the Python comment on this exact ordering.
        let now = self.clock.now_seconds(chunk.len());

        if normalized_rms >= self.threshold {
            self.has_detected_speech = true;
            if self.state.is_silent {
                self.state.is_silent = false;
                self.state.last_speech_time = now;
            }
        } else if !self.state.is_silent {
            self.state.is_silent = true;
            self.state.silence_start = now;
            self.state.total_silence_ms = 0.0;
        }

        if self.state.is_silent && self.has_detected_speech {
            self.state.total_silence_ms = (now - self.state.silence_start) * 1000.0;
            if self.state.total_silence_ms >= self.silence_duration_ms {
                return true;
            }
        }
        false
    }

    /// Reset detection state for a new turn. The audio clock keeps running
    /// across turns - it measures elapsed audio, not elapsed turn, and
    /// rewinding it would make the second turn of a session start from a
    /// silence window the first turn had already filled.
    pub fn reset(&mut self) {
        let carried_silence_start = if self.clock.is_audio_clock() {
            // Re-derive "now" without consuming any bytes, to keep the audio
            // clock's accumulated position.
            self.clock.now_seconds(0)
        } else {
            0.0
        };
        self.state = SilenceState {
            silence_start: carried_silence_start,
            ..SilenceState::default()
        };
        self.has_detected_speech = false;
    }
}

/// Encodes a sequence of `i16` samples as little-endian PCM bytes, for
/// building test fixtures without hand-writing byte arrays.
pub fn pcm16_from_samples(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence_chunk(n: usize) -> Vec<u8> {
        pcm16_from_samples(&vec![0i16; n])
    }

    fn speech_chunk(n: usize) -> Vec<u8> {
        pcm16_from_samples(&vec![20000i16; n])
    }

    #[test]
    fn audio_clock_detector_never_calls_the_wall_clock() {
        // 16kHz mono 16-bit: 1600 bytes/chunk = 0.05s of audio per chunk.
        let mut vad = VadDetector::new_audio_clock(0.5, 100, 16000, 2, 1);
        assert!(vad.uses_audio_clock());

        assert!(!vad.process_chunk(&speech_chunk(800)));
        assert!(vad.is_speaking());

        // The chunk that transitions speech->silence starts the silence
        // window at its own timestamp (total_silence_ms=0 immediately after
        // it), so it takes two more 0.05s chunks after that one to reach the
        // 100ms threshold.
        assert!(!vad.process_chunk(&silence_chunk(800))); // total=0ms
        assert!(!vad.process_chunk(&silence_chunk(800))); // total=50ms
        assert!(vad.process_chunk(&silence_chunk(800))); // total=100ms -> turn end
    }

    #[test]
    fn no_turn_end_before_any_speech_was_detected() {
        let mut vad = VadDetector::new_audio_clock(0.5, 50, 16000, 2, 1);
        for _ in 0..10 {
            assert!(!vad.process_chunk(&silence_chunk(800)));
        }
        assert!(!vad.has_detected_speech());
    }

    #[test]
    fn reset_clears_turn_state_but_keeps_the_audio_clock_running() {
        // A deliberately low threshold (50ms against 100ms-per-chunk audio)
        // so the post-transition chunk clears it with headroom, rather than
        // landing on an exact millisecond boundary that floating-point
        // summation of repeated 0.1s increments cannot be relied on to hit
        // (100 chunks of 1/32000s each do not sum to exactly 0.1 in f64).
        let mut vad = VadDetector::new_audio_clock(0.5, 50, 16000, 2, 1);
        vad.process_chunk(&speech_chunk(1600)); // advances the clock 0.1s
        vad.reset();
        assert!(!vad.is_speaking());
        assert!(!vad.has_detected_speech());

        // Silence alone can never trip turn-end right after a reset: with
        // `has_detected_speech` cleared, `total_silence_ms` is never even
        // computed (ported faithfully from the Python `reset()`, whose
        // comment says only that the *clock* - not turn-end reachability -
        // survives a reset).
        assert!(!vad.process_chunk(&silence_chunk(1600)));
        assert!(!vad.process_chunk(&silence_chunk(1600)));

        // A fresh speech->silence cycle after reset behaves like a normal
        // one: the clock kept advancing underneath (it did not rewind to
        // 0s), but turn-end timing is measured from this new silence
        // window's own start, not from any position carried across reset.
        vad.process_chunk(&speech_chunk(1600));
        assert!(!vad.process_chunk(&silence_chunk(1600))); // total=0ms
        assert!(vad.process_chunk(&silence_chunk(1600))); // total=~100ms >= 50ms threshold
    }

    #[test]
    fn is_speaking_is_false_before_any_chunk_is_processed() {
        let vad = VadDetector::new_audio_clock(0.5, 200, 16000, 2, 1);
        assert!(!vad.is_speaking());
        assert!(!vad.has_detected_speech());
    }

    #[test]
    fn wall_clock_detector_reports_itself_correctly() {
        let vad = VadDetector::new_wall_clock(0.5, 1200);
        assert!(!vad.uses_audio_clock());
    }
}
