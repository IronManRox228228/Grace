//! Audio-side ports: VAD is fully ported (pure logic, see `vad.rs`). Capture,
//! the mic pump and wake-word spotting are behind traits only - see
//! PORT_STATUS.md. No test in this crate touches a real microphone or
//! speaker; anything that would is `#[ignore]`d with a reason, per the task's
//! hard constraints.

pub mod vad;

/// One chunk of 16-bit PCM audio, as handed to a `VadDetector` or a pump.
pub type PcmChunk = Vec<u8>;

/// What `src/grace/audio/capture.py`'s `AudioCapture` exposes to the rest of
/// the backend: a source of PCM chunks, abstracted so tests can supply a
/// scripted sequence instead of opening a real device (mirrors the tape
/// harness's `TapePump` in `src/grace/harness/replay.py`).
pub trait AudioSource: Send {
    /// Blocks (or awaits, in an async impl) until the next chunk is
    /// available, or returns `None` at end of stream.
    fn next_chunk(&mut self) -> Option<PcmChunk>;
}

/// A scripted `AudioSource` for tests and the harness: replays a fixed
/// sequence of chunks, then ends the stream. This is the only `AudioSource`
/// this crate provides; a real device-backed implementation (cpal, WASAPI,
/// etc.) is out of scope for this phase of the port - see PORT_STATUS.md.
pub struct ScriptedAudioSource {
    chunks: std::collections::VecDeque<PcmChunk>,
}

impl ScriptedAudioSource {
    pub fn new(chunks: Vec<PcmChunk>) -> Self {
        Self {
            chunks: chunks.into(),
        }
    }
}

impl AudioSource for ScriptedAudioSource {
    fn next_chunk(&mut self) -> Option<PcmChunk> {
        self.chunks.pop_front()
    }
}

/// What `src/grace/audio/wake_word.py` exposes: a keyword spotter that
/// consumes chunks and reports a detection. The real sherpa-onnx-backed
/// spotter is not implemented in this phase (native binding - see
/// PORT_STATUS.md); this trait is what a live implementation and a test
/// fake both implement.
pub trait WakeWordDetector: Send {
    /// Feed one chunk; returns `true` exactly when the wake word was just
    /// matched (edge-triggered, matching Vosk's partial-result handling in
    /// the Python spotter).
    fn process_chunk(&mut self, chunk: &[u8]) -> bool;
    /// Pauses detection (ship-blocker: the wake word must resume on every
    /// exit path from a turn - see `grace-core`'s notes and
    /// `tests/test_ship_blockers.py::TestWakeWordResumesAfterActivation`).
    fn pause(&mut self);
    fn resume(&mut self);
}

/// Whole-word wake phrase matcher (mirrors R13 in src/grace/audio/wake_word.py).
///
/// Substrings like "gracefully" or "disgrace" must not match. Only whole-word
/// occurrences of `keyword` (or "grace" / "hey grace") trigger detection.
pub fn is_wake_word_match(text: &str, keyword: &str) -> bool {
    let lower = text.to_lowercase();
    let kw = keyword.to_lowercase();
    let target = if kw.is_empty() { "grace" } else { &kw };

    for phrase in [target, "grace", "hey grace"] {
        let words: Vec<&str> = phrase.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        let text_words: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .collect();

        if text_words.windows(words.len()).any(|window| window == words.as_slice()) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripted_audio_source_replays_then_ends() {
        let mut src = ScriptedAudioSource::new(vec![vec![1, 2], vec![3, 4]]);
        assert_eq!(src.next_chunk(), Some(vec![1, 2]));
        assert_eq!(src.next_chunk(), Some(vec![3, 4]));
        assert_eq!(src.next_chunk(), None);
    }

    #[test]
    fn whole_word_match_rejects_substrings() {
        assert!(!is_wake_word_match("gracefully", "grace"));
        assert!(!is_wake_word_match("disgrace", "grace"));
        assert!(!is_wake_word_match("disgraceful situation", "grace"));
    }

    #[test]
    fn whole_word_match_accepts_isolated_keyword() {
        assert!(is_wake_word_match("grace", "grace"));
        assert!(is_wake_word_match("hey grace", "grace"));
        assert!(is_wake_word_match("grace, can you hear me?", "grace"));
        assert!(is_wake_word_match("saving grace", "grace"));
    }
}

