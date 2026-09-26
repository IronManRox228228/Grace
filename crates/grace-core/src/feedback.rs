//! Ported from `src/grace/response/feedback.py`: audio earcons (activation
//! chime, success/cancel/error/listening blips).
//!
//! These play real audio through the speakers in Python (`sounddevice`),
//! which the task's hard constraints forbid touching in a test. This module
//! is the trait boundary only: `EarconPlayer`, with a `RecordingEarconPlayer`
//! fake for tests that assert "which earcon played", and no real
//! implementation - a real one belongs with `grace-audio`'s eventual
//! sherpa-onnx/output-device wiring (not attempted this phase; see
//! PORT_STATUS.md).

/// Which earcon was requested. Mirrors `FeedbackSounds`'s five static
/// methods as one enum, since every caller in this port needs is "which
/// one", never the tone-synthesis parameters Python's version exposes
/// (`duration`, `volume`) - those are audio-rendering details with no
/// bearing on turn logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Earcon {
    /// Wake-word activation chime.
    Chime,
    /// Gentle rising two-tone confirming an action finished.
    Success,
    /// Gentle descending two-tone confirming cancellation/stop.
    Cancel,
    /// Low-frequency double tone for an error or no-speech timeout.
    Error,
    /// Subtle high blip for "listening" in the follow-up window.
    Listening,
}

pub trait EarconPlayer: Send {
    fn play(&mut self, earcon: Earcon);
}

/// Records every earcon played, for tests to assert against instead of
/// touching real speakers.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct RecordingEarconPlayer {
    pub played: Vec<Earcon>,
}

impl EarconPlayer for RecordingEarconPlayer {
    fn play(&mut self, earcon: Earcon) {
        self.played.push(earcon);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_player_keeps_every_earcon_in_order() {
        let mut player = RecordingEarconPlayer::default();
        player.play(Earcon::Chime);
        player.play(Earcon::Error);
        assert_eq!(player.played, vec![Earcon::Chime, Earcon::Error]);
    }
}
