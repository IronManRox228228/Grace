use std::time::{Duration, Instant};
use grace_core::grace_app::ListenOutcome;
use grace_core::models::SpeechToText;

use crate::capture::AudioPump;
use crate::vad::VadDetector;

/// Source of transcripts for an activation turn.
pub trait TranscriptSource: Send {
    /// Listen for an utterance and return the resulting `ListenOutcome` and elapsed listen duration.
    fn capture_and_transcribe(&mut self) -> (ListenOutcome, f64);
}

/// Scripted transcript source for deterministic tests and tape replays.
pub struct ScriptedTranscriptSource {
    outcomes: std::collections::VecDeque<ListenOutcome>,
}

impl ScriptedTranscriptSource {
    pub fn new(outcomes: Vec<ListenOutcome>) -> Self {
        Self {
            outcomes: outcomes.into(),
        }
    }

    pub fn from_transcripts(transcripts: Vec<&str>) -> Self {
        Self {
            outcomes: transcripts
                .into_iter()
                .map(|t| ListenOutcome::Transcript(t.to_string()))
                .collect(),
        }
    }
}

impl TranscriptSource for ScriptedTranscriptSource {
    fn capture_and_transcribe(&mut self) -> (ListenOutcome, f64) {
        let outcome = self
            .outcomes
            .pop_front()
            .unwrap_or(ListenOutcome::Silence);
        (outcome, 0.0)
    }
}

/// Real hardware audio capture and transcription source.
///
/// Drives the microphone pump, runs VAD to detect speech start and turn-end boundary,
/// and transcribes the speech buffer using the configured ASR engine.
pub struct RealTranscriptSource<'a> {
    pump: &'a AudioPump,
    vad: &'a mut VadDetector,
    asr: &'a mut dyn SpeechToText,
    initial_timeout: Duration,
    max_timeout: Duration,
}

impl<'a> RealTranscriptSource<'a> {
    pub fn new(
        pump: &'a AudioPump,
        vad: &'a mut VadDetector,
        asr: &'a mut dyn SpeechToText,
        initial_timeout: Duration,
        max_timeout: Duration,
    ) -> Self {
        Self {
            pump,
            vad,
            asr,
            initial_timeout,
            max_timeout,
        }
    }
}

impl<'a> TranscriptSource for RealTranscriptSource<'a> {
    fn capture_and_transcribe(&mut self) -> (ListenOutcome, f64) {
        self.pump.drain();
        self.vad.reset();

        let start = Instant::now();
        let mut audio_buffer = Vec::new();
        let mut speech_started = false;

        while start.elapsed() < self.max_timeout {
            if let Some(chunk) = self.pump.get(Duration::from_millis(50)) {
                audio_buffer.extend_from_slice(&chunk);

                let is_turn_end = self.vad.process_chunk(&chunk);
                if self.vad.has_detected_speech() {
                    speech_started = true;
                }

                if is_turn_end {
                    break;
                }
            } else if !speech_started && start.elapsed() > self.initial_timeout {
                // Initial silence timeout reached
                break;
            }
        }

        let elapsed = start.elapsed().as_secs_f64();

        if !speech_started || audio_buffer.is_empty() {
            return (ListenOutcome::Silence, elapsed);
        }

        let text = self.asr.transcribe(&audio_buffer);
        if text.trim().is_empty() {
            (ListenOutcome::Silence, elapsed)
        } else {
            (ListenOutcome::Transcript(text), elapsed)
        }
    }
}

/// An owning real transcript source that bundles pump, vad, and asr.
pub struct OwnedRealTranscriptSource {
    pump: AudioPump,
    vad: VadDetector,
    asr: Box<dyn SpeechToText>,
    initial_timeout: Duration,
    max_timeout: Duration,
}

impl OwnedRealTranscriptSource {
    pub fn new(
        pump: AudioPump,
        vad: VadDetector,
        asr: Box<dyn SpeechToText>,
        initial_timeout: Duration,
        max_timeout: Duration,
    ) -> Self {
        Self {
            pump,
            vad,
            asr,
            initial_timeout,
            max_timeout,
        }
    }
}

impl TranscriptSource for OwnedRealTranscriptSource {
    fn capture_and_transcribe(&mut self) -> (ListenOutcome, f64) {
        let mut source = RealTranscriptSource::new(
            &self.pump,
            &mut self.vad,
            self.asr.as_mut(),
            self.initial_timeout,
            self.max_timeout,
        );
        source.capture_and_transcribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripted_source_yields_queued_transcripts() {
        let mut src = ScriptedTranscriptSource::from_transcripts(vec![
            "open the calculator",
            "close it",
        ]);
        let (res1, _) = src.capture_and_transcribe();
        assert_eq!(
            res1,
            ListenOutcome::Transcript("open the calculator".to_string())
        );

        let (res2, _) = src.capture_and_transcribe();
        assert_eq!(res2, ListenOutcome::Transcript("close it".to_string()));

        let (res3, _) = src.capture_and_transcribe();
        assert_eq!(res3, ListenOutcome::Silence);
    }
}
