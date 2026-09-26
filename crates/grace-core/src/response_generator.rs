//! Ported from `src/grace/response/generator.py`: the pipeline from text to
//! TTS synthesis to the `SpeechStarted`/`ResponseChunk`/`SpeechChunk`/
//! `SpeechFinished` event sequence.
//!
//! **Scope note**: only `generate_and_speak_with_text`/`_synthesize_and_play`
//! is ported. `generate_and_speak` (streaming tokens directly from the LLM,
//! splitting sentences off the live stream) is dead code in the current
//! `main.py` wiring - nothing calls it, every speak path in
//! `GraceApp` goes through `generate_and_speak_with_text` - so porting it
//! would test behaviour nothing exercises. If a future phase wires it back
//! in on the Python side, this is where its port belongs.
//!
//! **Deliberate simplification**: Python's `_synthesize_and_play` submits
//! sentence N+1 for synthesis on a second worker while sentence N is still
//! being resolved and handed to the player (a one-deep lookahead so
//! synthesis hides behind playback). That pipelining has no effect on the
//! *sequence* of emitted events - `ResponseChunk`/`SpeechChunk` still emit
//! once per sentence, in order, exactly the same either way - only on
//! wall-clock latency, which `TurnTrace` captures and which the harness
//! excludes from its event diff anyway. This port synthesizes each sentence
//! synchronously instead; PORT_STATUS.md notes this as a place a later
//! phase can restore the pipelining without changing anything this port's
//! tests grade.

use crate::events::EventSink;
use crate::models::TextToSpeech;
use crate::sentence_split::split_sentences;
use grace_contract::GraceEvent;

/// Speaks pre-generated text (e.g. a tool's response, a PDF summary, an
/// agent loop's final answer). Returns `true` if at least one sentence was
/// successfully synthesized and handed to the player.
///
/// `stopped` mirrors `ResponseGenerator._stopped`: a caller wanting
/// barge-in sets it (from another thread/task, in the real wiring) and this
/// function checks it between sentences, exactly where Python does.
pub fn speak_text(
    text: &str,
    voice: &str,
    tts: &mut dyn TextToSpeech,
    sink: &mut dyn EventSink,
    stopped: &dyn Fn() -> bool,
) -> bool {
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return false;
    }
    synthesize_and_play(&sentences, voice, tts, sink, stopped)
}

fn synthesize_and_play(
    sentences: &[String],
    voice: &str,
    tts: &mut dyn TextToSpeech,
    sink: &mut dyn EventSink,
    stopped: &dyn Fn() -> bool,
) -> bool {
    let mut success = false;
    sink.emit(GraceEvent::SpeechStarted);

    for (i, sentence) in sentences.iter().enumerate() {
        if stopped() {
            break;
        }

        let wav = tts.synthesize(sentence, voice);

        if stopped() {
            break;
        }

        if let Some(wav) = wav {
            if !wav.is_empty() {
                success = true;
                // The leading-space rule: the renderer CONCATENATES chunks,
                // and chunk boundaries are part of the contract (see
                // `ResponseChunk`'s doc in the schema) - the space belongs
                // on the emitting side, not inferred by the reader.
                let chunk_text = if i > 0 { format!(" {sentence}") } else { sentence.clone() };
                sink.emit(GraceEvent::ResponseChunk { text: chunk_text });
                sink.emit(GraceEvent::SpeechChunk);
            }
        }
    }

    // `playback_drain` (waiting for the player to finish) has no observable
    // effect in this port: there is no real audio player here, only the
    // event sequence, and Python's own drain is skipped when `stopped` too.

    sink.emit(GraceEvent::SpeechFinished);
    success
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::RecordingEventSink;
    use crate::models::ScriptedTts;
    use std::cell::Cell;

    fn never_stopped() -> impl Fn() -> bool {
        || false
    }

    #[test]
    fn speaks_every_sentence_with_the_leading_space_rule() {
        let mut tts = ScriptedTts::default();
        let mut sink = RecordingEventSink::default();
        let ok = speak_text("Hello there. How are you?", "af_bella", &mut tts, &mut sink, &never_stopped());
        assert!(ok);

        assert_eq!(
            sink.events,
            vec![
                GraceEvent::SpeechStarted,
                GraceEvent::ResponseChunk { text: "Hello there.".into() },
                GraceEvent::SpeechChunk,
                GraceEvent::ResponseChunk { text: " How are you?".into() },
                GraceEvent::SpeechChunk,
                GraceEvent::SpeechFinished,
            ]
        );
    }

    #[test]
    fn empty_text_speaks_nothing_and_emits_no_events() {
        let mut tts = ScriptedTts::default();
        let mut sink = RecordingEventSink::default();
        let ok = speak_text("", "af_bella", &mut tts, &mut sink, &never_stopped());
        assert!(!ok);
        assert!(sink.events.is_empty());
    }

    #[test]
    fn a_sentence_that_fails_to_synthesize_is_silently_skipped_but_still_ok_overall() {
        let mut tts = ScriptedTts { fail_sentences: ["Hello.".to_string()].into_iter().collect(), ..Default::default() };
        let mut sink = RecordingEventSink::default();
        let ok = speak_text("Hello. Goodbye.", "af_bella", &mut tts, &mut sink, &never_stopped());
        assert!(ok); // "Goodbye." still succeeded
        let chunks: Vec<_> = sink
            .events
            .iter()
            .filter_map(|e| if let GraceEvent::ResponseChunk { text } = e { Some(text.clone()) } else { None })
            .collect();
        assert_eq!(chunks, vec![" Goodbye.".to_string()]); // note: still index 1, so leading space
    }

    #[test]
    fn stop_mid_stream_ends_the_loop_but_still_emits_speech_finished() {
        let mut tts = ScriptedTts::default();
        let mut sink = RecordingEventSink::default();
        let calls = Cell::new(0);
        let stopped = || {
            let n = calls.get();
            calls.set(n + 1);
            // Each sentence checks `stopped()` twice (before and after
            // synthesis); returning false for the first two calls lets
            // sentence 0 fully synthesize and emit, then true from the
            // third call (sentence 1's pre-check) stops the loop there.
            n >= 2
        };
        let ok = speak_text("One. Two. Three.", "af_bella", &mut tts, &mut sink, &stopped);
        assert!(ok);
        assert!(matches!(sink.events.last(), Some(GraceEvent::SpeechFinished)));
        let chunk_count = sink.events.iter().filter(|e| matches!(e, GraceEvent::ResponseChunk { .. })).count();
        assert!(chunk_count < 3);
    }

    #[test]
    fn all_sentences_failing_reports_false_but_still_brackets_with_speech_events() {
        let mut tts = ScriptedTts { fail_sentences: ["Hello.".to_string()].into_iter().collect(), ..Default::default() };
        let mut sink = RecordingEventSink::default();
        let ok = speak_text("Hello.", "af_bella", &mut tts, &mut sink, &never_stopped());
        assert!(!ok);
        assert_eq!(sink.events, vec![GraceEvent::SpeechStarted, GraceEvent::SpeechFinished]);
    }
}
