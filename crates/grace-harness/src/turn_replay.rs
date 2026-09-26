//! Drives the real Rust turn engine (`grace_core::grace_app`) against a
//! tape, with every external edge pinned exactly the way
//! `src/grace/harness/replay.py` pins the Python backend's: the LLM's
//! recorded token stream (`llm.jsonl`), the recorded transcript
//! (`stt.jsonl`), and the recorded dispatch outcomes (implicitly, via a
//! `NoopSystemActions` that reproduces the same hardcoded success text the
//! dispatcher itself owns - see `grace_core::dispatcher::NoopSystemActions`'s
//! doc comment for why that's enough for most fast-path tools without
//! needing the tape's own `dispatch.jsonl` values).
//!
//! **What this actually replays**: the *event* stream, from `WakeWordDetected`
//! through `ConversationFinished` - not the raw microphone/VAD/whisper
//! portion of a turn (out of scope: real audio - see `grace_app`'s module
//! doc) and not the follow-up window (also not ported - same doc). The tape's
//! own event list is truncated to the same window before diffing, so a
//! passing replay is a real claim about a real slice of the turn, not a
//! silently-lenient comparison.
//!
//! **Why only fast-path tapes, for now**: an agentic-goal tape's events
//! depend on the agent loop actually planning multiple steps against a
//! *replayed* element graph/perception snapshot sequence, which needs a tape
//! reader for `snapshots.jsonl` this phase didn't build (see
//! PORT_STATUS.md). The fast-path tapes replayed here exercise the intent
//! parse -> route -> dispatch -> speak pipeline, which is the majority of
//! `CapabilityRouter::FAST_PATH_TOOLS` traffic and is exactly what
//! `grace_app::handle_activation_turn` fully implements today.

use crate::tape::Tape;
use grace_core::dispatcher::{Dispatcher, NoopSystemActions};
use grace_core::events::RecordingEventSink;
use grace_core::grace_app::{handle_activation_turn, ListenOutcome, RecordingWakeWordControl, TurnContext};
use grace_core::intent::IntentParser;
use grace_core::models::{ScriptedLlm, ScriptedTts};
use grace_core::perception::ScriptedPerception;
use grace_core::agent_loop::{AgentLoop, AgentLoopConfig};
use grace_core::feedback::RecordingEarconPlayer;
use serde_json::Value;

/// The result of replaying one tape's event stream.
pub struct TurnReplayResult {
    pub session_id: String,
    pub diff: crate::diff::Diff,
}

impl TurnReplayResult {
    pub fn ok(&self) -> bool {
        self.diff.ok()
    }
}

/// Concatenates a `llm.jsonl` entry's streamed `chunks` into the full raw
/// text the LLM produced - mirrors how `GemmaClient.generate_intent` joins
/// the token stream it collected (`"".join(tokens)`).
fn llm_response_text(entry: &Value) -> Option<String> {
    let chunks = entry.get("chunks")?.as_array()?;
    let joined: String = chunks.iter().filter_map(|c| c.as_str()).collect();
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// Replays one fast-path tape's activation turn and diffs the resulting
/// event stream (both sides truncated to `WakeWordDetected..=ConversationFinished`,
/// dropping `TurnTrace` and anything after it - see this module's doc
/// comment) against the tape's own recorded events.
pub fn replay_fastpath_tape(tape: &Tape) -> TurnReplayResult {
    let session_id = tape
        .meta
        .get("session_id")
        .and_then(Value::as_str)
        .unwrap_or(&tape.dir)
        .to_string();

    let transcript = tape.stt.first().and_then(|e| e.get("text")).and_then(Value::as_str).unwrap_or("");

    let llm_responses: Vec<Result<Option<String>, grace_core::models::RateLimitError>> =
        tape.llm.iter().map(|entry| Ok(llm_response_text(entry))).collect();

    let mut planner_llm = ScriptedLlm::new(llm_responses);
    let mut vision_llm = ScriptedLlm::new(vec![]);
    let mut actions = NoopSystemActions;
    let mut dispatcher = Dispatcher::new(None, &mut actions);
    let mut perception = ScriptedPerception::new(vec![]);
    let mut agent_loop = AgentLoop::new(AgentLoopConfig::default());
    let mut tts = ScriptedTts::default();
    let mut wake_word = RecordingWakeWordControl::default();
    let mut earcons = RecordingEarconPlayer::default();
    let mut sink = RecordingEventSink::default();
    let mut intent_parser = IntentParser::new();
    let mut last_spoken_text = String::new();

    {
        let mut ctx = TurnContext {
            wake_word: &mut wake_word,
            earcons: &mut earcons,
            sink: &mut sink,
            intent_parser: &mut intent_parser,
            planner_llm: &mut planner_llm,
            vision_llm: &mut vision_llm,
            dispatcher: &mut dispatcher,
            perception: &mut perception,
            agent_loop: &mut agent_loop,
            tts: &mut tts,
        };
        handle_activation_turn(&mut ctx, ListenOutcome::Transcript(transcript.to_string()), 0.0, &|_| {}, "intent system prompt (ignored by ScriptedLlm)", &mut last_spoken_text);
    }

    let actual: Vec<Value> = sink.events.iter().map(|e| serde_json::to_value(e).unwrap()).collect();

    let expected: Vec<Value> = tape
        .event_payloads()
        .into_iter()
        .take_while(|e| e.get("type").and_then(Value::as_str) != Some("TurnTrace"))
        .collect();

    let diff = crate::diff::diff_event_streams(&expected, &actual);
    TurnReplayResult { session_id, diff }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn corpus_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
    }

    /// Tapes this phase's turn engine can honestly be graded against: a
    /// single fast-path tool call with a plain, parameter-driven response
    /// (no follow-up window, no confirmation, no agent loop). Other fast-path
    /// tapes either need `SystemActions` to see the tape's own file/app data
    /// (`fastpath_search_files`, `fastpath_open_app`) or exercise the
    /// confirmation/follow-up paths this list intentionally keeps separate
    /// so a genuine mismatch there isn't hidden among passes.
    const REPLAYABLE_FASTPATH_TAPES: &[&str] = &["fastpath_open_calculator", "fastpath_lock_computer"];

    #[test]
    fn replays_the_open_calculator_tape_byte_exact() {
        let tape = Tape::load(&corpus_dir().join("fastpath_open_calculator")).unwrap();
        let result = replay_fastpath_tape(&tape);
        assert!(result.ok(), "{}", result.diff.report());
    }

    /// Runs every tape in `REPLAYABLE_FASTPATH_TAPES` and reports each one's
    /// outcome individually, so a regression in one doesn't hide a pass in
    /// another (and so PORT_STATUS.md's claim - "these specific tapes replay
    /// clean" - is exactly what this test enforces, not an approximation of
    /// it).
    #[test]
    fn replays_every_listed_fastpath_tape() {
        let mut failures = Vec::new();
        for name in REPLAYABLE_FASTPATH_TAPES {
            let tape = Tape::load(&corpus_dir().join(name)).unwrap();
            let result = replay_fastpath_tape(&tape);
            if !result.ok() {
                failures.push(format!("{}: {}", result.session_id, result.diff.report()));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    }

    /// Honest counterpart to the passes above: `fastpath_open_app` needs the
    /// tape's own resolved app path/alias behaviour
    /// (`grace_core::dispatcher`'s `NoopSystemActions` always succeeds with a
    /// generic "I've opened X" rather than the tape's actual recorded text
    /// for a website alias), so it is EXPECTED to diff, and this test pins
    /// that expectation rather than silently passing or silently failing.
    #[test]
    fn fastpath_open_app_is_a_known_non_replayable_case() {
        let tape = Tape::load(&corpus_dir().join("fastpath_open_app")).unwrap();
        let result = replay_fastpath_tape(&tape);
        // If this ever starts passing (e.g. because NoopSystemActions grew
        // smarter alias handling), that's good news - flip this assertion
        // and move the tape into REPLAYABLE_FASTPATH_TAPES rather than
        // leaving a silently-stale expectation here.
        assert!(!result.ok(), "expected this tape to still need real SystemActions data");
    }

    /// `adjust_volume`'s recorded result (`"Volume set to 60 percent."` for
    /// a `mode: "increase", amount: 10` request) reflects the REAL pycaw
    /// state of the recording machine's speakers at record time (50% ->
    /// 60%) - not something any fake can reproduce without also faking "what
    /// the volume happened to be right then". This is a hardware-state
    /// tape, not a route/prompt one; ported honestly as a documented
    /// non-replayable case rather than papered over with a fake that
    /// happens to return 60.
    #[test]
    fn fastpath_adjust_volume_is_a_known_non_replayable_case_due_to_real_hardware_state() {
        let tape = Tape::load(&corpus_dir().join("fastpath_adjust_volume")).unwrap();
        let result = replay_fastpath_tape(&tape);
        assert!(!result.ok(), "expected this tape to depend on real audio hardware state");
    }
}
