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
use grace_core::agent_loop::{AgentLoop, AgentLoopConfig};
use grace_core::dispatcher::{Dispatcher, NoopSystemActions};
use grace_core::events::RecordingEventSink;
use grace_core::feedback::RecordingEarconPlayer;
use grace_core::grace_app::{
    handle_activation_turn, handle_followup_transcript, ListenOutcome, RecordingWakeWordControl,
    TurnContext,
};
use grace_core::intent::IntentParser;
use grace_core::models::{RateLimitError, ScriptedLlm, ScriptedTts};
use grace_core::perception::{ElementGraph, ElementNode, ScreenSnapshot, ScriptedPerception};
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

/// Rehydrates a `ScreenSnapshot` from its taped JSON form in `snapshots.jsonl`.
pub fn rebuild_snapshot(entry: &Value) -> ScreenSnapshot {
    let snap = entry.get("snapshot").unwrap_or(entry);
    let win = snap.get("window");
    let window_title = win
        .and_then(|w| w.get("title"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let class_name = win
        .and_then(|w| w.get("class_name"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let is_browser = class_name == "Chrome_WidgetWin_1"
        || class_name == "MozillaWindowClass"
        || window_title.to_lowercase().contains("edge")
        || window_title.to_lowercase().contains("chrome")
        || window_title.to_lowercase().contains("firefox");

    let width = snap.get("width").and_then(Value::as_u64).unwrap_or(1920) as u32;
    let height = snap.get("height").and_then(Value::as_u64).unwrap_or(1080) as u32;
    let image_width = snap.get("image_width").and_then(Value::as_u64).unwrap_or(width as u64) as u32;
    let image_height = snap.get("image_height").and_then(Value::as_u64).unwrap_or(height as u64) as u32;

    let sources: Vec<String> = snap
        .get("graph_sources")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).map(String::from).collect())
        .unwrap_or_default();

    let elements: Vec<ElementNode> = snap
        .get("elements")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .map(|e| {
                    let rect = e
                        .get("rect")
                        .and_then(Value::as_array)
                        .and_then(|r| {
                            if r.len() >= 4 {
                                Some((
                                    r[0].as_i64().unwrap_or(0),
                                    r[1].as_i64().unwrap_or(0),
                                    r[2].as_i64().unwrap_or(0),
                                    r[3].as_i64().unwrap_or(0),
                                ))
                            } else {
                                None
                            }
                        })
                        .unwrap_or((0, 0, 0, 0));
                    let center = e
                        .get("center")
                        .and_then(Value::as_array)
                        .and_then(|c| {
                            if c.len() >= 2 {
                                Some((c[0].as_i64().unwrap_or(0), c[1].as_i64().unwrap_or(0)))
                            } else {
                                None
                            }
                        })
                        .unwrap_or((0, 0));

                    ElementNode {
                        id: e.get("id").and_then(Value::as_i64).unwrap_or(0),
                        role: e.get("role").and_then(Value::as_str).unwrap_or("").to_string(),
                        name: e.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
                        rect,
                        center,
                        value: e.get("value").and_then(Value::as_str).unwrap_or("").to_string(),
                        placeholder: e.get("placeholder").and_then(Value::as_str).unwrap_or("").to_string(),
                        frame: e.get("frame").and_then(Value::as_str).unwrap_or("app").to_string(),
                        container: e.get("container").and_then(Value::as_str).unwrap_or("").to_string(),
                        focused: e.get("focused").and_then(Value::as_bool).unwrap_or(false),
                        focusable: e.get("focusable").and_then(Value::as_bool).unwrap_or(false),
                        enabled: e.get("enabled").and_then(Value::as_bool).unwrap_or(true),
                        offscreen: e.get("offscreen").and_then(Value::as_bool).unwrap_or(false),
                        automation_id: e.get("automation_id").and_then(Value::as_str).unwrap_or("").to_string(),
                        source: e.get("source").and_then(Value::as_str).unwrap_or("uia").to_string(),
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let graph = ElementGraph {
        elements,
        window_title: window_title.clone(),
        is_browser,
        sources,
    };

    ScreenSnapshot {
        graph,
        window_title,
        width,
        height,
        png_bytes: None,
        image_width,
        image_height,
    }
}

/// Replays a tape through the Rust turn engine with every external edge pinned:
/// recorded snapshots through `ScriptedPerception`, recorded LLM outputs through `ScriptedLlm`,
/// recorded dispatches through `Dispatcher::with_tape_dispatches`.
pub fn replay_tape(tape: &Tape) -> TurnReplayResult {
    let session_id = tape
        .meta
        .get("session_id")
        .and_then(Value::as_str)
        .unwrap_or(&tape.dir)
        .to_string();

    let transcript = tape.stt.first().and_then(|e| e.get("text")).and_then(Value::as_str).unwrap_or("");

    let llm_responses: Vec<Result<Option<String>, RateLimitError>> = tape
        .llm
        .iter()
        .map(|entry| {
            if let Some(err) = entry.get("error").and_then(Value::as_str) {
                if err.starts_with("RateLimitError") {
                    return Err(RateLimitError(err.to_string()));
                }
            }
            Ok(llm_response_text(entry))
        })
        .collect();

    let snapshots: Vec<ScreenSnapshot> = tape.snapshots.iter().map(rebuild_snapshot).collect();

    let mut loop_config = AgentLoopConfig::default();
    if let Some(cfg) = tape.meta.get("config") {
        if let Some(n) = cfg.get("agent_max_iterations").and_then(Value::as_u64) {
            loop_config.max_iterations = n as u32;
        }
        if let Some(n) = cfg.get("agent_max_consecutive_plan_failures").and_then(Value::as_u64) {
            loop_config.max_consecutive_failures = n as u32;
        }
        if let Some(n) = cfg.get("agent_max_repeated_actions").and_then(Value::as_u64) {
            loop_config.max_repeated_actions = n as u32;
        }
        if let Some(n) = cfg.get("agent_max_seconds").and_then(Value::as_f64) {
            loop_config.max_seconds = n;
        }
        if let Some(n) = cfg.get("observability_min_actionable").and_then(Value::as_u64) {
            loop_config.min_actionable = n as usize;
        }
        if let Some(n) = cfg.get("planner_max_calls_per_goal").and_then(Value::as_u64) {
            loop_config.planner_max_calls_per_goal = n as u32;
        }
        if let Some(s) = cfg.get("stronger_planner_model").and_then(Value::as_str) {
            loop_config.stronger_model = Some(s.to_string());
        }
    }

    let mut planner_llm = ScriptedLlm::new(llm_responses);
    let mut vision_llm = ScriptedLlm::new(vec![]);
    let mut actions = NoopSystemActions;
    let mut dispatcher = Dispatcher::new(None, &mut actions).with_tape_dispatches(tape.dispatch.clone());
    let mut perception = ScriptedPerception::new(snapshots);
    let mut agent_loop = AgentLoop::new(loop_config);
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
        let ok = handle_activation_turn(
            &mut ctx,
            ListenOutcome::Transcript(transcript.to_string()),
            0.0,
            &|_| {},
            "intent system prompt (ignored by ScriptedLlm)",
            &mut last_spoken_text,
        );
        if ok && tape.stt.len() > 1 {
            for followup_stt in &tape.stt[1..] {
                let followup_text = followup_stt.get("text").and_then(Value::as_str).unwrap_or("");
                handle_followup_transcript(&mut ctx, followup_text, "intent system prompt (ignored by ScriptedLlm)", &mut last_spoken_text);
            }
        }
    }

    let actual: Vec<Value> = sink.events.iter().map(|e| serde_json::to_value(e).unwrap()).collect();

    let expected: Vec<Value> = if tape.stt.len() <= 1 {
        tape.event_payloads()
            .into_iter()
            .take_while(|e| e.get("type").and_then(Value::as_str) != Some("TurnTrace"))
            .collect()
    } else {
        tape.event_payloads()
            .into_iter()
            .filter(|e| {
                let t = e.get("type").and_then(Value::as_str).unwrap_or("");
                t != "TurnTrace" && t != "FollowupListeningStarted" && t != "Idle"
            })
            .collect()
    };

    let diff = crate::diff::diff_event_streams(&expected, &actual);
    TurnReplayResult { session_id, diff }
}

/// Compatibility alias for fast-path tapes.
pub fn replay_fastpath_tape(tape: &Tape) -> TurnReplayResult {
    replay_tape(tape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn corpus_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus")
    }

    #[test]
    fn replays_the_open_calculator_tape_byte_exact() {
        let tape = Tape::load(&corpus_dir().join("fastpath_open_calculator")).unwrap();
        let result = replay_tape(&tape);
        assert!(result.ok(), "{}", result.diff.report());
    }

    #[test]
    fn replays_agentic_two_step_tape() {
        let tape = Tape::load(&corpus_dir().join("agentic_two_step")).unwrap();
        let result = replay_tape(&tape);
        assert!(result.ok(), "{}", result.diff.report());
    }

    #[test]
    fn replays_agentic_step_failed_tape() {
        let tape = Tape::load(&corpus_dir().join("agentic_step_failed")).unwrap();
        let result = replay_tape(&tape);
        assert!(result.ok(), "{}", result.diff.report());
    }

    #[test]
    fn replays_fastpath_open_app() {
        let tape = Tape::load(&corpus_dir().join("fastpath_open_app")).unwrap();
        let result = replay_tape(&tape);
        assert!(result.ok(), "{}", result.diff.report());
    }

    #[test]
    fn replays_fastpath_adjust_volume() {
        let tape = Tape::load(&corpus_dir().join("fastpath_adjust_volume")).unwrap();
        let result = replay_tape(&tape);
        assert!(result.ok(), "{}", result.diff.report());
    }

    #[test]
    fn replays_every_single_corpus_tape_cleanly() {
        let tapes = Tape::load_corpus(&corpus_dir()).unwrap();
        assert_eq!(tapes.len(), 30, "expected all 30 corpus tapes to be loaded");
        let mut failures = Vec::new();
        for tape in &tapes {
            let res = replay_tape(tape);
            if !res.ok() {
                failures.push(format!("{}: {}", res.session_id, res.diff.report()));
            }
        }
        assert!(failures.is_empty(), "Replay failures on {} tape(s):\n\n{}", failures.len(), failures.join("\n\n"));
    }
}
