//! Rust types for `contract/grace-events.schema.json`.
//!
//! This is the ONE seam between the Grace backend (Python today, Rust after
//! the migration) and the React renderer. It is FROZEN: additive-only, and
//! only for genuinely new UI features. If the Rust backend cannot reproduce
//! an existing event at the right moment, that is a bug in the port, not a
//! reason to change the schema.
//!
//! `tests/schema_roundtrip.rs` loads the schema at
//! `../../contract/grace-events.schema.json` and validates a sample of every
//! variant against it, so this enum cannot silently drift from the frozen
//! contract the way the hand-written TypeScript union once did (see
//! `contract/README.md`).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One stage of a `TurnTrace`. Mirrors `util/timing.py TurnTrace.to_dict()`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnTraceStage {
    pub name: String,
    pub depth: u32,
    /// `null` when the stage never closed (e.g. the turn aborted mid-stage).
    pub ms: Option<f64>,
    pub detail: Option<String>,
}

/// The full per-turn latency trace. Diagnostic only - the renderer ignores
/// it, and it is excluded from the harness's event diff because it carries
/// wall-clock durations that legitimately differ on every run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnTraceData {
    pub label: String,
    pub total_ms: f64,
    pub stages: Vec<TurnTraceStage>,
    pub events: BTreeMap<String, f64>,
}

/// The complete backend -> frontend event stream.
///
/// Serializes with an adjacently-untagged `"type"` discriminant, exactly as
/// `contract/grace-events.schema.json`'s `oneOf` expects. Field order and
/// optionality is transcribed 1:1 from the schema's `required` /
/// `additionalProperties: false` per variant; see each variant's doc comment
/// for the Python emission site it corresponds to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GraceEvent {
    /// Turn is over; the overlay resets to its resting state. main.py:586,593,738.
    Idle,
    /// Vosk (later: the wake-word spotter) matched the wake word, or the
    /// UI/global-hotkey forced a wake. main.py:531.
    WakeWordDetected,
    /// Mic is open for the primary utterance of a turn. main.py:534.
    ListeningStarted,
    /// The post-response follow-up window has opened. main.py:697. `timeout`
    /// is the window length in whole seconds
    /// (`config.followup_timeout_seconds`, default 10).
    FollowupListeningStarted { timeout: f64 },
    /// The STT transcript for the utterance just captured. main.py:597,743.
    /// Grace has no partial/streaming transcript: STT runs once on the whole
    /// buffered utterance after VAD closes the turn.
    FinalTranscript { text: String },
    /// Mic closed for this utterance. main.py:598.
    ListeningStopped,
    /// Intent inference began. main.py:607. `label` is user-facing prose,
    /// e.g. "Understanding request…".
    UnderstandingStarted { label: String },
    /// Intent inference ended (success or failure). main.py:617,635.
    UnderstandingFinished,
    /// A tool is about to run. `label` is the only field the renderer
    /// consumes; `tool` and `step` are emitted by the agent loop
    /// (loop.py:211) for tracing and are absent on the fast-path
    /// dispatcher's emissions (dispatcher.py:182,209).
    ToolExecutionStarted {
        label: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<u32>,
    },
    /// The tool finished. `tool` and `status` are emitted by the agent loop
    /// (loop.py:237) and absent from the dispatcher's emission
    /// (dispatcher.py:186).
    ToolExecutionFinished {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    /// One sentence-shaped slice of the assistant's response, split off the
    /// LLM token stream as it arrives (response/generator.py:127,142,218).
    /// The renderer CONCATENATES these; the leading-space rule (`" "`
    /// prefix for i > 0) lives on the emitting side, so chunk boundaries are
    /// part of the contract, not an implementation detail.
    ResponseChunk { text: String },
    /// TTS playback began for this response. generator.py:95,190.
    SpeechStarted,
    /// One synthesized sentence was handed to the player. Paired 1:1 with
    /// `ResponseChunk`. generator.py:128,143,219.
    SpeechChunk,
    /// TTS playback drained. generator.py:150,228.
    SpeechFinished,
    /// The turn's work is done; the follow-up window may still be open.
    /// main.py:585,591,603,682.
    ConversationFinished,
    /// Per-turn latency instrumentation. main.py:520. Diagnostic only.
    TurnTrace { trace: TurnTraceData },
    /// A turn failed in a way the user must be told about. RESERVED: the
    /// Python backend never emits this (it logs and falls through to
    /// `ConversationFinished` + `Idle`). The renderer has always handled it,
    /// and the Rust port needs it for sidecar-unavailable and
    /// supervisor-failure paths. Emitting it anywhere the Python backend
    /// wouldn't is itself a behaviour change and must be justified the same
    /// way any other deliberate deviation is (see PORT_STATUS.md).
    Error { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_events() -> Vec<GraceEvent> {
        vec![
            GraceEvent::Idle,
            GraceEvent::WakeWordDetected,
            GraceEvent::ListeningStarted,
            GraceEvent::FollowupListeningStarted { timeout: 10.0 },
            GraceEvent::FinalTranscript {
                text: "open notepad".into(),
            },
            GraceEvent::ListeningStopped,
            GraceEvent::UnderstandingStarted {
                label: "Understanding request…".into(),
            },
            GraceEvent::UnderstandingFinished,
            GraceEvent::ToolExecutionStarted {
                label: "Opening Notepad".into(),
                tool: None,
                step: None,
            },
            GraceEvent::ToolExecutionStarted {
                label: "Opening Notepad".into(),
                tool: Some("open_app".into()),
                step: Some(1),
            },
            GraceEvent::ToolExecutionFinished {
                tool: None,
                status: None,
            },
            GraceEvent::ToolExecutionFinished {
                tool: Some("open_app".into()),
                status: Some("ok".into()),
            },
            GraceEvent::ResponseChunk {
                text: "Done.".into(),
            },
            GraceEvent::SpeechStarted,
            GraceEvent::SpeechChunk,
            GraceEvent::SpeechFinished,
            GraceEvent::ConversationFinished,
            GraceEvent::TurnTrace {
                trace: TurnTraceData {
                    label: "activation".into(),
                    total_ms: 812.3,
                    stages: vec![TurnTraceStage {
                        name: "stt".into(),
                        depth: 0,
                        ms: Some(300.1),
                        detail: None,
                    }],
                    events: BTreeMap::new(),
                },
            },
            GraceEvent::Error {
                message: "sidecar unavailable".into(),
            },
        ]
    }

    /// Every variant round-trips through JSON with the exact shape it had
    /// going in - no field gained or lost by serde's defaults.
    #[test]
    fn round_trips_through_json() {
        for event in sample_events() {
            let json = serde_json::to_value(&event).expect("serialize");
            let back: GraceEvent = serde_json::from_value(json).expect("deserialize");
            assert_eq!(event, back);
        }
    }

    /// `additionalProperties: false` in the schema means an optional field
    /// that is `None` must be OMITTED, not emitted as `null` - both
    /// `ToolExecutionStarted` and `ToolExecutionFinished` have optional
    /// fields the dispatcher's fast-path emissions never populate.
    #[test]
    fn optional_fields_are_omitted_not_null() {
        let json = serde_json::to_value(GraceEvent::ToolExecutionStarted {
            label: "Opening Notepad".into(),
            tool: None,
            step: None,
        })
        .unwrap();
        let obj = json.as_object().unwrap();
        assert!(!obj.contains_key("tool"));
        assert!(!obj.contains_key("step"));
        assert_eq!(obj.get("type").unwrap(), "ToolExecutionStarted");
    }
}
