//! Validates that every `GraceEvent` variant this crate can produce actually
//! satisfies `contract/grace-events.schema.json` - the frozen, hand-reviewed
//! schema the Python backend is also checked against
//! (`src/grace/harness/contract.py`, `tests/test_contract.py`). This is the
//! test PLAN.md §11 and the task brief ask for on `grace-contract`: a
//! round-trip against the schema, not just against itself.
//!
//! This test reads the schema file from the repo (a relative path from the
//! crate root); it does not embed a copy, so the schema stays a single
//! source of truth and this test fails loudly if the two ever diverge.

use grace_contract::{GraceEvent, TurnTraceData, TurnTraceStage};
use jsonschema::validator_for;
use std::collections::BTreeMap;
use std::path::Path;

fn schema() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contract/grace-events.schema.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("schema is valid JSON")
}

fn all_variants() -> Vec<GraceEvent> {
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
                stages: vec![
                    TurnTraceStage {
                        name: "stt".into(),
                        depth: 0,
                        ms: Some(300.1),
                        detail: None,
                    },
                    TurnTraceStage {
                        name: "aborted_stage".into(),
                        depth: 1,
                        ms: None,
                        detail: None,
                    },
                ],
                events: BTreeMap::from([("wake_to_listen_ms".to_string(), 22.9)]),
            },
        },
        GraceEvent::Error {
            message: "sidecar unavailable".into(),
        },
    ]
}

#[test]
fn every_variant_validates_against_the_frozen_schema() {
    let schema = schema();
    let validator = validator_for(&schema).expect("schema compiles");

    for event in all_variants() {
        let json = serde_json::to_value(&event).expect("serialize");
        let errors: Vec<String> = validator.iter_errors(&json).map(|e| e.to_string()).collect();
        assert!(
            errors.is_empty(),
            "{:?} serialized to {json} but failed schema validation: {errors:?}",
            event
        );
    }
}

/// The schema's `oneOf` has exactly as many branches as this enum has
/// *distinct* variants (`all_variants()` deliberately samples
/// `ToolExecutionStarted`/`ToolExecutionFinished` twice each, with and
/// without their optional fields, so it has more entries than there are
/// variants - hence comparing the set of `"type"` discriminants produced,
/// not the sample count). If someone adds a branch to the schema without a
/// matching Rust variant (or vice versa), this is the tripwire.
#[test]
fn schema_and_enum_have_the_same_number_of_variants() {
    let schema = schema();
    let one_of = schema["oneOf"].as_array().expect("oneOf is an array");
    // Deliberately a plain literal, not derived from any Rust-side count, so
    // a variant that forgot to get added to `all_variants` still fails this
    // test rather than passing by accident.
    const SCHEMA_VARIANT_COUNT: usize = 17;
    assert_eq!(one_of.len(), SCHEMA_VARIANT_COUNT);

    let distinct_types: std::collections::BTreeSet<String> = all_variants()
        .iter()
        .map(|event| serde_json::to_value(event).unwrap()["type"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(distinct_types.len(), SCHEMA_VARIANT_COUNT);
}
