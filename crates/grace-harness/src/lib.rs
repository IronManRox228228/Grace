//! Rust half of the migration harness (`src/grace/harness/tape.py`,
//! `replay.py`). Ported so far: the tape *reader* (`tape.rs`, read-only, so
//! it can never write to or regenerate `corpus/`) and the event/dispatch
//! diff machinery (`diff.rs`), both faithful to the Python originals.
//!
//! What this crate's `grace-replay` binary currently checks against
//! `corpus/` is **structural parity**: every tape loads, and every event in
//! it deserializes as a valid `grace_contract::GraceEvent` - i.e. the Rust
//! contract types can represent everything the existing corpus contains.
//! It does NOT yet re-run a turn against the Rust backend and diff the
//! result; that needs the ported agent loop, dispatcher and perception
//! pipeline this phase did not reach (see PORT_STATUS.md). Running
//! `diff_event_streams`/`diff_dispatches` against a Rust-produced event
//! stream is the next step once those are ported, and this crate's `diff`
//! module is already the piece that will grade it.
//!
//! Per PLAN.md §0/§10.2 and the task brief: the committed `corpus/` tapes
//! were recorded against Python `HEAD`, not the uncommitted working tree -
//! the working tree's tool schema changed, so prompts (and therefore some
//! downstream events) differ starting at prompt char 1607. A structural
//! failure to parse is a real bug; a tape that parses cleanly says nothing
//! about whether the *content* still matches the working tree, and the
//! `grace-replay` report says so explicitly rather than claiming false
//! parity.

pub mod diff;
pub mod tape;
pub mod turn_replay;

pub use diff::{diff_dispatches, diff_event_streams, without_turn_trace, Diff};
pub use tape::Tape;
pub use turn_replay::{replay_fastpath_tape, TurnReplayResult};

use grace_contract::GraceEvent;

/// The result of checking one tape's structural parity against the frozen
/// contract.
#[derive(Debug)]
pub struct TapeReport {
    pub session_id: String,
    pub is_synthetic: bool,
    pub event_count: usize,
    /// Events that failed to deserialize as a `GraceEvent`, as
    /// `(index, error)` pairs.
    pub schema_failures: Vec<(usize, String)>,
}

impl TapeReport {
    pub fn ok(&self) -> bool {
        self.schema_failures.is_empty()
    }
}

/// Checks that every event in `tape` deserializes as a `GraceEvent` - i.e.
/// the frozen contract types can represent everything this tape recorded.
pub fn check_tape_schema(tape: &Tape) -> TapeReport {
    let session_id = tape
        .meta
        .get("session_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(&tape.dir)
        .to_string();

    let payloads = tape.event_payloads();
    let mut failures = Vec::new();
    for (index, payload) in payloads.iter().enumerate() {
        if let Err(e) = serde_json::from_value::<GraceEvent>(payload.clone()) {
            failures.push((index, e.to_string()));
        }
    }

    TapeReport {
        session_id,
        is_synthetic: tape.is_synthetic(),
        event_count: payloads.len(),
        schema_failures: failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Runs the structural-parity check against every tape actually
    /// committed under `corpus/`. This is a real regression test - it reads
    /// the real corpus - but it purposefully does not assert route/prompt
    /// parity (see this module's doc comment for why that gate isn't ready
    /// yet), only that every event the existing corpus contains still fits
    /// the frozen `GraceEvent` contract.
    #[test]
    fn every_corpus_tape_event_fits_the_frozen_contract() {
        let corpus_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
        if !corpus_dir.exists() {
            eprintln!("corpus/ not found at {}; skipping", corpus_dir.display());
            return;
        }
        let tapes = Tape::load_corpus(&corpus_dir).expect("corpus loads");
        assert!(!tapes.is_empty(), "expected at least one tape in corpus/");

        let mut failing = Vec::new();
        for tape in &tapes {
            let report = check_tape_schema(tape);
            if !report.ok() {
                failing.push(format!("{}: {:?}", report.session_id, report.schema_failures));
            }
        }
        assert!(
            failing.is_empty(),
            "tapes with events the GraceEvent contract cannot represent: {failing:#?}"
        );
    }
}
