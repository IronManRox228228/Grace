//! Ported from `src/grace/util/timing.py`: per-turn latency instrumentation,
//! producing exactly the shape `contract/grace-events.schema.json`'s
//! `TurnTrace` event carries (`grace_contract::TurnTraceData`).
//!
//! **Deliberate simplification, noted in PORT_STATUS.md**: Python keeps one
//! implicit "current trace" as a module-level global (guarded by a lock),
//! set for the whole turn so deeply nested code can record a stage without
//! threading a `&mut TurnTrace` through every call signature. Grace
//! processes one turn at a time regardless of language, so nothing this
//! port grades depends on the implicit-global mechanism itself - only on the
//! shape of the resulting trace, which is unaffected by whether the trace
//! reference travels implicitly or explicitly. This port passes `&mut
//! TurnTrace` explicitly instead, which is the idiomatic and safer choice in
//! Rust and produces identical `TurnTraceData` output. `TurnTrace` is
//! diagnostic-only and excluded from the harness's event diff regardless
//! (see `contract/README.md`), so this cannot affect parity grading.

use grace_contract::{TurnTraceData, TurnTraceStage};
use std::collections::BTreeMap;
use std::time::Instant;

struct StageRecord {
    name: String,
    start: Instant,
    depth: u32,
    duration_ms: Option<f64>,
    detail: Option<String>,
}

/// Timing record for a single user turn.
pub struct TurnTrace {
    label: String,
    start: Instant,
    stages: Vec<StageRecord>,
    events: BTreeMap<String, f64>,
    depth: u32,
}

/// A handle to one open stage. Closes (records its duration) when the
/// caller invokes `finish` or `detail`-then-drops-the-return-value - Rust
/// has no `with`/context-manager sugar, so this is called explicitly rather
/// than relying on `Drop` (a stage that must run to completion inside a
/// fallible block is exactly the case where an implicit-on-drop timer would
/// silently record a wrong duration if a `?` short-circuited early; explicit
/// `finish()` makes that impossible to get wrong by accident).
pub struct StageHandle {
    index: usize,
}

impl TurnTrace {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            start: Instant::now(),
            stages: Vec::new(),
            events: BTreeMap::new(),
            depth: 0,
        }
    }

    /// Opens a stage, incrementing nesting depth for whatever opens next.
    pub fn open_stage(&mut self, name: impl Into<String>) -> StageHandle {
        let record = StageRecord {
            name: name.into(),
            start: Instant::now(),
            depth: self.depth,
            duration_ms: None,
            detail: None,
        };
        self.depth += 1;
        self.stages.push(record);
        StageHandle { index: self.stages.len() - 1 }
    }

    /// Closes a stage opened with `open_stage`, recording its duration.
    pub fn close_stage(&mut self, handle: StageHandle) {
        if let Some(record) = self.stages.get_mut(handle.index) {
            if record.duration_ms.is_none() {
                record.duration_ms = Some(record.start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        self.depth = self.depth.saturating_sub(1);
    }

    /// Attaches a short note to an open (or closed) stage, e.g. element
    /// counts or token counts.
    pub fn detail(&mut self, handle: &StageHandle, text: impl Into<String>) {
        if let Some(record) = self.stages.get_mut(handle.index) {
            record.detail = Some(text.into());
        }
    }

    /// Records a stage whose duration was measured elsewhere.
    pub fn mark(&mut self, name: impl Into<String>, duration_ms: f64, detail: Option<String>) {
        self.stages.push(StageRecord {
            name: name.into(),
            start: Instant::now(),
            depth: self.depth,
            duration_ms: Some(duration_ms),
            detail,
        });
    }

    /// Records a point-in-time milestone as ms since the turn started. First
    /// occurrence wins - "first audio out" should not be overwritten by the
    /// second sentence.
    pub fn mark_event(&mut self, name: impl Into<String>) -> f64 {
        let offset_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        self.events.entry(name.into()).or_insert(offset_ms);
        offset_ms
    }

    pub fn total_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    pub fn to_data(&self) -> TurnTraceData {
        TurnTraceData {
            label: self.label.clone(),
            total_ms: round1(self.total_ms()),
            stages: self
                .stages
                .iter()
                .map(|s| TurnTraceStage {
                    name: s.name.clone(),
                    depth: s.depth,
                    ms: s.duration_ms.map(round1),
                    detail: s.detail.clone(),
                })
                .collect(),
            events: self.events.iter().map(|(k, v)| (k.clone(), round1(*v))).collect(),
        }
    }
}

fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_nest_by_depth() {
        let mut trace = TurnTrace::new("activation");
        let outer = trace.open_stage("outer");
        let inner = trace.open_stage("inner");
        trace.close_stage(inner);
        trace.close_stage(outer);

        let data = trace.to_data();
        assert_eq!(data.stages[0].name, "outer");
        assert_eq!(data.stages[0].depth, 0);
        assert_eq!(data.stages[1].name, "inner");
        assert_eq!(data.stages[1].depth, 1);
    }

    #[test]
    fn an_unclosed_stage_has_a_null_duration() {
        let mut trace = TurnTrace::new("t");
        let _open = trace.open_stage("never closes");
        let data = trace.to_data();
        assert_eq!(data.stages[0].ms, None);
    }

    #[test]
    fn detail_attaches_to_the_named_stage() {
        let mut trace = TurnTrace::new("t");
        let handle = trace.open_stage("element_graph");
        trace.detail(&handle, "12 elements via uia");
        trace.close_stage(handle);
        assert_eq!(trace.to_data().stages[0].detail.as_deref(), Some("12 elements via uia"));
    }

    #[test]
    fn mark_event_keeps_the_first_occurrence() {
        // Mirrors Python's `mark_event`: it always RETURNS the current
        // offset (both calls' return values legitimately differ), but the
        // stored `events[name]` keeps only the first occurrence - "first
        // audio out" must not be overwritten by the second sentence.
        let mut trace = TurnTrace::new("t");
        let first = trace.mark_event("first_audio_out");
        std::thread::sleep(std::time::Duration::from_millis(5));
        let second = trace.mark_event("first_audio_out");
        assert!(second > first);
        assert_eq!(trace.to_data().events.get("first_audio_out"), Some(&round1(first)));
    }

    #[test]
    fn to_data_matches_the_grace_contract_shape() {
        let mut trace = TurnTrace::new("activation");
        let s = trace.open_stage("whisper");
        trace.close_stage(s);
        trace.mark_event("route_fast_path");
        let data = trace.to_data();
        // Round-trips through the real contract type used in GraceEvent::TurnTrace.
        let event = grace_contract::GraceEvent::TurnTrace { trace: data };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "TurnTrace");
        assert!(json["trace"]["stages"].is_array());
    }
}
