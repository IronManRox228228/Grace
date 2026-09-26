//! Ported from `src/grace/harness/tape.py`'s `Diff`, `diff_event_streams`
//! and `diff_dispatches`. Byte-exact, order-sensitive comparison: "same
//! events, different order" or "right outcome, different route" are both
//! reported as mismatches, not near-misses.

use serde_json::Value;

#[derive(Debug, Clone, Default)]
pub struct Diff {
    pub kind: String,
    pub mismatches: Vec<String>,
}

impl Diff {
    pub fn ok(&self) -> bool {
        self.mismatches.is_empty()
    }

    pub fn report(&self) -> String {
        if self.ok() {
            return format!("{}: OK", self.kind);
        }
        let mut lines = vec![format!("{}: {} mismatch(es)", self.kind, self.mismatches.len())];
        lines.extend(self.mismatches.iter().map(|m| format!("  {m}")));
        lines.join("\n")
    }
}

fn brief(value: Option<&Value>, limit: usize) -> String {
    match value {
        None => "<none>".to_string(),
        Some(v) => {
            let s = v.to_string();
            if s.len() > limit {
                format!("{}...", &s[..limit])
            } else {
                s
            }
        }
    }
}

/// Byte-exact comparison of two event sequences. Order matters and is the
/// point: the UI is a state machine driven by this sequence.
///
/// `TurnTrace` events must be filtered out by the caller before calling this
/// (mirrors `contract/README.md`: "TurnTrace is excluded from the event
/// diff: it carries wall-clock durations that legitimately differ on every
/// run").
pub fn diff_event_streams(expected: &[Value], actual: &[Value]) -> Diff {
    let mut diff = Diff {
        kind: "events".to_string(),
        mismatches: Vec::new(),
    };
    let len = expected.len().max(actual.len());
    for index in 0..len {
        let want = expected.get(index);
        let got = actual.get(index);
        if want == got {
            continue;
        }
        match (want, got) {
            (None, Some(g)) => diff
                .mismatches
                .push(format!("[{index}] unexpected extra event: {}", brief(Some(g), 160))),
            (Some(w), None) => diff
                .mismatches
                .push(format!("[{index}] missing event: {}", brief(Some(w), 160))),
            (Some(w), Some(g)) => {
                let want_type = w.get("type").and_then(Value::as_str);
                let got_type = g.get("type").and_then(Value::as_str);
                if want_type != got_type {
                    diff.mismatches.push(format!(
                        "[{index}] type: expected {:?}, got {:?}",
                        want_type, got_type
                    ));
                } else {
                    diff.mismatches.push(format!(
                        "[{index}] {:?} payload: expected {}, got {}",
                        want_type,
                        brief(Some(w), 160),
                        brief(Some(g), 160)
                    ));
                }
            }
            (None, None) => unreachable!(),
        }
    }
    diff
}

/// Byte-exact comparison of the outbound tool-dispatch sequence. Catches
/// "ended up in the right place, but took a different route" - PLAN.md's
/// note that a fast-path tool call and an agentic one that reaches the same
/// state have NOT reproduced the same behaviour.
pub fn diff_dispatches(expected: &[Value], actual: &[Value]) -> Diff {
    let mut diff = Diff {
        kind: "dispatch".to_string(),
        mismatches: Vec::new(),
    };
    let len = expected.len().max(actual.len());
    for index in 0..len {
        let want = expected.get(index);
        let got = actual.get(index);
        if want == got {
            continue;
        }
        match (want, got) {
            (None, Some(g)) => diff
                .mismatches
                .push(format!("[{index}] unexpected extra call: {}", brief(Some(g), 160))),
            (Some(w), None) => diff
                .mismatches
                .push(format!("[{index}] missing call: {}", brief(Some(w), 160))),
            (Some(w), Some(g)) => diff.mismatches.push(format!(
                "[{index}] expected {}({}), got {}({})",
                w.get("tool").map(|v| v.to_string()).unwrap_or_default(),
                w.get("params").map(|v| v.to_string()).unwrap_or_default(),
                g.get("tool").map(|v| v.to_string()).unwrap_or_default(),
                g.get("params").map(|v| v.to_string()).unwrap_or_default(),
            )),
            (None, None) => unreachable!(),
        }
    }
    diff
}

/// Filters `TurnTrace` entries out of a raw event-payload stream before
/// diffing, per `contract/README.md`'s exclusion rule.
pub fn without_turn_trace(events: &[Value]) -> Vec<Value> {
    events
        .iter()
        .filter(|e| e.get("type").and_then(Value::as_str) != Some("TurnTrace"))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn identical_streams_diff_clean() {
        let events = vec![json!({"type": "Idle"}), json!({"type": "WakeWordDetected"})];
        let diff = diff_event_streams(&events, &events);
        assert!(diff.ok());
    }

    #[test]
    fn reordered_events_are_a_mismatch_even_with_the_same_set() {
        let expected = vec![json!({"type": "A"}), json!({"type": "B"})];
        let actual = vec![json!({"type": "B"}), json!({"type": "A"})];
        let diff = diff_event_streams(&expected, &actual);
        assert!(!diff.ok());
        assert_eq!(diff.mismatches.len(), 2);
    }

    #[test]
    fn missing_and_extra_events_are_reported_distinctly() {
        let expected = vec![json!({"type": "Idle"})];
        let actual = vec![json!({"type": "Idle"}), json!({"type": "Extra"})];
        let diff = diff_event_streams(&expected, &actual);
        assert!(diff.mismatches[0].contains("unexpected extra event"));
    }

    #[test]
    fn same_type_different_payload_is_reported_as_a_payload_mismatch() {
        let expected = vec![json!({"type": "FinalTranscript", "text": "open notepad"})];
        let actual = vec![json!({"type": "FinalTranscript", "text": "open notpad"})];
        let diff = diff_event_streams(&expected, &actual);
        assert!(diff.mismatches[0].contains("payload"));
    }

    #[test]
    fn turn_trace_is_excluded_before_diffing() {
        let events = vec![
            json!({"type": "Idle"}),
            json!({"type": "TurnTrace", "trace": {"total_ms": 123.0}}),
        ];
        assert_eq!(without_turn_trace(&events), vec![json!({"type": "Idle"})]);
    }

    #[test]
    fn dispatch_diff_reports_a_different_route() {
        let expected = vec![json!({"tool": "open_app", "params": {"name": "Notepad"}})];
        let actual = vec![json!({"tool": "cua_launch", "params": {"name": "Notepad"}})];
        let diff = diff_dispatches(&expected, &actual);
        assert!(!diff.ok());
        assert!(diff.mismatches[0].contains("open_app"));
        assert!(diff.mismatches[0].contains("cua_launch"));
    }
}
