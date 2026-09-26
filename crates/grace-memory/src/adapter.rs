//! Wires [`facts::FactStore`] and [`history::HistoryStore`] onto
//! `grace_core::memory::PersistentStore`, so `AgentMemory` can use this
//! crate exactly where it uses `InMemoryStore` today.
//!
//! The legacy trait is call-by-step (`save_step` mirrors the Python
//! original's flat, one-row-per-step `task_history`), with no episode
//! boundary and no app name - both of which §12.2's model wants. Rather
//! than guess at boundaries from timing gaps (fragile), this adapter
//! records each `save_step` call as its own one-step episode immediately;
//! that's honest given what the trait tells it, but it forfeits the
//! "several steps collapse into one routine" benefit that a caller using
//! [`begin_episode`]/[`end_episode`] (or `HistoryStore` directly) gets. See
//! PORT_STATUS.md for wiring `AgentLoop` to call the episode-boundary hooks
//! directly as a follow-up - not done here because it touches the agent
//! loop, outside this crate's scope.
//!
//! [`begin_episode`]: grace_core::memory::PersistentStore::begin_episode
//! [`end_episode`]: grace_core::memory::PersistentStore::end_episode

use crate::error::Result;
use crate::facts::{FactKind, FactStore, NewFactProvenance};
use crate::history::{EpisodeInput, HistoryConfig, HistoryStore, Outcome, StepInput, StepOutcome};
use grace_core::memory::PersistentStore;
use serde_json::Value;
use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct GraceMemoryStore {
    facts: FactStore,
    history: HistoryStore,
    turn: AtomicI64,
}

impl GraceMemoryStore {
    /// Opens (creating if needed) `<dir>/facts.db` and `<dir>/history.db`.
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let facts = FactStore::open(&dir.join("facts.db"))?;
        let history = HistoryStore::open(&dir.join("history.db"), HistoryConfig::default())?;
        Ok(Self { facts, history, turn: AtomicI64::new(0) })
    }

    /// Direct access to the facts store, for callers (confirmation flows,
    /// "forget that", conflict resolution by voice) that need §12.1's full
    /// API rather than the legacy trait's flat preference get/set.
    pub fn facts(&self) -> &FactStore {
        &self.facts
    }

    /// Direct access to the history store, for callers that track their own
    /// episode boundaries and want real routine collapsing rather than the
    /// one-step-per-episode fallback `save_step` uses.
    pub fn history(&self) -> &HistoryStore {
        &self.history
    }
}

impl PersistentStore for GraceMemoryStore {
    fn save_step(&mut self, user_goal: &str, action: &str, params: &Value, result: &Value) {
        let turn = self.turn.fetch_add(1, Ordering::Relaxed);
        let outcome = infer_outcome(result);
        let typed_text = params.get("text").and_then(Value::as_str).map(str::to_string);
        let episode = EpisodeInput {
            goal: user_goal.to_string(),
            // The legacy trait carries no app/window name; see this
            // module's doc comment.
            app: "unknown".to_string(),
            started_at_unix: unix_now(),
            duration_ms: 0,
            outcome,
            corrected: false,
            steps: vec![StepInput {
                ts_delta_ms: 0,
                action: action.to_string(),
                label: None,
                outcome: if matches!(outcome, Outcome::Failure) { StepOutcome::Error } else { StepOutcome::Ok },
                typed_text,
            }],
        };
        // A persistence hiccup must never crash a turn - best-effort, like
        // the Python original's own broad `except Exception` around every
        // SQLite call.
        let _ = self.history.record_episode(episode);
        let _ = turn;
    }

    fn set_preference(&mut self, key: &str, value: &str) {
        let topic = format!("preference:{key}");
        let turn = self.turn.load(Ordering::Relaxed);
        let result = if self.facts.active_fact(&topic).ok().flatten().is_some() {
            self.facts.supersede(&topic, FactKind::Preference, value, NewFactProvenance::Model, "preference updated", turn, false)
        } else {
            self.facts.store(&topic, None, FactKind::Preference, value, NewFactProvenance::Model, turn, false)
        };
        let _ = result;
    }

    fn get_preference(&self, key: &str) -> Option<String> {
        let topic = format!("preference:{key}");
        self.facts.active_fact(&topic).ok().flatten().map(|fact| fact.value)
    }
}

fn infer_outcome(result: &Value) -> Outcome {
    if result.get("error").is_some() {
        return Outcome::Failure;
    }
    match result.get("status").and_then(Value::as_str) {
        Some("error") | Some("failed") => Outcome::Failure,
        Some(_) => Outcome::Success,
        None => Outcome::Unknown,
    }
}

fn unix_now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn save_step_records_a_queryable_episode() {
        let dir = tempdir().unwrap();
        let mut store = GraceMemoryStore::open(dir.path()).unwrap();
        store.save_step("open calculator", "open_app", &json!({"name": "Calculator"}), &json!({"status": "ok"}));

        let results = store.history().search_episodes("calculator", 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn set_and_get_preference_round_trip_through_the_facts_store() {
        let dir = tempdir().unwrap();
        let mut store = GraceMemoryStore::open(dir.path()).unwrap();
        assert_eq!(store.get_preference("voice"), None);
        store.set_preference("voice", "af_bella");
        assert_eq!(store.get_preference("voice"), Some("af_bella".to_string()));
        // set_preference again must supersede, not fail on the unique index.
        store.set_preference("voice", "af_heart");
        assert_eq!(store.get_preference("voice"), Some("af_heart".to_string()));
    }

    #[test]
    fn implements_the_grace_core_persistent_store_trait_object() {
        let dir = tempdir().unwrap();
        let store: Box<dyn PersistentStore> = Box::new(GraceMemoryStore::open(dir.path()).unwrap());
        let mut memory = grace_core::memory::AgentMemory::new("goal", 0, 0.0).with_persistent_store(store);
        memory.add_step("t", "open_app", json!({"name": "Notepad"}), json!({"status": "ok"}), None);
        assert_eq!(memory.steps_taken.len(), 1);
    }
}
