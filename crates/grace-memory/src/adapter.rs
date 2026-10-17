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

struct ActiveEpisode {
    goal: String,
    app: String,
    started_at_unix: i64,
    steps: Vec<StepInput>,
}

pub struct GraceMemoryStore {
    facts: FactStore,
    history: HistoryStore,
    turn: AtomicI64,
    current_episode: Option<ActiveEpisode>,
}

impl GraceMemoryStore {
    /// Opens (creating if needed) `<dir>/facts.db` and `<dir>/history.db`.
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let facts = FactStore::open(&dir.join("facts.db"))?;
        let history = HistoryStore::open(&dir.join("history.db"), HistoryConfig::default())?;
        Ok(Self {
            facts,
            history,
            turn: AtomicI64::new(0),
            current_episode: None,
        })
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

impl Drop for GraceMemoryStore {
    fn drop(&mut self) {
        if let Some(active) = self.current_episode.take() {
            if !active.steps.is_empty() {
                let app = if active.app.is_empty() { "unknown".to_string() } else { active.app };
                let episode = EpisodeInput {
                    goal: active.goal,
                    app,
                    started_at_unix: active.started_at_unix,
                    duration_ms: (((unix_now() - active.started_at_unix).max(0) * 1000).min(u32::MAX as i64)) as u32,
                    outcome: Outcome::Unknown,
                    corrected: false,
                    steps: active.steps,
                };
                let _ = self.history.record_episode(episode);
            }
        }
    }
}

impl PersistentStore for GraceMemoryStore {
    fn save_step(&mut self, user_goal: &str, app: &str, action: &str, params: &Value, result: &Value) {
        let _turn = self.turn.fetch_add(1, Ordering::Relaxed);
        let outcome = infer_outcome(result);
        let typed_text = params.get("text").and_then(Value::as_str).map(str::to_string);
        let step = StepInput {
            ts_delta_ms: 0,
            action: action.to_string(),
            label: None,
            outcome: if matches!(outcome, Outcome::Failure) { StepOutcome::Error } else { StepOutcome::Ok },
            typed_text,
        };

        if let Some(active) = &mut self.current_episode {
            if active.app.is_empty() && !app.is_empty() && app != "unknown" {
                active.app = app.to_string();
            }
            active.steps.push(step);
        } else {
            // Fallback for callers that don't bracket turns with begin_episode / end_episode:
            let app_name = if app.is_empty() { "unknown".to_string() } else { app.to_string() };
            let episode = EpisodeInput {
                goal: user_goal.to_string(),
                app: app_name,
                started_at_unix: unix_now(),
                duration_ms: 0,
                outcome,
                corrected: false,
                steps: vec![step],
            };
            let _ = self.history.record_episode(episode);
        }
    }

    fn begin_episode(&mut self, user_goal: &str) {
        if let Some(previous) = self.current_episode.take() {
            if !previous.steps.is_empty() {
                let app = if previous.app.is_empty() { "unknown".to_string() } else { previous.app };
                let episode = EpisodeInput {
                    goal: previous.goal,
                    app,
                    started_at_unix: previous.started_at_unix,
                    duration_ms: (((unix_now() - previous.started_at_unix).max(0) * 1000).min(u32::MAX as i64)) as u32,
                    outcome: Outcome::Unknown,
                    corrected: false,
                    steps: previous.steps,
                };
                let _ = self.history.record_episode(episode);
            }
        }
        self.current_episode = Some(ActiveEpisode {
            goal: user_goal.to_string(),
            app: String::new(),
            started_at_unix: unix_now(),
            steps: Vec::new(),
        });
    }

    fn end_episode(&mut self, outcome_hint: grace_core::memory::EpisodeOutcomeHint) {
        if let Some(active) = self.current_episode.take() {
            let outcome = match outcome_hint {
                grace_core::memory::EpisodeOutcomeHint::Success => Outcome::Success,
                grace_core::memory::EpisodeOutcomeHint::Failure => Outcome::Failure,
                grace_core::memory::EpisodeOutcomeHint::Unknown => Outcome::Unknown,
            };
            let app = if active.app.is_empty() { "unknown".to_string() } else { active.app };
            let episode = EpisodeInput {
                goal: active.goal,
                app,
                started_at_unix: active.started_at_unix,
                duration_ms: (((unix_now() - active.started_at_unix).max(0) * 1000).min(u32::MAX as i64)) as u32,
                outcome,
                corrected: false,
                steps: active.steps,
            };
            let _ = self.history.record_episode(episode);
        }
    }

    fn render_prompt_context(&self, budget_chars: usize) -> Option<String> {
        self.facts.render_context(budget_chars).ok().filter(|s| !s.is_empty())
    }

    fn store_fact_heard(&mut self, topic: &str, group_key: Option<&str>, value: &str) -> std::result::Result<i64, String> {
        let turn = self.turn.fetch_add(1, Ordering::Relaxed);
        self.facts
            .store(topic, group_key, FactKind::Fact, value, NewFactProvenance::UserHeard, turn, false)
            .map_err(|e| e.to_string())
    }

    fn confirm_fact(&mut self, topic: &str) -> std::result::Result<(), String> {
        let turn = self.turn.fetch_add(1, Ordering::Relaxed);
        self.facts.confirm(topic, turn).map_err(|e| e.to_string())
    }

    fn forget_fact(&mut self, topic: &str) -> std::result::Result<(), String> {
        self.facts.forget(topic).map_err(|e| e.to_string())
    }

    fn check_conflicts(&self, group_key: &str) -> Vec<String> {
        self.facts
            .conflicts(group_key)
            .map(|list| list.into_iter().map(|f| format!("{}: {}", f.topic, f.value)).collect())
            .unwrap_or_default()
    }

    fn forget_today(&mut self, start_unix: i64, end_unix: i64) -> std::result::Result<u64, String> {
        self.history.forget_today(start_unix, end_unix).map_err(|e| e.to_string())
    }

    fn forget_app(&mut self, app: &str) -> std::result::Result<u64, String> {
        self.history.forget_app(app).map_err(|e| e.to_string())
    }

    fn forget_everything(&mut self) -> std::result::Result<(), String> {
        self.facts.forget_everything().map_err(|e| e.to_string())?;
        self.history.forget_everything().map_err(|e| e.to_string())
    }

    fn compact_once(&mut self, batch_limit: usize) -> std::result::Result<(), String> {
        self.history.compact_once(unix_now(), batch_limit).map(|_| ()).map_err(|e| e.to_string())
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
        store.save_step("open calculator", "Calculator", "open_app", &json!({"name": "Calculator"}), &json!({"status": "ok"}));

        let results = store.history().search_episodes("calculator", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].app, "Calculator");
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

    #[test]
    fn multi_step_episode_collapses_at_boundaries() {
        let dir = tempdir().unwrap();
        let mut store = GraceMemoryStore::open(dir.path()).unwrap();
        store.begin_episode("search docs");
        store.save_step("search docs", "Chrome", "open_app", &json!({"name": "Chrome"}), &json!({"status": "ok"}));
        store.save_step("search docs", "Chrome", "cua_type_text", &json!({"text": "rust docs"}), &json!({"status": "ok"}));
        store.end_episode(grace_core::memory::EpisodeOutcomeHint::Success);

        let results = store.history().search_episodes("search docs", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].app, "Chrome");
        assert_eq!(results[0].step_count, 2);
    }

    #[test]
    fn fact_confirmation_flow_and_prompt_injection() {
        let dir = tempdir().unwrap();
        let mut store = GraceMemoryStore::open(dir.path()).unwrap();
        store.store_fact_heard("family:sister", Some("family"), "Priya").unwrap();
        // Pending fact is not rendered in prompt context
        assert_eq!(store.render_prompt_context(1000), None);

        // Confirm promotes to user_confirmed and renders in prompt
        store.confirm_fact("family:sister").unwrap();
        let rendered = store.render_prompt_context(1000).unwrap();
        assert!(rendered.contains("family:sister: Priya"));

        // Forget fact removes it
        store.forget_fact("family:sister").unwrap();
        assert_eq!(store.render_prompt_context(1000), None);
    }

    #[test]
    fn voice_hard_forget_and_compaction() {
        let dir = tempdir().unwrap();
        let mut store = GraceMemoryStore::open(dir.path()).unwrap();
        store.save_step("open chrome", "Chrome", "open_app", &json!({"name": "Chrome"}), &json!({"status": "ok"}));
        store.save_step("open word", "Word", "open_app", &json!({"name": "Word"}), &json!({"status": "ok"}));
        assert_eq!(store.history().search_episodes("chrome", 10).unwrap().len(), 1);

        // Forget app
        store.forget_app("Chrome").unwrap();
        assert_eq!(store.history().search_episodes("chrome", 10).unwrap().len(), 0);
        assert_eq!(store.history().search_episodes("word", 10).unwrap().len(), 1);

        // Compact once
        store.compact_once(100).unwrap();

        // Forget everything
        store.forget_everything().unwrap();
        assert_eq!(store.history().search_episodes("word", 10).unwrap().len(), 0);
    }
}
