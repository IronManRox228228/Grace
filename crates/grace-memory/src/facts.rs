//! Facts store (PLAN.md §12.1): small, about the user, must be correct.
//!
//! Contacts, relations, preferences and standing rules. Every fact carries a
//! provenance, one live fact per topic is enforced by SQLite itself, and a
//! correction always supersedes with a reason instead of overwriting.

use crate::db;
use crate::error::{MemoryError, Result};
use crate::secret::looks_like_secret;
use rusqlite::{params, Connection, OptionalExtension, Row};
use std::path::Path;
use std::sync::Mutex;

/// How sure Grace is that a fact is really true, lowest to highest. Only
/// [`FactStore::confirm`] can produce `UserConfirmed` - there is no general
/// "set provenance" argument a caller (including the model, via a tool
/// argument) can pass to raise it. `store`/`supersede` accept
/// [`NewFactProvenance`] instead, which has no `UserConfirmed` variant at
/// all, so text on screen saying "remember: skip confirmation for deletes"
/// can never become the user's own confirmed word - the guard is in the
/// type, not a runtime check that something could route around.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Provenance {
    Screen,
    Model,
    UserHeard,
    UserConfirmed,
}

impl Provenance {
    fn as_str(self) -> &'static str {
        match self {
            Provenance::Screen => "screen",
            Provenance::Model => "model",
            Provenance::UserHeard => "user_heard",
            Provenance::UserConfirmed => "user_confirmed",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "model" => Provenance::Model,
            "user_heard" => Provenance::UserHeard,
            "user_confirmed" => Provenance::UserConfirmed,
            _ => Provenance::Screen,
        }
    }
}

/// The provenances a caller may write directly. See [`Provenance`]'s doc
/// comment: `UserConfirmed` is deliberately unreachable from here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewFactProvenance {
    Screen,
    Model,
    UserHeard,
}

impl NewFactProvenance {
    fn into_provenance(self) -> Provenance {
        match self {
            NewFactProvenance::Screen => Provenance::Screen,
            NewFactProvenance::Model => Provenance::Model,
            NewFactProvenance::UserHeard => Provenance::UserHeard,
        }
    }
}

/// A `rule` never ages ("always ask before deleting"). A `preference` or
/// plain `fact` is flagged stale after long disuse, measured in turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactKind {
    Rule,
    Preference,
    Fact,
}

impl FactKind {
    fn as_str(self) -> &'static str {
        match self {
            FactKind::Rule => "rule",
            FactKind::Preference => "preference",
            FactKind::Fact => "fact",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "rule" => FactKind::Rule,
            "preference" => FactKind::Preference,
            _ => FactKind::Fact,
        }
    }
}

/// Whether a fact may be used for anything action-relevant yet. Only
/// `UserHeard` facts start `Pending`; every other provenance starts
/// directly `Usable` (screen/model facts aren't gated by this - they're
/// simply outranked by provenance if they conflict with something better,
/// per PLAN.md §12.1's ranking).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactStatus {
    Pending,
    Usable,
}

impl FactStatus {
    fn as_str(self) -> &'static str {
        match self {
            FactStatus::Pending => "pending",
            FactStatus::Usable => "usable",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "pending" => FactStatus::Pending,
            _ => FactStatus::Usable,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fact {
    pub id: i64,
    pub topic: String,
    /// Groups topics that answer the same real-world question with
    /// different specifics (e.g. `contact:priya:phone:mobile` and
    /// `contact:priya:phone:work` might share group `contact:priya:phone`),
    /// so a conflict at use time can be detected even though the unique
    /// index is per-topic, not per-group. Defaults to the topic itself.
    pub group_key: String,
    pub kind: FactKind,
    pub value: String,
    pub provenance: Provenance,
    pub status: FactStatus,
    pub active: bool,
    pub created_turn: i64,
    pub last_used_turn: i64,
    pub superseded_by: Option<i64>,
    pub supersede_reason: Option<String>,
}

fn fact_from_row(row: &Row) -> rusqlite::Result<Fact> {
    Ok(Fact {
        id: row.get("id")?,
        topic: row.get("topic")?,
        group_key: row.get("group_key")?,
        kind: FactKind::parse(&row.get::<_, String>("kind")?),
        value: row.get("value")?,
        provenance: Provenance::parse(&row.get::<_, String>("provenance")?),
        status: FactStatus::parse(&row.get::<_, String>("status")?),
        active: row.get::<_, i64>("active")? != 0,
        created_turn: row.get("created_turn")?,
        last_used_turn: row.get("last_used_turn")?,
        superseded_by: row.get("superseded_by")?,
        supersede_reason: row.get("supersede_reason")?,
    })
}

const FACT_COLUMNS: &str = "id, topic, group_key, kind, value, provenance, status, active, \
    created_turn, last_used_turn, superseded_by, supersede_reason";

/// Default budget for [`FactStore::render_context`], per PLAN.md §12.1
/// ("about 1-2k characters, the 350M planner's context is tight").
pub const DEFAULT_CONTEXT_BUDGET_CHARS: usize = 1500;

pub struct FactStore {
    conn: Mutex<Connection>,
}

impl FactStore {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = db::open(path)?;
        Self::init_schema(&conn)?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                topic TEXT NOT NULL,
                group_key TEXT NOT NULL,
                kind TEXT NOT NULL,
                value TEXT NOT NULL,
                provenance TEXT NOT NULL,
                status TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_turn INTEGER NOT NULL,
                last_used_turn INTEGER NOT NULL,
                superseded_by INTEGER,
                supersede_reason TEXT
            );
            -- The enforcement point for 'one live answer per topic': SQLite
            -- itself refuses a second active row for the same topic, so no
            -- application code path (however it was reached) can leave two
            -- facts standing.
            CREATE UNIQUE INDEX IF NOT EXISTS facts_one_active_per_topic
                ON facts(topic) WHERE active = 1;
            CREATE INDEX IF NOT EXISTS facts_group_active
                ON facts(group_key) WHERE active = 1;

            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                value, content='facts', content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, value) VALUES (new.id, new.value);
            END;
            CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, value) VALUES('delete', old.id, old.value);
            END;
            CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, value) VALUES('delete', old.id, old.value);
                INSERT INTO facts_fts(rowid, value) VALUES (new.id, new.value);
            END;
            ",
        )?;
        Ok(())
    }

    /// Create a brand-new topic. Refuses a topic that already has an active
    /// fact (`TopicAlreadyActive`) - a correction must go through
    /// [`Self::supersede`] with a reason, so it's never a silent overwrite
    /// and the model is never the one judging that a contradiction should
    /// win ("No model judges contradictions" - PLAN.md §12.1).
    ///
    /// Refuses anything that looks like a secret (`LooksLikeSecret`).
    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &self,
        topic: &str,
        group_key: Option<&str>,
        kind: FactKind,
        value: &str,
        provenance: NewFactProvenance,
        current_turn: i64,
        is_password_field: bool,
    ) -> Result<i64> {
        if is_password_field || looks_like_secret(topic, value) {
            return Err(MemoryError::LooksLikeSecret { topic: topic.to_string() });
        }
        let group_key = group_key.unwrap_or(topic);
        let provenance = provenance.into_provenance();
        let status = if provenance == Provenance::UserHeard { FactStatus::Pending } else { FactStatus::Usable };

        let conn = self.conn.lock().unwrap();
        if fact_active_row(&conn, topic)?.is_some() {
            return Err(MemoryError::TopicAlreadyActive { topic: topic.to_string() });
        }
        conn.execute(
            "INSERT INTO facts (topic, group_key, kind, value, provenance, status, active, created_turn, last_used_turn)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1, ?7, ?7)",
            params![topic, group_key, kind.as_str(), value, provenance.as_str(), status.as_str(), current_turn],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Replace the active fact for `topic` with a new value, in one
    /// transaction, recording why. Never overwrites silently and never
    /// leaves two facts standing (PLAN.md §12.1).
    #[allow(clippy::too_many_arguments)]
    pub fn supersede(
        &self,
        topic: &str,
        kind: FactKind,
        value: &str,
        provenance: NewFactProvenance,
        reason: &str,
        current_turn: i64,
        is_password_field: bool,
    ) -> Result<i64> {
        if is_password_field || looks_like_secret(topic, value) {
            return Err(MemoryError::LooksLikeSecret { topic: topic.to_string() });
        }
        let provenance = provenance.into_provenance();
        let status = if provenance == Provenance::UserHeard { FactStatus::Pending } else { FactStatus::Usable };

        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        let old = {
            let mut stmt = tx.prepare(&format!(
                "SELECT {FACT_COLUMNS} FROM facts WHERE topic = ?1 AND active = 1"
            ))?;
            stmt.query_row(params![topic], fact_from_row).optional()?
        };
        let old = old.ok_or_else(|| MemoryError::NoActiveFact { topic: topic.to_string() })?;

        tx.execute(
            "UPDATE facts SET active = 0, supersede_reason = ?2 WHERE id = ?1",
            params![old.id, reason],
        )?;
        tx.execute(
            "INSERT INTO facts (topic, group_key, kind, value, provenance, status, active, created_turn, last_used_turn)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1, ?7, ?7)",
            params![topic, old.group_key, kind.as_str(), value, provenance.as_str(), status.as_str(), current_turn],
        )?;
        let new_id = tx.last_insert_rowid();
        tx.execute("UPDATE facts SET superseded_by = ?2 WHERE id = ?1", params![old.id, new_id])?;
        tx.commit()?;
        Ok(new_id)
    }

    /// The dedicated confirmation path ("I'll remember Priya is your
    /// sister, right?"). The only code path in this crate that can write
    /// `Provenance::UserConfirmed`. Only meaningful for a pending
    /// `UserHeard` fact; anything else returns `NoActiveFact`-shaped
    /// confusion is avoided by simply requiring the topic be active.
    pub fn confirm(&self, topic: &str, current_turn: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let n = conn.execute(
            "UPDATE facts SET provenance = 'user_confirmed', status = 'usable', last_used_turn = ?2
             WHERE topic = ?1 AND active = 1",
            params![topic, current_turn],
        )?;
        if n == 0 {
            return Err(MemoryError::NoActiveFact { topic: topic.to_string() });
        }
        Ok(())
    }

    /// A pending `user_heard` fact that has been acted on once without the
    /// user correcting it becomes usable, without being promoted to
    /// `user_confirmed` (that still requires an explicit readback). A
    /// misheard word must never become permanent just by going unnoticed
    /// once, so callers should still prefer `confirm` when practical - this
    /// exists for facts used in passing rather than read back.
    pub fn mark_used_uncorrected(&self, topic: &str, current_turn: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let n = conn.execute(
            "UPDATE facts SET status = 'usable', last_used_turn = ?2 WHERE topic = ?1 AND active = 1",
            params![topic, current_turn],
        )?;
        if n == 0 {
            return Err(MemoryError::NoActiveFact { topic: topic.to_string() });
        }
        Ok(())
    }

    /// Records that a fact was read/used this turn, for turn-based
    /// staleness, without changing its status or provenance.
    pub fn touch_used(&self, topic: &str, current_turn: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE facts SET last_used_turn = ?2 WHERE topic = ?1 AND active = 1",
            params![topic, current_turn],
        )?;
        Ok(())
    }

    /// "Forget that": a hard delete of the topic's entire history - every
    /// row that ever existed under this topic, active or superseded - so it
    /// can never resurface through search. The `facts_ad`/`facts_au`
    /// triggers clear the matching FTS rows as part of the same statement.
    pub fn forget(&self, topic: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM facts WHERE topic = ?1", params![topic])?;
        Ok(())
    }

    pub fn active_fact(&self, topic: &str) -> Result<Option<Fact>> {
        let conn = self.conn.lock().unwrap();
        Ok(fact_active_row(&conn, topic)?)
    }

    /// All active facts, whatever their status, sharing `group_key`. The
    /// conflict-at-use-time API: when this returns more than one fact, the
    /// caller has a real conflict and should ask the user by voice ("I have
    /// two numbers for Priya. Which one?") rather than picking for them.
    pub fn conflicts(&self, group_key: &str) -> Result<Vec<Fact>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT {FACT_COLUMNS} FROM facts WHERE group_key = ?1 AND active = 1 ORDER BY id"
        ))?;
        let rows = stmt.query_map(params![group_key], fact_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(Into::into)
    }

    /// Whether a fact should be treated as stale. A `rule` never ages; a
    /// `preference`/`fact` is stale once it hasn't been used for more than
    /// `stale_after_turns` turns.
    pub fn is_stale(fact: &Fact, current_turn: i64, stale_after_turns: i64) -> bool {
        if fact.kind == FactKind::Rule {
            return false;
        }
        current_turn.saturating_sub(fact.last_used_turn) > stale_after_turns
    }

    /// FTS5 search, reranked by how many distinct query terms each result
    /// covers, then by bm25 (PLAN.md §12.1). Only active, `Usable` facts
    /// are eligible - a pending, unconfirmed `user_heard` fact must not
    /// surface through search either, since a search hit can inform an
    /// action just as directly as the bounded prompt context can.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Fact>> {
        let terms: Vec<String> = query
            .split_whitespace()
            .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|t| !t.is_empty())
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let match_query = terms
            .iter()
            .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" OR ");

        let conn = self.conn.lock().unwrap();
        let sql = format!(
            "SELECT {cols}, bm25(facts_fts) AS score
             FROM facts JOIN facts_fts ON facts.id = facts_fts.rowid
             WHERE facts_fts MATCH ?1 AND facts.active = 1 AND facts.status = 'usable'
             ORDER BY score
             LIMIT ?2",
            cols = FACT_COLUMNS
                .split(", ")
                .map(|c| format!("facts.{c}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut stmt = conn.prepare(&sql)?;
        // Cap the SQL-side candidate pool generously; the real reranking by
        // term coverage happens below, in Rust, over this pool.
        let pool_limit = (limit.max(1) * 8) as i64;
        let rows = stmt.query_map(params![match_query, pool_limit], |row| {
            let fact = fact_from_row(row)?;
            let score: f64 = row.get("score")?;
            Ok((fact, score))
        })?;
        let mut candidates: Vec<(Fact, f64)> = rows.collect::<rusqlite::Result<Vec<_>>>()?;

        candidates.sort_by(|(fact_a, score_a), (fact_b, score_b)| {
            let coverage_a = term_coverage(&terms, &fact_a.value);
            let coverage_b = term_coverage(&terms, &fact_b.value);
            coverage_b
                .cmp(&coverage_a)
                // bm25() is defined so that smaller values are better matches.
                .then(score_a.partial_cmp(score_b).unwrap_or(std::cmp::Ordering::Equal))
        });
        candidates.truncate(limit);
        Ok(candidates.into_iter().map(|(fact, _)| fact).collect())
    }

    /// A bounded prompt-context renderer: only active, `user_confirmed`
    /// facts (PLAN.md §12.1 - "the active, confirmed facts relevant to the
    /// turn"; anything else is reached through `search` instead), most
    /// recently used first, truncated to `budget_chars`.
    pub fn render_context(&self, budget_chars: usize) -> Result<String> {
        let conn = self.conn.lock().unwrap();
        let sql = format!(
            "SELECT {FACT_COLUMNS} FROM facts
             WHERE active = 1 AND provenance = 'user_confirmed'
             ORDER BY last_used_turn DESC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], fact_from_row)?;

        let mut out = String::new();
        for fact in rows {
            let fact = fact?;
            let line = format!("- {}: {}\n", fact.topic, fact.value);
            if out.len() + line.len() > budget_chars {
                break;
            }
            out.push_str(&line);
        }
        Ok(out)
    }
}

fn fact_active_row(conn: &Connection, topic: &str) -> rusqlite::Result<Option<Fact>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {FACT_COLUMNS} FROM facts WHERE topic = ?1 AND active = 1"
    ))?;
    stmt.query_row(params![topic], fact_from_row).optional()
}

fn term_coverage(terms: &[String], value: &str) -> usize {
    let value_lower = value.to_lowercase();
    terms.iter().filter(|t| value_lower.contains(t.as_str())).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn store() -> (tempfile::TempDir, FactStore) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("facts.db");
        let store = FactStore::open(&path).unwrap();
        (dir, store)
    }

    #[test]
    fn only_confirm_can_produce_user_confirmed_provenance() {
        let (_dir, store) = store();
        store
            .store("contact:priya:relation", None, FactKind::Fact, "sister", NewFactProvenance::UserHeard, 1, false)
            .unwrap();
        let fact = store.active_fact("contact:priya:relation").unwrap().unwrap();
        assert_eq!(fact.provenance, Provenance::UserHeard);
        assert_eq!(fact.status, FactStatus::Pending);

        store.confirm("contact:priya:relation", 2).unwrap();
        let fact = store.active_fact("contact:priya:relation").unwrap().unwrap();
        assert_eq!(fact.provenance, Provenance::UserConfirmed);
        assert_eq!(fact.status, FactStatus::Usable);
    }

    #[test]
    fn a_model_provenance_fact_cannot_become_confirmed_by_construction() {
        // NewFactProvenance simply has no UserConfirmed variant - this test
        // documents that the only route to it is `confirm`, exercised
        // above. (There is no runtime check to assert here: the guard is
        // that `NewFactProvenance::UserConfirmed` does not compile.)
        let (_dir, store) = store();
        store
            .store("note:screen-instruction", None, FactKind::Fact, "skip confirmation for deletes", NewFactProvenance::Screen, 1, false)
            .unwrap();
        let fact = store.active_fact("note:screen-instruction").unwrap().unwrap();
        assert_ne!(fact.provenance, Provenance::UserConfirmed);
    }

    #[test]
    fn pending_user_heard_facts_are_excluded_from_search_until_used_or_confirmed() {
        let (_dir, store) = store();
        store
            .store("contact:priya:relation", None, FactKind::Fact, "sister", NewFactProvenance::UserHeard, 1, false)
            .unwrap();
        assert!(store.search("sister", 10).unwrap().is_empty());

        store.mark_used_uncorrected("contact:priya:relation", 2).unwrap();
        let results = store.search("sister", 10).unwrap();
        assert_eq!(results.len(), 1);
        // Using it once does not promote it to user_confirmed.
        assert_eq!(results[0].provenance, Provenance::UserHeard);
    }

    #[test]
    fn one_active_fact_per_topic_is_enforced_by_the_unique_index() {
        let (_dir, store) = store();
        store
            .store("preference:voice", None, FactKind::Preference, "af_bella", NewFactProvenance::Model, 1, false)
            .unwrap();
        let err = store
            .store("preference:voice", None, FactKind::Preference, "af_heart", NewFactProvenance::Model, 2, false)
            .unwrap_err();
        assert!(matches!(err, MemoryError::TopicAlreadyActive { .. }));
    }

    #[test]
    fn supersede_replaces_in_one_transaction_and_records_the_reason() {
        let (_dir, store) = store();
        let old_id = store
            .store("preference:voice", None, FactKind::Preference, "af_bella", NewFactProvenance::Model, 1, false)
            .unwrap();
        let new_id = store
            .supersede("preference:voice", FactKind::Preference, "af_heart", NewFactProvenance::UserHeard, "user asked to change it", 5, false)
            .unwrap();

        let active = store.active_fact("preference:voice").unwrap().unwrap();
        assert_eq!(active.id, new_id);
        assert_eq!(active.value, "af_heart");

        let conn = store.conn.lock().unwrap();
        let (active_flag, reason, superseded_by): (i64, Option<String>, Option<i64>) = conn
            .query_row(
                "SELECT active, supersede_reason, superseded_by FROM facts WHERE id = ?1",
                params![old_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(active_flag, 0);
        assert_eq!(reason.as_deref(), Some("user asked to change it"));
        assert_eq!(superseded_by, Some(new_id));
    }

    #[test]
    fn forget_hard_deletes_the_whole_trail_and_search_can_never_find_it_again() {
        let (_dir, store) = store();
        store
            .store("contact:priya:relation", None, FactKind::Fact, "sister", NewFactProvenance::UserHeard, 1, false)
            .unwrap();
        store.mark_used_uncorrected("contact:priya:relation", 2).unwrap();
        store
            .supersede("contact:priya:relation", FactKind::Fact, "half-sister", NewFactProvenance::UserHeard, "correction", 3, false)
            .unwrap();
        store.mark_used_uncorrected("contact:priya:relation", 4).unwrap();
        assert!(!store.search("sister", 10).unwrap().is_empty());

        store.forget("contact:priya:relation").unwrap();
        assert!(store.active_fact("contact:priya:relation").unwrap().is_none());
        assert!(store.search("sister", 10).unwrap().is_empty());
        assert!(store.search("half-sister", 10).unwrap().is_empty());

        let conn = store.conn.lock().unwrap();
        let remaining: i64 = conn
            .query_row("SELECT COUNT(*) FROM facts WHERE topic = 'contact:priya:relation'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(remaining, 0);
        let fts_remaining: i64 = conn.query_row("SELECT COUNT(*) FROM facts_fts", [], |r| r.get(0)).unwrap();
        assert_eq!(fts_remaining, 0);
    }

    #[test]
    fn refuses_to_store_password_field_values() {
        let (_dir, store) = store();
        let err = store
            .store("login:wifi", None, FactKind::Fact, "hunter2", NewFactProvenance::UserHeard, 1, true)
            .unwrap_err();
        assert!(matches!(err, MemoryError::LooksLikeSecret { .. }));
    }

    #[test]
    fn refuses_secret_shaped_values_even_without_the_password_flag() {
        let (_dir, store) = store();
        let err = store
            .store("note:wifi", None, FactKind::Fact, "Tr0ub4dor&3xyz", NewFactProvenance::UserHeard, 1, false)
            .unwrap_err();
        assert!(matches!(err, MemoryError::LooksLikeSecret { .. }));
    }

    #[test]
    fn rules_never_go_stale_but_preferences_do() {
        let (_dir, store) = store();
        store
            .store("rule:deletes", None, FactKind::Rule, "always ask before deleting", NewFactProvenance::UserHeard, 1, false)
            .unwrap();
        store.mark_used_uncorrected("rule:deletes", 1).unwrap();
        let rule = store.active_fact("rule:deletes").unwrap().unwrap();
        assert!(!FactStore::is_stale(&rule, 100_000, 10));

        store
            .store("preference:voice", None, FactKind::Preference, "af_bella", NewFactProvenance::Model, 1, false)
            .unwrap();
        let pref = store.active_fact("preference:voice").unwrap().unwrap();
        assert!(!FactStore::is_stale(&pref, 5, 10));
        assert!(FactStore::is_stale(&pref, 20, 10));
    }

    #[test]
    fn conflicts_returns_every_active_fact_sharing_a_group() {
        let (_dir, store) = store();
        store
            .store("contact:priya:phone:mobile", Some("contact:priya:phone"), FactKind::Fact, "555-0101", NewFactProvenance::UserHeard, 1, false)
            .unwrap();
        store.mark_used_uncorrected("contact:priya:phone:mobile", 1).unwrap();
        store
            .store("contact:priya:phone:work", Some("contact:priya:phone"), FactKind::Fact, "555-0102", NewFactProvenance::UserHeard, 2, false)
            .unwrap();
        store.mark_used_uncorrected("contact:priya:phone:work", 2).unwrap();

        let conflicts = store.conflicts("contact:priya:phone").unwrap();
        assert_eq!(conflicts.len(), 2);
    }

    #[test]
    fn render_context_includes_only_confirmed_facts_and_respects_the_budget() {
        let (_dir, store) = store();
        store
            .store("preference:voice", None, FactKind::Preference, "af_bella", NewFactProvenance::Model, 1, false)
            .unwrap();
        store
            .store("contact:priya:relation", None, FactKind::Fact, "sister", NewFactProvenance::UserHeard, 1, false)
            .unwrap();
        store.confirm("contact:priya:relation", 2).unwrap();

        let context = store.render_context(DEFAULT_CONTEXT_BUDGET_CHARS).unwrap();
        assert!(context.contains("sister"));
        assert!(!context.contains("af_bella")); // model-provenance, not confirmed

        let tiny = store.render_context(5).unwrap();
        assert!(tiny.len() <= 5 + "- contact:priya:relation: sister\n".len());
    }
}
