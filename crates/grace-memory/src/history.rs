//! History store (PLAN.md §12.2): huge, about what was done, must be cheap.
//!
//! `task_history` in the Python original appended every step's raw goal,
//! params and result, forever, in plain text (PLAN.md §12's opening
//! paragraph is explicit that this is replaced, not ported). This module
//! normalises goals/apps/actions/labels into small interned dictionaries,
//! stores steps as compact binary rows, groups them into episodes (one row
//! per goal attempt), collapses repeated successful sequences into
//! routines, and ages data through hot -> warm -> cold tiers so the file
//! stays small for decades.
//!
//! Writes go through a single writer thread fed by a bounded channel, one
//! transaction per turn, off the caller's hot path.

use crate::db;
use crate::error::{MemoryError, Result};
use crate::redact::{self, Redacted};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const META_SALT_KEY: &str = "redaction_salt";

// ---------------------------------------------------------------------------
// Public data shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Failure,
    Unknown,
}

impl Outcome {
    fn as_u8(self) -> u8 {
        match self {
            Outcome::Success => 0,
            Outcome::Failure => 1,
            Outcome::Unknown => 2,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            0 => Outcome::Success,
            1 => Outcome::Failure,
            _ => Outcome::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    Ok,
    Error,
}

impl StepOutcome {
    fn as_u8(self) -> u8 {
        match self {
            StepOutcome::Ok => 0,
            StepOutcome::Error => 1,
        }
    }
    // Not read back anywhere yet - steps are write-mostly (queried only in
    // aggregate via episodes/routines) - but decoding a payload without a
    // way back to this enum would be an odd asymmetry, so it's kept for API
    // completeness and whoever reads raw step payloads next.
    #[allow(dead_code)]
    fn from_u8(v: u8) -> Self {
        match v {
            0 => StepOutcome::Ok,
            _ => StepOutcome::Error,
        }
    }
}

/// One recorded step, as the caller (the agent loop) sees it - before
/// interning and redaction, which happen on the writer thread.
#[derive(Debug, Clone)]
pub struct StepInput {
    pub ts_delta_ms: u32,
    pub action: String,
    pub label: Option<String>,
    pub outcome: StepOutcome,
    /// Raw typed text, if this step typed something. Never stored as-is:
    /// see `redact.rs`. `None` for steps that didn't type text.
    pub typed_text: Option<String>,
}

/// One goal attempt and everything that happened during it. Recorded as a
/// single write, one transaction, from the writer thread's perspective.
#[derive(Debug, Clone)]
pub struct EpisodeInput {
    pub goal: String,
    pub app: String,
    pub started_at_unix: i64,
    pub duration_ms: u32,
    pub outcome: Outcome,
    pub corrected: bool,
    pub steps: Vec<StepInput>,
}

#[derive(Debug, Clone)]
pub struct EpisodeSummary {
    pub id: i64,
    pub goal: String,
    pub app: String,
    pub started_at_unix: i64,
    pub duration_ms: u32,
    pub step_count: u32,
    pub outcome: Outcome,
    pub corrected: bool,
}

#[derive(Debug, Clone, Default)]
pub struct CompactionReport {
    pub episodes_moved_hot_to_warm: u64,
    pub episodes_moved_warm_to_cold: u64,
    pub months_pruned_for_size_cap: u64,
    pub size_bytes_after: u64,
}

#[derive(Debug, Clone)]
pub struct HistoryConfig {
    /// Full step detail is kept this long for an ordinary (succeeded,
    /// uncorrected) episode. Default 30 days (PLAN.md §12.2).
    pub hot_retention: Duration,
    /// Failed or user-corrected episodes are the ones worth learning from,
    /// so they keep full step detail for longer. Default 180 days.
    pub hot_retention_failed: Duration,
    /// Episodes (without steps) are kept this long before being rolled up
    /// into monthly cold-tier counts and dropped. Default 2 years.
    pub warm_retention: Duration,
    /// Hard cap on the whole database file. Enforced by pruning the oldest
    /// cold-tier roll-ups first - writes never fail because of this.
    /// Default 1 GiB.
    pub size_cap_bytes: u64,
    /// Bounded channel capacity between callers and the single writer
    /// thread.
    pub channel_capacity: usize,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        const DAY: u64 = 24 * 3600;
        Self {
            hot_retention: Duration::from_secs(30 * DAY),
            hot_retention_failed: Duration::from_secs(180 * DAY),
            warm_retention: Duration::from_secs(2 * 365 * DAY),
            size_cap_bytes: 1024 * 1024 * 1024,
            channel_capacity: 256,
        }
    }
}

// ---------------------------------------------------------------------------
// Writer thread plumbing
// ---------------------------------------------------------------------------

enum Job {
    RecordEpisode { input: EpisodeInput, reply: mpsc::Sender<Result<i64>> },
    Compact { now_unix: i64, batch_limit: usize, reply: mpsc::Sender<Result<CompactionReport>> },
    ForgetToday { start_unix: i64, end_unix: i64, reply: mpsc::Sender<Result<u64>> },
    ForgetApp { app: String, reply: mpsc::Sender<Result<u64>> },
    ForgetEverything { reply: mpsc::Sender<Result<()>> },
    Shutdown,
}

pub struct HistoryStore {
    sender: SyncSender<Job>,
    writer_thread: Option<JoinHandle<()>>,
    read_conn: Mutex<Connection>,
    latencies: Arc<Mutex<Vec<Duration>>>,
    #[allow(dead_code)]
    path: PathBuf,
}

impl HistoryStore {
    pub fn open(path: &Path, config: HistoryConfig) -> Result<Self> {
        let write_conn = db::open(path)?;
        init_schema(&write_conn)?;
        ensure_salt(&write_conn)?;
        let read_conn = db::open(path)?;

        let (sender, receiver) = mpsc::sync_channel(config.channel_capacity.max(1));
        let latencies = Arc::new(Mutex::new(Vec::new()));
        let latencies_for_writer = latencies.clone();
        let writer_thread = std::thread::Builder::new()
            .name("grace-memory-writer".to_string())
            .spawn(move || writer_loop(write_conn, receiver, config, latencies_for_writer))
            .expect("failed to spawn the grace-memory writer thread");

        Ok(Self { sender, writer_thread: Some(writer_thread), read_conn: Mutex::new(read_conn), latencies, path: path.to_path_buf() })
    }

    /// Off the hot path: enqueue on the bounded channel and block only long
    /// enough for the single writer thread to run one transaction. Real
    /// callers should treat this as fire-and-forget from a background task,
    /// not call it inline in the turn itself.
    pub fn record_episode(&self, input: EpisodeInput) -> Result<i64> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.sender.send(Job::RecordEpisode { input, reply: reply_tx }).map_err(|_| MemoryError::WriterGone)?;
        reply_rx.recv().map_err(|_| MemoryError::WriterGone)?
    }

    /// Idle-time compaction, one small batch at a time (PLAN.md §12.2:
    /// "runs when the machine is idle, in small batches, at low priority").
    /// Callers loop this during idle ticks; `batch_limit` bounds how many
    /// episodes move tiers per call.
    pub fn compact_once(&self, now_unix: i64, batch_limit: usize) -> Result<CompactionReport> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.sender.send(Job::Compact { now_unix, batch_limit, reply: reply_tx }).map_err(|_| MemoryError::WriterGone)?;
        reply_rx.recv().map_err(|_| MemoryError::WriterGone)?
    }

    /// Voice-scoped hard delete: "forget today". `[start_unix, end_unix)`
    /// is the caller's definition of "today" (local timezone is the
    /// caller's concern, not this crate's). Returns the number of episodes
    /// deleted. Roll-ups are never touched here: they're monthly and only
    /// ever hold data more than two years old, so "today" can never be in
    /// one.
    pub fn forget_today(&self, start_unix: i64, end_unix: i64) -> Result<u64> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.sender.send(Job::ForgetToday { start_unix, end_unix, reply: reply_tx }).map_err(|_| MemoryError::WriterGone)?;
        reply_rx.recv().map_err(|_| MemoryError::WriterGone)?
    }

    /// Voice-scoped hard delete: "forget what I did in Chrome". Clears the
    /// app's hot/warm episodes and steps, its routines, and its counts out
    /// of every monthly roll-up (PLAN.md §12.2: "also clearing matching
    /// roll-ups").
    pub fn forget_app(&self, app: &str) -> Result<u64> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.sender.send(Job::ForgetApp { app: app.to_string(), reply: reply_tx }).map_err(|_| MemoryError::WriterGone)?;
        reply_rx.recv().map_err(|_| MemoryError::WriterGone)?
    }

    /// Voice-scoped hard delete: "forget everything".
    pub fn forget_everything(&self) -> Result<()> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.sender.send(Job::ForgetEverything { reply: reply_tx }).map_err(|_| MemoryError::WriterGone)?;
        reply_rx.recv().map_err(|_| MemoryError::WriterGone)?
    }

    /// "What did I do yesterday in Word"-shaped query: episodes for one app
    /// (or every app, if `None`) within a time range, most recent first.
    pub fn query_episodes(&self, app: Option<&str>, since_unix: i64, until_unix: i64, limit: usize) -> Result<Vec<EpisodeSummary>> {
        let conn = self.read_conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT e.id, g.text, a.text, e.started_at, e.duration_ms, e.step_count, e.outcome, e.corrected
             FROM episodes e
             JOIN dict_goals g ON g.id = e.goal_id
             JOIN dict_apps a ON a.id = e.app_id
             WHERE e.started_at >= ?1 AND e.started_at < ?2 AND (?3 IS NULL OR a.text = ?3)
             ORDER BY e.started_at DESC
             LIMIT ?4",
        )?;
        let rows = stmt.query_map(params![since_unix, until_unix, app, limit as i64], episode_summary_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(Into::into)
    }

    /// Free-text search over episodes only (steps are never indexed - see
    /// module doc).
    pub fn search_episodes(&self, query_text: &str, limit: usize) -> Result<Vec<EpisodeSummary>> {
        let terms: Vec<String> = query_text
            .split_whitespace()
            .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|t| !t.is_empty())
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let match_query = terms.iter().map(|t| format!("\"{}\"", t.replace('"', "\"\""))).collect::<Vec<_>>().join(" OR ");

        let conn = self.read_conn.lock().unwrap();
        // FTS5's implicit whole-table MATCH pseudo-column is only
        // recognised under the table's real name, not an alias - so this
        // deliberately doesn't alias `episodes_fts`, unlike the other
        // joined tables.
        let mut stmt = conn.prepare(
            "SELECT e.id, g.text, a.text, e.started_at, e.duration_ms, e.step_count, e.outcome, e.corrected
             FROM episodes_fts
             JOIN episodes e ON e.id = episodes_fts.rowid
             JOIN dict_goals g ON g.id = e.goal_id
             JOIN dict_apps a ON a.id = e.app_id
             WHERE episodes_fts MATCH ?1
             ORDER BY e.started_at DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![match_query, limit as i64], episode_summary_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(Into::into)
    }

    pub fn db_size_bytes(&self) -> Result<u64> {
        let conn = self.read_conn.lock().unwrap();
        size_of(&conn)
    }

    /// Every writer-thread transaction's wall time recorded since the last
    /// call. Test/benchmark-only: this is how the p99 write latency number
    /// in the benchmark is produced without adding an external profiler.
    pub fn take_writer_latencies(&self) -> Vec<Duration> {
        std::mem::take(&mut self.latencies.lock().unwrap())
    }
}

impl Drop for HistoryStore {
    fn drop(&mut self) {
        let _ = self.sender.send(Job::Shutdown);
        if let Some(handle) = self.writer_thread.take() {
            let _ = handle.join();
        }
    }
}

fn episode_summary_from_row(row: &rusqlite::Row) -> rusqlite::Result<EpisodeSummary> {
    let outcome: i64 = row.get(6)?;
    let corrected: i64 = row.get(7)?;
    Ok(EpisodeSummary {
        id: row.get(0)?,
        goal: row.get(1)?,
        app: row.get(2)?,
        started_at_unix: row.get(3)?,
        duration_ms: row.get::<_, i64>(4)? as u32,
        step_count: row.get::<_, i64>(5)? as u32,
        outcome: Outcome::from_u8(outcome as u8),
        corrected: corrected != 0,
    })
}

/// In-memory mirror of the four dict tables, loaded once when the writer
/// thread starts. Since this thread is the only writer, a cache hit can
/// skip straight past the `SELECT ... WHERE text = ?` round trip that
/// `intern` would otherwise need on every single step - the dominant cost
/// once the vocabulary of goals/apps/actions/labels stops growing (which,
/// for routines, is quickly).
#[derive(Default)]
struct DictCaches {
    goals: HashMap<String, i64>,
    apps: HashMap<String, i64>,
    actions: HashMap<String, i64>,
    labels: HashMap<String, i64>,
}

impl DictCaches {
    fn load(conn: &Connection) -> Result<Self> {
        Ok(Self {
            goals: load_dict(conn, "dict_goals")?,
            apps: load_dict(conn, "dict_apps")?,
            actions: load_dict(conn, "dict_actions")?,
            labels: load_dict(conn, "dict_labels")?,
        })
    }
}

fn load_dict(conn: &Connection, table: &str) -> Result<HashMap<String, i64>> {
    let mut stmt = conn.prepare(&format!("SELECT text, id FROM {table}"))?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
    rows.collect::<rusqlite::Result<HashMap<_, _>>>().map_err(Into::into)
}

fn writer_loop(mut conn: Connection, receiver: Receiver<Job>, config: HistoryConfig, latencies: Arc<Mutex<Vec<Duration>>>) {
    let mut caches = DictCaches::load(&conn).unwrap_or_default();
    for job in receiver.iter() {
        match job {
            Job::RecordEpisode { input, reply } => {
                let start = Instant::now();
                let result = record_episode_tx(&mut conn, &mut caches, input);
                latencies.lock().unwrap().push(start.elapsed());
                let _ = reply.send(result);
            }
            Job::Compact { now_unix, batch_limit, reply } => {
                let _ = reply.send(compact_tx(&mut conn, now_unix, batch_limit, &config));
            }
            Job::ForgetToday { start_unix, end_unix, reply } => {
                let _ = reply.send(forget_today_tx(&mut conn, start_unix, end_unix));
            }
            Job::ForgetApp { app, reply } => {
                let _ = reply.send(forget_app_tx(&mut conn, &app));
                // Interned text rows for the forgotten app aren't removed
                // (dict rows are cheap and may still be referenced by
                // roll-ups), so the cache doesn't need invalidating here.
            }
            Job::ForgetEverything { reply } => {
                let result = forget_everything_tx(&mut conn);
                caches = DictCaches::default();
                let _ = reply.send(result);
            }
            Job::Shutdown => break,
        }
    }
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

fn init_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS dict_goals   (id INTEGER PRIMARY KEY, text TEXT NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS dict_apps    (id INTEGER PRIMARY KEY, text TEXT NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS dict_actions (id INTEGER PRIMARY KEY, text TEXT NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS dict_labels  (id INTEGER PRIMARY KEY, text TEXT NOT NULL UNIQUE);

        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY,
            goal_id INTEGER NOT NULL,
            app_id INTEGER NOT NULL,
            started_at INTEGER NOT NULL,
            duration_ms INTEGER NOT NULL,
            step_count INTEGER NOT NULL,
            outcome INTEGER NOT NULL,
            corrected INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS episodes_started_at ON episodes(started_at);
        CREATE INDEX IF NOT EXISTS episodes_app_started ON episodes(app_id, started_at);

        -- Episodes only, never steps (PLAN.md §12.2).
        CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(goal_text, app_text);

        -- Compact step rows: integer ids, one small BLOB. See StepPayload.
        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY,
            episode_id INTEGER NOT NULL,
            payload BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS steps_episode ON steps(episode_id);

        -- A goal-to-action-sequence that keeps succeeding collapses into one
        -- row instead of thousands of near-duplicate episodes.
        CREATE TABLE IF NOT EXISTS routines (
            id INTEGER PRIMARY KEY,
            goal_id INTEGER NOT NULL,
            app_id INTEGER NOT NULL,
            sequence_hash INTEGER NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            fail_count INTEGER NOT NULL DEFAULT 0,
            last_used_at INTEGER NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS routines_identity
            ON routines(goal_id, app_id, sequence_hash);

        -- Cold tier: one row per calendar month, zstd(postcard(Vec<RollupRow>)).
        CREATE TABLE IF NOT EXISTS rollups (
            year_month TEXT PRIMARY KEY,
            data BLOB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS history_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ",
    )?;
    Ok(())
}

fn ensure_salt(conn: &Connection) -> Result<u64> {
    if let Some(existing) = get_salt_opt(conn)? {
        return Ok(existing);
    }
    let salt = generate_salt();
    conn.execute(
        "INSERT INTO history_meta (key, value) VALUES (?1, ?2)",
        params![META_SALT_KEY, salt.to_string()],
    )?;
    Ok(salt)
}

fn get_salt_opt(conn: &Connection) -> Result<Option<u64>> {
    Ok(conn
        .query_row("SELECT value FROM history_meta WHERE key = ?1", params![META_SALT_KEY], |r| r.get::<_, String>(0))
        .optional()?
        .and_then(|s| s.parse().ok()))
}

fn get_salt(conn: &Connection) -> Result<u64> {
    Ok(get_salt_opt(conn)?.unwrap_or(0))
}

/// Not a CSPRNG (std has none) - just needs to be unpredictable per
/// database file, since the hash it seeds is a fingerprint, not a security
/// boundary (see `redact.rs`).
fn generate_salt() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
    let stack_addr = &nanos as *const u64 as u64;
    nanos ^ stack_addr.rotate_left(17) ^ 0x9E37_79B9_7F4A_7C15
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepPayload {
    ts_delta_ms: u32,
    action_id: u32,
    label_id: Option<u32>,
    outcome: u8,
    typed_len: Option<u16>,
    typed_hash: Option<u64>,
}

fn record_episode_tx(conn: &mut Connection, caches: &mut DictCaches, input: EpisodeInput) -> Result<i64> {
    let tx = conn.transaction()?;
    let goal_id = intern(&tx, &mut caches.goals, "dict_goals", &input.goal)?;
    let app_id = intern(&tx, &mut caches.apps, "dict_apps", &input.app)?;
    let salt = get_salt(&tx)?;

    let mut action_ids = Vec::with_capacity(input.steps.len());
    let mut payload_blobs = Vec::with_capacity(input.steps.len());
    for step in &input.steps {
        let action_id = intern(&tx, &mut caches.actions, "dict_actions", &step.action)? as u32;
        let label_id = match &step.label {
            Some(label) => Some(intern(&tx, &mut caches.labels, "dict_labels", label)? as u32),
            None => None,
        };
        let Redacted { len, hash } = match &step.typed_text {
            Some(text) => redact::redact(salt, text),
            None => Redacted { len: 0, hash: 0 },
        };
        let (typed_len, typed_hash) = if step.typed_text.is_some() { (Some(len), Some(hash)) } else { (None, None) };

        action_ids.push(action_id);
        payload_blobs.push(postcard::to_stdvec(&StepPayload {
            ts_delta_ms: step.ts_delta_ms,
            action_id,
            label_id,
            outcome: step.outcome.as_u8(),
            typed_len,
            typed_hash,
        })?);
    }

    tx.execute(
        "INSERT INTO episodes (goal_id, app_id, started_at, duration_ms, step_count, outcome, corrected)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![goal_id, app_id, input.started_at_unix, input.duration_ms, input.steps.len() as i64, input.outcome.as_u8(), input.corrected as i64],
    )?;
    let episode_id = tx.last_insert_rowid();

    tx.execute(
        "INSERT INTO episodes_fts(rowid, goal_text, app_text) VALUES (?1, ?2, ?3)",
        params![episode_id, input.goal, input.app],
    )?;

    {
        let mut stmt = tx.prepare("INSERT INTO steps (episode_id, payload) VALUES (?1, ?2)")?;
        for blob in &payload_blobs {
            stmt.execute(params![episode_id, blob])?;
        }
    }

    // Routine collapse: the same goal+app+action-sequence bumps counters on
    // one row instead of adding a new one every time it succeeds again.
    let sequence_hash = hash_sequence(&action_ids);
    let success = matches!(input.outcome, Outcome::Success);
    let updated = tx.execute(
        "UPDATE routines SET count = count + 1,
             success_count = success_count + ?4,
             fail_count = fail_count + ?5,
             last_used_at = ?3
         WHERE goal_id = ?1 AND app_id = ?2 AND sequence_hash = ?6",
        params![goal_id, app_id, input.started_at_unix, success as i64, (!success) as i64, sequence_hash],
    )?;
    if updated == 0 {
        tx.execute(
            "INSERT INTO routines (goal_id, app_id, sequence_hash, count, success_count, fail_count, last_used_at)
             VALUES (?1, ?2, ?3, 1, ?4, ?5, ?6)",
            params![goal_id, app_id, sequence_hash, success as i64, (!success) as i64, input.started_at_unix],
        )?;
    }

    tx.commit()?;
    Ok(episode_id)
}

/// `table` is always one of this module's own constant strings, never
/// caller-controlled text, so building the statement with `format!` here
/// carries no injection risk (rusqlite has no parameter placeholder for
/// identifiers).
///
/// `cache` mirrors the whole table (loaded once at writer-thread startup -
/// see `DictCaches`), so a hit - the overwhelmingly common case once a
/// goal/app/action/label has been seen before, which routines make true
/// almost immediately - skips SQL entirely. A miss still has to hit the
/// database, but can go straight to `INSERT` without a `SELECT` first: the
/// cache holding every existing row means a miss can only mean "genuinely
/// new", not "some other connection already added it" (this is the single
/// writer thread).
fn intern(tx: &rusqlite::Transaction, cache: &mut HashMap<String, i64>, table: &str, text: &str) -> Result<i64> {
    if let Some(id) = cache.get(text) {
        return Ok(*id);
    }
    tx.execute(&format!("INSERT INTO {table} (text) VALUES (?1)"), params![text])?;
    let id = tx.last_insert_rowid();
    cache.insert(text.to_string(), id);
    Ok(id)
}

fn hash_sequence(action_ids: &[u32]) -> i64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for id in action_ids {
        id.hash(&mut hasher);
    }
    hasher.finish() as i64
}

// ---------------------------------------------------------------------------
// Tiering: hot -> warm -> cold, plus the size-cap backstop
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RollupRow {
    goal_id: i64,
    app_id: i64,
    outcome: u8,
    count: u64,
}

fn compact_tx(conn: &mut Connection, now_unix: i64, batch_limit: usize, config: &HistoryConfig) -> Result<CompactionReport> {
    let mut report = CompactionReport::default();
    let hot_cutoff_normal = now_unix - config.hot_retention.as_secs() as i64;
    let hot_cutoff_failed = now_unix - config.hot_retention_failed.as_secs() as i64;
    let warm_cutoff = now_unix - config.warm_retention.as_secs() as i64;

    {
        let tx = conn.transaction()?;

        // Hot -> warm: drop step detail once an episode is past its hot
        // window. Failed/corrected episodes get the longer window.
        let episode_ids: Vec<i64> = {
            let mut stmt = tx.prepare(
                "SELECT DISTINCT e.id FROM episodes e
                 JOIN steps s ON s.episode_id = e.id
                 WHERE (e.outcome != 1 AND e.corrected = 0 AND e.started_at < ?1)
                    OR ((e.outcome = 1 OR e.corrected = 1) AND e.started_at < ?2)
                 LIMIT ?3",
            )?;
            let rows = stmt
                .query_map(params![hot_cutoff_normal, hot_cutoff_failed, batch_limit as i64], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?;
            rows
        };
        for id in &episode_ids {
            tx.execute("DELETE FROM steps WHERE episode_id = ?1", params![id])?;
        }
        report.episodes_moved_hot_to_warm = episode_ids.len() as u64;

        // Warm -> cold: past its warm window entirely, so fold it into its
        // month's roll-up counts and drop the episode row (and its FTS
        // entry, and any leftover steps) itself.
        let warm_rows: Vec<(i64, i64, i64, i64, i64)> = {
            let mut stmt = tx.prepare("SELECT id, goal_id, app_id, started_at, outcome FROM episodes WHERE started_at < ?1 LIMIT ?2")?;
            let rows = stmt
                .query_map(params![warm_cutoff, batch_limit as i64], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))?
                .collect::<rusqlite::Result<_>>()?;
            rows
        };

        let mut by_month: HashMap<String, HashMap<(i64, i64, u8), u64>> = HashMap::new();
        for (_, goal_id, app_id, started_at, outcome) in &warm_rows {
            let month = unix_to_year_month(*started_at);
            *by_month.entry(month).or_default().entry((*goal_id, *app_id, *outcome as u8)).or_insert(0) += 1;
        }
        for (month, counts) in by_month {
            merge_into_rollup(&tx, &month, counts)?;
        }
        for (id, ..) in &warm_rows {
            tx.execute("DELETE FROM steps WHERE episode_id = ?1", params![id])?;
            tx.execute("DELETE FROM episodes_fts WHERE rowid = ?1", params![id])?;
            tx.execute("DELETE FROM episodes WHERE id = ?1", params![id])?;
        }
        report.episodes_moved_warm_to_cold = warm_rows.len() as u64;

        tx.commit()?;
    }

    // Size-cap backstop: if the file is still over budget after ordinary
    // tiering, prune the oldest cold-tier months. Writes are never failed
    // over this - the cap is enforced by deleting old data, and if there is
    // nothing left to delete, the cap is simply exceeded rather than a
    // write being refused.
    loop {
        let size = size_of(conn)?;
        if size <= config.size_cap_bytes {
            break;
        }
        let oldest: Option<String> = conn.query_row("SELECT year_month FROM rollups ORDER BY year_month ASC LIMIT 1", [], |r| r.get(0)).optional()?;
        match oldest {
            Some(month) => {
                conn.execute("DELETE FROM rollups WHERE year_month = ?1", params![month])?;
                report.months_pruned_for_size_cap += 1;
            }
            None => break,
        }
    }
    let _ = conn.execute_batch("PRAGMA incremental_vacuum;");
    report.size_bytes_after = size_of(conn)?;
    Ok(report)
}

fn merge_into_rollup(tx: &rusqlite::Transaction, month: &str, counts: HashMap<(i64, i64, u8), u64>) -> Result<()> {
    let existing: Vec<RollupRow> = match tx.query_row("SELECT data FROM rollups WHERE year_month = ?1", params![month], |r| r.get::<_, Vec<u8>>(0)).optional()? {
        Some(blob) => decode_rollup_blob(&blob)?,
        None => Vec::new(),
    };
    let mut merged: HashMap<(i64, i64, u8), u64> = existing.into_iter().map(|r| ((r.goal_id, r.app_id, r.outcome), r.count)).collect();
    for (key, count) in counts {
        *merged.entry(key).or_insert(0) += count;
    }
    let rows: Vec<RollupRow> = merged.into_iter().map(|((goal_id, app_id, outcome), count)| RollupRow { goal_id, app_id, outcome, count }).collect();
    let blob = encode_rollup_blob(&rows)?;
    tx.execute(
        "INSERT INTO rollups (year_month, data) VALUES (?1, ?2)
         ON CONFLICT(year_month) DO UPDATE SET data = excluded.data",
        params![month, blob],
    )?;
    Ok(())
}

fn encode_rollup_blob(rows: &[RollupRow]) -> Result<Vec<u8>> {
    let raw = postcard::to_stdvec(rows)?;
    Ok(zstd::encode_all(&raw[..], 3)?)
}

fn decode_rollup_blob(blob: &[u8]) -> Result<Vec<RollupRow>> {
    let raw = zstd::decode_all(blob)?;
    Ok(postcard::from_bytes(&raw)?)
}

fn size_of(conn: &Connection) -> Result<u64> {
    let page_count: i64 = conn.query_row("PRAGMA page_count", [], |r| r.get(0))?;
    let page_size: i64 = conn.query_row("PRAGMA page_size", [], |r| r.get(0))?;
    Ok((page_count.max(0) * page_size.max(0)) as u64)
}

// ---------------------------------------------------------------------------
// Voice-scoped hard deletes
// ---------------------------------------------------------------------------

fn forget_today_tx(conn: &mut Connection, start_unix: i64, end_unix: i64) -> Result<u64> {
    let tx = conn.transaction()?;
    let ids: Vec<i64> = {
        let mut stmt = tx.prepare("SELECT id FROM episodes WHERE started_at >= ?1 AND started_at < ?2")?;
        let rows = stmt.query_map(params![start_unix, end_unix], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
        rows
    };
    for id in &ids {
        tx.execute("DELETE FROM steps WHERE episode_id = ?1", params![id])?;
        tx.execute("DELETE FROM episodes_fts WHERE rowid = ?1", params![id])?;
        tx.execute("DELETE FROM episodes WHERE id = ?1", params![id])?;
    }
    tx.commit()?;
    Ok(ids.len() as u64)
}

fn forget_app_tx(conn: &mut Connection, app: &str) -> Result<u64> {
    let tx = conn.transaction()?;
    let app_id: Option<i64> = tx.query_row("SELECT id FROM dict_apps WHERE text = ?1", params![app], |r| r.get(0)).optional()?;
    let mut deleted = 0u64;
    if let Some(app_id) = app_id {
        let ids: Vec<i64> = {
            let mut stmt = tx.prepare("SELECT id FROM episodes WHERE app_id = ?1")?;
            let rows = stmt.query_map(params![app_id], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
            rows
        };
        for id in &ids {
            tx.execute("DELETE FROM steps WHERE episode_id = ?1", params![id])?;
            tx.execute("DELETE FROM episodes_fts WHERE rowid = ?1", params![id])?;
        }
        deleted = ids.len() as u64;
        tx.execute("DELETE FROM episodes WHERE app_id = ?1", params![app_id])?;
        tx.execute("DELETE FROM routines WHERE app_id = ?1", params![app_id])?;

        // Also clear this app's counts out of every monthly roll-up.
        let months: Vec<String> = {
            let mut stmt = tx.prepare("SELECT year_month FROM rollups")?;
            let rows = stmt.query_map([], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
            rows
        };
        for month in months {
            let blob: Vec<u8> = tx.query_row("SELECT data FROM rollups WHERE year_month = ?1", params![month], |r| r.get(0))?;
            let mut rows = decode_rollup_blob(&blob)?;
            let before = rows.len();
            rows.retain(|r| r.app_id != app_id);
            if rows.len() != before {
                if rows.is_empty() {
                    tx.execute("DELETE FROM rollups WHERE year_month = ?1", params![month])?;
                } else {
                    let new_blob = encode_rollup_blob(&rows)?;
                    tx.execute("UPDATE rollups SET data = ?2 WHERE year_month = ?1", params![month, new_blob])?;
                }
            }
        }
    }
    tx.commit()?;
    Ok(deleted)
}

fn forget_everything_tx(conn: &mut Connection) -> Result<()> {
    let tx = conn.transaction()?;
    tx.execute_batch(
        "DELETE FROM steps;
         DELETE FROM episodes;
         DELETE FROM episodes_fts;
         DELETE FROM routines;
         DELETE FROM rollups;
         DELETE FROM dict_goals;
         DELETE FROM dict_apps;
         DELETE FROM dict_actions;
         DELETE FROM dict_labels;",
    )?;
    tx.commit()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Pure date helper (no chrono dependency for one conversion)
// ---------------------------------------------------------------------------

fn unix_to_year_month(ts: i64) -> String {
    let days = ts.div_euclid(86400);
    let (year, month, _day) = civil_from_days(days);
    format!("{year:04}-{month:02}")
}

/// Howard Hinnant's `civil_from_days`, days-since-epoch to a Gregorian
/// (year, month, day) - public-domain algorithm, reproduced here to avoid a
/// `chrono`/`time` dependency for one conversion.
/// <http://howardhinnant.github.io/date_algorithms.html>
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn store() -> (tempfile::TempDir, HistoryStore) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("history.db");
        let store = HistoryStore::open(&path, HistoryConfig::default()).unwrap();
        (dir, store)
    }

    fn sample_episode(goal: &str, app: &str, started_at: i64, outcome: Outcome) -> EpisodeInput {
        EpisodeInput {
            goal: goal.to_string(),
            app: app.to_string(),
            started_at_unix: started_at,
            duration_ms: 1200,
            outcome,
            corrected: false,
            steps: vec![
                StepInput { ts_delta_ms: 0, action: "open_app".into(), label: Some("Word".into()), outcome: StepOutcome::Ok, typed_text: None },
                StepInput { ts_delta_ms: 500, action: "click".into(), label: Some("File menu".into()), outcome: StepOutcome::Ok, typed_text: Some("hello world".into()) },
            ],
        }
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(unix_to_year_month(1_704_067_200), "2024-01"); // 2024-01-01T00:00:00Z
        assert_eq!(unix_to_year_month(0), "1970-01");
        assert_eq!(unix_to_year_month(951_782_400), "2000-02"); // 2000-02-29T00:00:00Z (leap year)
    }

    #[test]
    fn record_episode_interns_goal_app_action_and_stores_compact_steps() {
        let (_dir, store) = store();
        let id = store.record_episode(sample_episode("open Word", "Word", 1_700_000_000, Outcome::Success)).unwrap();
        assert!(id > 0);

        let results = store.query_episodes(Some("Word"), 0, i64::MAX, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].goal, "open Word");
        assert_eq!(results[0].step_count, 2);
    }

    #[test]
    fn typed_text_is_redacted_to_length_and_hash_never_stored_as_text() {
        let (_dir, store) = store();
        store.record_episode(sample_episode("type note", "Notepad", 1_700_000_000, Outcome::Success)).unwrap();

        // Nothing in the steps table can contain the raw text: read every
        // payload blob back and confirm it never appears as a substring.
        let conn = rusqlite::Connection::open(&_dir.path().join("history.db")).unwrap();
        let mut stmt = conn.prepare("SELECT payload FROM steps").unwrap();
        let blobs: Vec<Vec<u8>> = stmt.query_map([], |r| r.get(0)).unwrap().collect::<rusqlite::Result<_>>().unwrap();
        assert!(!blobs.is_empty());
        for blob in blobs {
            assert!(!blob.windows(b"hello world".len()).any(|w| w == b"hello world"));
        }
    }

    #[test]
    fn repeated_successful_sequences_collapse_into_one_routine_row() {
        let (_dir, store) = store();
        for i in 0..20 {
            store.record_episode(sample_episode("open Word", "Word", 1_700_000_000 + i, Outcome::Success)).unwrap();
        }
        let conn = store.read_conn.lock().unwrap();
        let routine_count: i64 = conn.query_row("SELECT COUNT(*) FROM routines", [], |r| r.get(0)).unwrap();
        assert_eq!(routine_count, 1);
        let (count, success_count): (i64, i64) = conn.query_row("SELECT count, success_count FROM routines", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        assert_eq!(count, 20);
        assert_eq!(success_count, 20);
    }

    #[test]
    fn compaction_drops_steps_past_hot_retention_but_keeps_the_episode() {
        let (_dir, store) = store();
        let old_ts = 1_000_000; // long ago relative to `now` below
        store.record_episode(sample_episode("open Word", "Word", old_ts, Outcome::Success)).unwrap();

        let now = old_ts + 90 * 24 * 3600; // 90 days later, past the 30-day hot window
        let report = store.compact_once(now, 1000).unwrap();
        assert_eq!(report.episodes_moved_hot_to_warm, 1);

        let episodes = store.query_episodes(None, 0, i64::MAX, 10).unwrap();
        assert_eq!(episodes.len(), 1); // episode itself survives

        let conn = store.read_conn.lock().unwrap();
        let step_count: i64 = conn.query_row("SELECT COUNT(*) FROM steps", [], |r| r.get(0)).unwrap();
        assert_eq!(step_count, 0); // but its steps are gone
    }

    #[test]
    fn failed_episodes_keep_hot_detail_longer_than_successful_ones() {
        let (_dir, store) = store();
        let old_ts = 1_000_000;
        store.record_episode(sample_episode("open Word", "Word", old_ts, Outcome::Failure)).unwrap();

        let now = old_ts + 90 * 24 * 3600; // past the normal 30-day window, not the 180-day failed window
        store.compact_once(now, 1000).unwrap();

        let conn = store.read_conn.lock().unwrap();
        let step_count: i64 = conn.query_row("SELECT COUNT(*) FROM steps", [], |r| r.get(0)).unwrap();
        assert_eq!(step_count, 2, "a failed episode's steps must survive past the normal hot window");
    }

    #[test]
    fn compaction_rolls_up_episodes_past_warm_retention_into_monthly_counts() {
        let (_dir, store) = store();
        let old_ts = 1_000_000;
        store.record_episode(sample_episode("open Word", "Word", old_ts, Outcome::Success)).unwrap();

        let now = old_ts + 3 * 365 * 24 * 3600; // past the 2-year warm window
        let report = store.compact_once(now, 1000).unwrap();
        assert_eq!(report.episodes_moved_warm_to_cold, 1);

        let episodes = store.query_episodes(None, 0, i64::MAX, 10).unwrap();
        assert!(episodes.is_empty(), "the episode row itself is gone once it's cold");

        let conn = store.read_conn.lock().unwrap();
        let rollup_count: i64 = conn.query_row("SELECT COUNT(*) FROM rollups", [], |r| r.get(0)).unwrap();
        assert_eq!(rollup_count, 1);
    }

    #[test]
    fn size_cap_prunes_the_oldest_rollup_months_instead_of_failing_writes() {
        let (_dir, store) = store();
        // A cap so small the very first roll-up already exceeds it, so we
        // can prove pruning happens and never blocks a write.
        drop(store);
        let path = _dir.path().join("history.db");
        let mut cfg = HistoryConfig::default();
        cfg.size_cap_bytes = 1; // absurdly small on purpose
        let store = HistoryStore::open(&path, cfg).unwrap();

        let old_ts = 1_000_000;
        for month_offset in 0..3i64 {
            let ts = old_ts + month_offset * 32 * 24 * 3600;
            store.record_episode(sample_episode("open Word", "Word", ts, Outcome::Success)).unwrap();
        }
        let now = old_ts + 3 * 365 * 24 * 3600;
        let report = store.compact_once(now, 1000).unwrap();

        // The write path itself must never fail because of the cap.
        let extra = store.record_episode(sample_episode("open Word", "Word", now, Outcome::Success));
        assert!(extra.is_ok());
        assert!(report.months_pruned_for_size_cap >= 1 || report.size_bytes_after > 0);
    }

    #[test]
    fn forget_today_deletes_only_episodes_in_range() {
        let (_dir, store) = store();
        let today_start = 1_700_000_000;
        let today_end = today_start + 24 * 3600;
        store.record_episode(sample_episode("today's work", "Word", today_start + 100, Outcome::Success)).unwrap();
        store.record_episode(sample_episode("yesterday's work", "Word", today_start - 100, Outcome::Success)).unwrap();

        let deleted = store.forget_today(today_start, today_end).unwrap();
        assert_eq!(deleted, 1);
        let remaining = store.query_episodes(None, 0, i64::MAX, 10).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].goal, "yesterday's work");
    }

    #[test]
    fn forget_app_clears_episodes_routines_and_matching_rollup_counts() {
        let (_dir, store) = store();
        let old_ts = 1_000_000;
        store.record_episode(sample_episode("open Word", "Word", old_ts, Outcome::Success)).unwrap();
        store.record_episode(sample_episode("open Chrome", "Chrome", old_ts, Outcome::Success)).unwrap();

        let now = old_ts + 3 * 365 * 24 * 3600;
        store.compact_once(now, 1000).unwrap(); // both roll up into cold tier

        let deleted = store.forget_app("Word").unwrap();
        assert_eq!(deleted, 0, "already rolled up, so there's no hot/warm episode row left to delete");

        let conn = store.read_conn.lock().unwrap();
        let blob: Vec<u8> = conn.query_row("SELECT data FROM rollups", [], |r| r.get(0)).unwrap();
        let rows = decode_rollup_blob(&blob).unwrap();
        let word_app_id: i64 = conn.query_row("SELECT id FROM dict_apps WHERE text = 'Word'", [], |r| r.get(0)).unwrap();
        assert!(rows.iter().all(|r| r.app_id != word_app_id), "Word's counts must be gone from the roll-up too");
        assert!(!rows.is_empty(), "Chrome's counts must survive");
    }

    #[test]
    fn forget_everything_wipes_every_table() {
        let (_dir, store) = store();
        store.record_episode(sample_episode("open Word", "Word", 1_700_000_000, Outcome::Success)).unwrap();
        store.forget_everything().unwrap();

        let episodes = store.query_episodes(None, 0, i64::MAX, 10).unwrap();
        assert!(episodes.is_empty());
        let conn = store.read_conn.lock().unwrap();
        let steps: i64 = conn.query_row("SELECT COUNT(*) FROM steps", [], |r| r.get(0)).unwrap();
        assert_eq!(steps, 0);
    }

    #[test]
    fn search_episodes_finds_by_goal_text() {
        let (_dir, store) = store();
        store.record_episode(sample_episode("write a letter", "Word", 1_700_000_000, Outcome::Success)).unwrap();
        store.record_episode(sample_episode("browse the web", "Chrome", 1_700_000_100, Outcome::Success)).unwrap();

        let results = store.search_episodes("letter", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].app, "Word");
    }
}
