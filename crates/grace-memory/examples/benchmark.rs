//! Synthetic 10-year benchmark for the history store (PLAN.md §12.2:
//! "Proved by a benchmark, not assumed"). Simulates roughly 2,000 steps a
//! day of heavy use, with realistic repetition (most episodes replay one of
//! a small set of routines, e.g. "open Word", so they collapse in the
//! `routines` table; a minority are one-off/novel goals that keep the
//! `episodes` table's warm tier honest). Idle-time compaction runs once per
//! simulated day, as the real background tick would.
//!
//! Reports, per PLAN.md §12.2's own benchmark requirement:
//! - file size after each simulated year;
//! - p99 write latency (the caller's round trip through `record_episode` -
//!   channel send, one writer-thread transaction, reply - since that is
//!   what actually blocks a caller that doesn't fire-and-forget);
//! - the latency of a "what did I do yesterday in Word"-shaped query.
//!
//! Run in release mode - a debug build's SQLite is dramatically slower and
//! not representative:
//! ```sh
//! cargo run -p grace-memory --release --example benchmark
//! ```
//! Writes only to a temp directory (this crate's hard rule: tests/examples
//! never touch `~/.grace`).

use grace_memory::history::{EpisodeInput, HistoryConfig, HistoryStore, Outcome, StepInput, StepOutcome};
use std::time::{Duration, Instant};

const SECONDS_PER_DAY: i64 = 24 * 3600;
const DAYS_PER_YEAR: i64 = 365;
const YEARS: i64 = 10;
const STEPS_PER_DAY_TARGET: usize = 2000;

struct Routine {
    goal: &'static str,
    app: &'static str,
    actions: &'static [&'static str],
}

/// A small, realistic set of everyday goals. 90% of generated episodes
/// reuse one of these verbatim (so their action sequence is identical every
/// time - what makes them collapse into one `routines` row); the remaining
/// 10% are synthesized one-off goals that never repeat, keeping the warm
/// tier non-trivial the way real occasional/unusual tasks would.
const ROUTINES: &[Routine] = &[
    Routine { goal: "open Word", app: "Word", actions: &["open_app", "click", "type_text"] },
    Routine { goal: "check email", app: "Outlook", actions: &["open_app", "click", "click", "read"] },
    Routine { goal: "browse the web", app: "Chrome", actions: &["open_app", "click", "scroll"] },
    Routine { goal: "open calculator", app: "Calculator", actions: &["open_app", "click"] },
    Routine { goal: "take notes", app: "Notepad", actions: &["open_app", "type_text"] },
    Routine { goal: "video call", app: "Teams", actions: &["open_app", "click", "click"] },
    Routine { goal: "browse files", app: "Explorer", actions: &["open_app", "click", "click"] },
    Routine { goal: "play music", app: "Spotify", actions: &["open_app", "click"] },
];

/// A small, deterministic xorshift PRNG - no extra dependency, and a fixed
/// seed makes the benchmark's output reproducible run to run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("benchmark_history.db");
    let store = HistoryStore::open(&path, HistoryConfig::default()).expect("open history store");

    let mut rng = Rng(0x1234_5678_9abc_def0);
    let start_unix: i64 = 1_700_000_000; // arbitrary anchor, not wall-clock "now"

    let mut write_latencies: Vec<Duration> = Vec::with_capacity(2_000_000);
    let mut ts = start_unix;
    let mut total_steps: u64 = 0;
    let mut total_episodes: u64 = 0;
    let overall_start = Instant::now();

    for year in 1..=YEARS {
        let year_start = Instant::now();
        for _day in 0..DAYS_PER_YEAR {
            let mut steps_today = 0usize;
            while steps_today < STEPS_PER_DAY_TARGET {
                let r = rng.next();
                let (goal, app, actions, outcome, corrected) = if r % 100 < 90 {
                    let routine = &ROUTINES[(r / 100) as usize % ROUTINES.len()];
                    let failed = r % 1000 < 20; // ~2% of routine runs fail
                    (routine.goal.to_string(), routine.app.to_string(), routine.actions.to_vec(), if failed { Outcome::Failure } else { Outcome::Success }, failed && r % 3 == 0)
                } else {
                    let n = 3 + (r % 5) as usize;
                    let actions: Vec<&str> = std::iter::repeat("novel_action").take(n).collect();
                    (format!("one-off task {}", r % 5_000_000), "SomeApp".to_string(), actions, Outcome::Unknown, false)
                };

                let steps: Vec<StepInput> = actions
                    .iter()
                    .enumerate()
                    .map(|(i, action)| StepInput {
                        ts_delta_ms: (i as u32) * 400,
                        action: action.to_string(),
                        label: Some(format!("{action} label")),
                        outcome: StepOutcome::Ok,
                        typed_text: if *action == "type_text" { Some("a realistic sentence the user typed".to_string()) } else { None },
                    })
                    .collect();
                steps_today += steps.len();
                total_steps += steps.len() as u64;
                total_episodes += 1;

                let episode = EpisodeInput { goal, app, started_at_unix: ts, duration_ms: 1200, outcome, corrected, steps };

                let t0 = Instant::now();
                store.record_episode(episode).expect("record_episode");
                write_latencies.push(t0.elapsed());

                ts += 1;
            }
            ts = start_unix + ((year - 1) * DAYS_PER_YEAR + _day + 1) * SECONDS_PER_DAY;
            // Idle-time compaction, once per simulated day - small batches,
            // as the real background tick would run it.
            store.compact_once(ts, 200_000).expect("compact_once");
        }

        let size = store.db_size_bytes().expect("db_size_bytes");
        println!(
            "year {year:>2}: db size = {size:>10} bytes ({:>7.2} MB)  [{:>6} episodes/day avg, generated in {:.1?}]",
            size as f64 / 1e6,
            total_episodes / (year as u64 * DAYS_PER_YEAR as u64),
            year_start.elapsed()
        );
    }

    println!();
    println!("total episodes: {total_episodes}, total steps: {total_steps}, wall time: {:.1?}", overall_start.elapsed());

    write_latencies.sort();
    let p50 = percentile(&write_latencies, 0.50);
    let p99 = percentile(&write_latencies, 0.99);
    let p999 = percentile(&write_latencies, 0.999);
    let max = *write_latencies.last().unwrap();
    println!();
    println!("write latency (record_episode round trip: channel send + one writer-thread transaction + reply):");
    println!("  p50 = {p50:?}, p99 = {p99:?}, p99.9 = {p999:?}, max = {max:?}");

    // "What did I do yesterday in Word" - the day immediately before the
    // benchmark's final simulated timestamp, filtered to one app.
    let yesterday_start = ts - SECONDS_PER_DAY;
    let query_start = Instant::now();
    let results = store.query_episodes(Some("Word"), yesterday_start, ts, 1000).expect("query_episodes");
    let query_latency = query_start.elapsed();
    println!();
    println!("\"what did I do yesterday in Word\" query: {} episodes in {:?}", results.len(), query_latency);

    let search_start = Instant::now();
    let search_results = store.search_episodes("Word", 50).expect("search_episodes");
    let search_latency = search_start.elapsed();
    println!("equivalent FTS search (\"Word\"): {} episodes in {:?}", search_results.len(), search_latency);
}
