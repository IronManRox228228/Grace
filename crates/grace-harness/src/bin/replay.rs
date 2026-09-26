//! `grace-replay --corpus <dir>`: reports structural parity between
//! `corpus/` and the Rust `GraceEvent` contract. See `grace_harness`'s crate
//! doc for exactly what this does and does not check yet - in short, this is
//! "does every tape event fit the frozen contract", not yet "does the Rust
//! backend reproduce the tape's route", because the agent loop / dispatcher
//! / perception pipeline this phase ported does not yet include a full turn
//! driver to replay against. That is the honest scope of this phase's
//! harness; see PORT_STATUS.md for the plan to close the gap.

use grace_harness::{check_tape_schema, Tape};
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut corpus_dir: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => corpus_dir = args.next().map(PathBuf::from),
            other => {
                eprintln!("unrecognised argument: {other}");
                std::process::exit(2);
            }
        }
    }
    let corpus_dir = corpus_dir.unwrap_or_else(|| PathBuf::from("corpus"));

    let tapes = match Tape::load_corpus(&corpus_dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed to load corpus at {}: {e}", corpus_dir.display());
            std::process::exit(1);
        }
    };

    println!(
        "grace-replay: structural-parity check ({} tape(s) in {})",
        tapes.len(),
        corpus_dir.display()
    );
    println!(
        "NOTE: this checks that every tape event fits the frozen GraceEvent contract."
    );
    println!(
        "It does NOT yet re-run a turn and diff routes/prompts (see PORT_STATUS.md)."
    );
    println!(
        "The committed corpus/ tapes were recorded against Python HEAD, not the"
    );
    println!(
        "uncommitted working tree; a working-tree tool-schema change means recorded"
    );
    println!("prompts (and some downstream events) are expected to be stale.\n");

    let mut any_failed = false;
    for tape in &tapes {
        let report = check_tape_schema(tape);
        let synthetic = if report.is_synthetic { "synthetic" } else { "recorded" };
        if report.ok() {
            println!(
                "  OK    {:<40} {:>3} events ({synthetic})",
                report.session_id, report.event_count
            );
        } else {
            any_failed = true;
            println!(
                "  FAIL  {:<40} {:>3} events ({synthetic}) - {} schema failure(s)",
                report.session_id,
                report.event_count,
                report.schema_failures.len()
            );
            for (index, error) in &report.schema_failures {
                println!("          [{index}] {error}");
            }
        }
    }

    if any_failed {
        std::process::exit(1);
    }
}
