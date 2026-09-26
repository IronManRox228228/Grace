//! Shared SQLite connection setup for both stores: WAL mode,
//! `synchronous=NORMAL` and incremental auto-vacuum (PLAN.md §12.2). The
//! facts store and the history store each open their own connection (the
//! history store's writer thread additionally opens a second, read-only
//! connection for queries - see `history::HistoryStore`), but both need the
//! same pragmas, so it lives here once.

use crate::error::Result;
use rusqlite::Connection;
use std::path::Path;

pub(crate) fn open(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)?;
    apply_pragmas(&conn)?;
    Ok(conn)
}

fn apply_pragmas(conn: &Connection) -> Result<()> {
    // auto_vacuum must be set before any tables exist to take effect without
    // a full VACUUM, so callers open the connection and call this before
    // running their schema's CREATE TABLE statements.
    conn.pragma_update(None, "auto_vacuum", "INCREMENTAL")?;
    // `PRAGMA journal_mode = WAL` uniquely among pragmas returns a row (the
    // resulting mode), so `pragma_update` (which expects none) reports it
    // as "execute returned results" - `pragma_update_and_check` is the
    // variant built for exactly this pragma.
    conn.pragma_update_and_check(None, "journal_mode", "WAL", |_row| Ok(()))?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts5_is_compiled_into_the_bundled_sqlite() {
        // Confirms the `fts5` cargo feature actually did what it promises:
        // without it, this CREATE VIRTUAL TABLE fails with "no such module:
        // fts5" at the SQLite level, which is exactly the failure mode
        // PLAN.md's task asked us to guard against ("confirm FTS5 is
        // compiled in; if not, enable it").
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE VIRTUAL TABLE t USING fts5(body);")
            .expect("FTS5 must be compiled into the bundled SQLite");
    }
}
