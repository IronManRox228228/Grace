//! Error type shared by the facts and history stores.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// `FactStore::store`/`confirm` refused a value that looks like a
    /// secret (PLAN.md §12.1: "Never stored: anything typed into a password
    /// field or a field that looks secret.").
    #[error("refused to store a secret-looking value for topic {topic:?}")]
    LooksLikeSecret { topic: String },

    /// `FactStore::supersede`/`confirm` was asked to act on a topic that has
    /// no currently-active fact.
    #[error("no active fact for topic {topic:?}")]
    NoActiveFact { topic: String },

    /// `FactStore::store` was asked to create a topic that already has an
    /// active fact. Callers must go through `supersede` (with a reason) so
    /// a correction is always recorded, never a silent overwrite.
    #[error("topic {topic:?} already has an active fact; use supersede")]
    TopicAlreadyActive { topic: String },

    #[error("postcard encode/decode error: {0}")]
    Postcard(#[from] postcard::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("the history writer thread is gone")]
    WriterGone,
}

pub type Result<T> = std::result::Result<T, MemoryError>;
