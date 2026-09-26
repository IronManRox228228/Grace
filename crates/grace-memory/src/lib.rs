//! Grace's memory, written from scratch in Rust behind `grace-core`'s
//! `PersistentStore` trait (PLAN.md §12; ported/replaced from
//! `src/grace/agent/memory.py`'s `PersistentMemoryStore`, which is not kept:
//! it never wrote `user_preferences` and appended every step's raw goal,
//! params and result to `task_history` forever, in plain text).
//!
//! Two stores, because they have different jobs:
//! - [`facts`]: small, about the user, must be correct (§12.1).
//! - [`history`]: huge, about what was done, must be cheap (§12.2).
//!
//! [`adapter::GraceMemoryStore`] wires both onto the existing
//! `grace_core::memory::PersistentStore` trait so the agent loop can use
//! this crate as a drop-in replacement for `InMemoryStore` without further
//! changes to `grace-core`.

pub mod adapter;
pub mod db;
pub mod error;
pub mod facts;
pub mod history;
pub mod redact;
pub mod secret;

pub use error::{MemoryError, Result};
