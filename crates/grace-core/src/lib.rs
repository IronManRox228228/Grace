//! Pure, heavily-tested core logic ported from the Python `grace` package:
//! config, the safety guard, the confirmation-answer parser, the capability
//! router / pattern fast path, and sentence splitting. See PORT_STATUS.md
//! for what from `src/grace/agent/loop.py`, `perception.py` and `memory.py`
//! is not yet ported into this crate.

pub mod agent_loop;
pub mod config;
pub mod confirmation;
pub mod dispatcher;
pub mod events;
pub mod feedback;
pub mod grace_app;
pub mod grounder;
pub mod intent;
pub mod memory;
pub mod models;
pub mod perception;
pub mod planner;
pub mod response_generator;
pub mod router;
pub mod safety;
pub mod sentence_split;
pub mod timing;
pub mod tools;
pub mod ui_tars_parser;

pub use config::Config;
