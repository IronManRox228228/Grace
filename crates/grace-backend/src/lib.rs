//! The Rust backend's transport layer: the WebSocket event server, with the
//! Origin allowlist ship-blocker fix (PLAN.md §10.1 item 5) ported and
//! tested against real loopback connections. See `server.rs`'s doc comment
//! for what this phase does and does not do - in short, this proves the
//! wire-format and security property, not a working turn.

pub mod origin;
pub mod server;
pub mod turn;

pub use origin::{default_allowed_origins, is_origin_allowed, parse_extra_origins};
pub use server::{WakeCallback, WsEventServer};
pub use turn::run_demo_turn;
