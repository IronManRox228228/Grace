//! Redaction at write time (PLAN.md §12.2): typed text is stored as its
//! length and a salted hash, never its content. The hash only needs to be a
//! stable fingerprint (so, e.g., "did I type the same password again" can
//! be answered without ever storing the password) - it is not a security
//! primitive, so a `std`-only SipHash keeps this crate's dependency list
//! small.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A random-at-creation salt stored once per history database (see
/// `history::META_SALT_KEY`), so the hash can't be reversed by a rainbow
/// table built against no salt at all.
pub fn salted_hash(salt: u64, text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    salt.hash(&mut hasher);
    text.hash(&mut hasher);
    hasher.finish()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Redacted {
    pub len: u16,
    pub hash: u64,
}

pub fn redact(salt: u64, text: &str) -> Redacted {
    Redacted { len: text.chars().count().min(u16::MAX as usize) as u16, hash: salted_hash(salt, text) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn different_salts_hash_the_same_text_differently() {
        assert_ne!(salted_hash(1, "hello"), salted_hash(2, "hello"));
    }

    #[test]
    fn redact_never_returns_the_original_text() {
        let r = redact(42, "super secret typed text");
        assert_eq!(r.len, 23);
        // Nothing to assert against the hash's value directly (it's a hash),
        // but the type itself has no field that could hold the source text.
    }
}
