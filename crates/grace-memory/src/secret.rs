//! The "looks like a secret" guard from PLAN.md §12.1: "Never stored:
//! anything typed into a password field or a field that looks secret."
//!
//! Two independent checks, either of which refuses the write:
//! 1. an explicit `is_password_field` flag the caller sets when the value
//!    came from a UI field UIA/DOM marks as a password (the reliable case);
//! 2. a conservative heuristic over the topic name and the value itself,
//!    for callers that don't have field metadata (ASR, free text).
//!
//! "Conservative" here means biased against false positives: a fact about
//! the user is useful precisely because Grace remembers it, so the
//! heuristic only trips on values that are actively secret-shaped (dense
//! mixed-case-and-digit tokens with no spaces, or things that look like
//! card/account numbers), not merely long or unusual ones.

/// Substrings in a topic name that mark it as secret regardless of the
/// value's shape. Checked case-insensitively.
const SECRET_TOPIC_HINTS: [&str; 13] = [
    "password", "passwd", "pwd", "secret", "token", "pin", "otp", "cvv",
    "cvc", "ssn", "api_key", "apikey", "credit_card",
];

pub fn looks_like_secret(topic: &str, value: &str) -> bool {
    let topic_lower = topic.to_lowercase();
    if SECRET_TOPIC_HINTS.iter().any(|hint| topic_lower.contains(hint)) {
        return true;
    }
    looks_like_secret_value(value)
}

fn looks_like_secret_value(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return false;
    }

    // A run of 13-19 digits (with optional spaces/dashes every 4, as people
    // read card numbers aloud) is card-number-shaped.
    if looks_like_card_number(trimmed) {
        return true;
    }

    // A single token (no whitespace) that mixes at least three of
    // {lowercase, uppercase, digit, symbol} character classes and has no
    // dictionary-word spaces is password-shaped: normal facts ("Priya",
    // "3rd floor", "af_bella") don't look like this.
    if trimmed.split_whitespace().count() == 1 && trimmed.chars().count() >= 8 {
        let mut lower = false;
        let mut upper = false;
        let mut digit = false;
        let mut symbol = false;
        for c in trimmed.chars() {
            if c.is_ascii_lowercase() {
                lower = true;
            } else if c.is_ascii_uppercase() {
                upper = true;
            } else if c.is_ascii_digit() {
                digit = true;
            } else if !c.is_alphanumeric() {
                symbol = true;
            }
        }
        let classes = [lower, upper, digit, symbol].iter().filter(|b| **b).count();
        if classes >= 3 {
            return true;
        }
    }

    false
}

fn looks_like_card_number(s: &str) -> bool {
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }
    // Only digits, spaces and dashes - not a sentence that happens to
    // contain a long number (e.g. "born in 1990, room 12345678901234").
    s.chars().all(|c| c.is_ascii_digit() || c == ' ' || c == '-')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_topic_names_that_say_password() {
        assert!(looks_like_secret("login.password", "anything"));
        assert!(looks_like_secret("wifi_pin", "1234"));
        assert!(looks_like_secret("card CVV", "123"));
    }

    #[test]
    fn flags_password_shaped_single_tokens() {
        assert!(looks_like_secret("note", "Tr0ub4dor&3xyz"));
        assert!(looks_like_secret("note", "aB3!fooBar9$"));
    }

    #[test]
    fn flags_card_shaped_digit_runs() {
        assert!(looks_like_secret("note", "4111 1111 1111 1111"));
        assert!(looks_like_secret("note", "4111-1111-1111-1111"));
    }

    #[test]
    fn does_not_flag_ordinary_facts() {
        assert!(!looks_like_secret("contact:priya:relation", "sister"));
        assert!(!looks_like_secret("preference:voice", "af_bella"));
        assert!(!looks_like_secret("contact:priya:phone", "555-0142"));
        assert!(!looks_like_secret("rule:deletes", "always ask before deleting"));
        assert!(!looks_like_secret("note", "3rd floor, room 12"));
    }
}
