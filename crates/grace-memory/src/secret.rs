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
const SECRET_TOPIC_HINTS: [&str; 14] = [
    "password", "passwd", "pwd", "secret", "token", "pin", "otp", "cvv",
    "cvc", "ssn", "api_key", "apikey", "credit_card", "passphrase",
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

    // A run of 13-19 digits (with optional spaces/dashes/commas/dots, as people
    // or ASR format card numbers) is card-number-shaped (R25).
    if looks_like_card_number(trimmed) {
        return true;
    }

    // Passphrase heuristic (R25): 3+ words separated by hyphens, underscores,
    // or dots, with length >= 14 (e.g. "correct-horse-battery-staple").
    if looks_like_passphrase(trimmed) {
        return true;
    }

    // Check character classes per token (R25). A token of length >= 8 that mixes
    // >= 3 classes (lowercase, uppercase, digit, symbol) or weak 2-class passwords
    // (letters + digits with >= 2 digits and >= 3 letters) is password-shaped.
    for token in trimmed.split_whitespace() {
        let clean = token.trim_matches(|c: char| c.is_ascii_punctuation());
        if clean.chars().count() >= 8 {
            let mut lower = false;
            let mut upper = false;
            let mut digit = false;
            let mut symbol = false;
            let mut digit_count = 0;
            let mut letter_count = 0;
            for c in clean.chars() {
                if c.is_ascii_lowercase() {
                    lower = true;
                    letter_count += 1;
                } else if c.is_ascii_uppercase() {
                    upper = true;
                    letter_count += 1;
                } else if c.is_ascii_digit() {
                    digit = true;
                    digit_count += 1;
                } else if !c.is_alphanumeric() {
                    symbol = true;
                }
            }
            let classes = [lower, upper, digit, symbol].iter().filter(|b| **b).count();
            if classes >= 3 {
                return true;
            }
            if (lower || upper) && digit && digit_count >= 2 && letter_count >= 3 {
                return true;
            }
        }
    }

    false
}

fn looks_like_passphrase(s: &str) -> bool {
    if !s.contains(char::is_whitespace) {
        for delim in ['-', '_', '.'] {
            if s.contains(delim) {
                let parts: Vec<&str> = s.split(delim).collect();
                if parts.len() >= 3
                    && parts.iter().all(|p| p.len() >= 3 && p.chars().all(|c| c.is_ascii_alphabetic()))
                    && s.len() >= 14
                {
                    return true;
                }
            }
        }
    }
    false
}

fn looks_like_card_number(s: &str) -> bool {
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }
    // Only digits, spaces, dashes, commas, periods (ASR punctuation) - not a
    // sentence that happens to contain a long number.
    s.chars().all(|c| c.is_ascii_digit() || c == ' ' || c == '-' || c == ',' || c == '.')
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

    #[test]
    fn flags_r25_punctuated_cards_and_passphrases_and_weak_passwords() {
        // Punctuated card numbers (commas/dots from ASR)
        assert!(looks_like_secret("note", "4111, 1111, 1111, 1111"));
        assert!(looks_like_secret("note", "4111.1111.1111.1111"));

        // Delimiter-separated passphrases
        assert!(looks_like_secret("note", "correct-horse-battery-staple"));
        assert!(looks_like_secret("note", "blue_sky_morning_coffee"));

        // Passphrase topic
        assert!(looks_like_secret("my passphrase", "anything"));

        // Multi-word strings with secret token
        assert!(looks_like_secret("note", "the password was Tr0ub4dor&3xyz yesterday"));

        // Weak two-class passwords
        assert!(looks_like_secret("note", "hunter22"));
        assert!(looks_like_secret("note", "password123"));
    }
}
