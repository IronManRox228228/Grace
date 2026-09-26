//! Ported from `src/grace/text/sentence_split.py`.
//!
//! Shared sentence-splitting utility for TTS chunking. Used by both the
//! Kokoro engine and the response generator to break long text into
//! speakable sentence chunks without splitting on abbreviations (e.g. "Dr.
//! Smith"). `ResponseChunk`/`SpeechChunk` boundaries in the event contract
//! are exactly the boundaries this function produces, so it has to match
//! the Python behaviour byte-for-byte.

use std::collections::BTreeSet;
use std::sync::LazyLock;

static ABBREVIATIONS: LazyLock<BTreeSet<&'static str>> = LazyLock::new(|| {
    [
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "eg", "ie", "vs", "etc", "approx",
        "dept", "est", "govt", "capt", "lt", "col", "gen", "sgt", "vol", "no", "co", "inc", "ltd",
        "corp",
    ]
    .into_iter()
    .collect()
});

/// Split on whitespace that follows `.`, `!` or `?` - i.e. `re.split(r"(?<=[.!?])\s+", text)`.
fn split_on_sentence_boundaries(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        current.push(c);
        if matches!(c, '.' | '!' | '?') {
            // Consume the boundary: one or more whitespace chars immediately
            // following count as the split point, mirroring `\s+`.
            let mut j = i + 1;
            let mut saw_ws = false;
            while j < chars.len() && chars[j].is_whitespace() {
                saw_ws = true;
                j += 1;
            }
            if saw_ws {
                parts.push(std::mem::take(&mut current));
                i = j;
                continue;
            }
        }
        i += 1;
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// Split text into sentences, merging false splits on abbreviations.
pub fn split_sentences(text: &str) -> Vec<String> {
    let parts = split_on_sentence_boundaries(text.trim());
    let mut merged = Vec::new();
    let mut i = 0;
    while i < parts.len() {
        let mut part = parts[i].trim().to_string();
        if part.is_empty() {
            i += 1;
            continue;
        }
        while i + 1 < parts.len() {
            let tail = part
                .split_whitespace()
                .last()
                .map(|w| w.trim_end_matches('.').to_lowercase())
                .unwrap_or_default();
            if ABBREVIATIONS.contains(tail.as_str()) {
                part = format!("{} {}", part, parts[i + 1].trim());
                i += 1;
            } else {
                break;
            }
        }
        merged.push(part);
        i += 1;
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_plain_sentences() {
        assert_eq!(
            split_sentences("Hello there. How are you?"),
            vec!["Hello there.", "How are you?"]
        );
    }

    #[test]
    fn does_not_split_on_an_abbreviation() {
        assert_eq!(
            split_sentences("Dr. Smith is here. He is early."),
            vec!["Dr. Smith is here.", "He is early."]
        );
    }

    #[test]
    fn handles_a_string_of_abbreviations() {
        assert_eq!(
            split_sentences("See Mr. and Mrs. Smith tomorrow."),
            vec!["See Mr. and Mrs. Smith tomorrow."]
        );
    }

    #[test]
    fn single_sentence_with_no_terminal_punctuation() {
        assert_eq!(split_sentences("hello world"), vec!["hello world"]);
    }

    #[test]
    fn empty_and_whitespace_only_input() {
        assert_eq!(split_sentences(""), Vec::<String>::new());
        assert_eq!(split_sentences("   "), Vec::<String>::new());
    }

    #[test]
    fn exclamation_and_question_marks_split_too() {
        assert_eq!(
            split_sentences("Wait! Really? Yes."),
            vec!["Wait!", "Really?", "Yes."]
        );
    }
}
