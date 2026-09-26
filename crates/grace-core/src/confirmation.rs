//! Ported from `GraceApp._confirmation_answer` in `src/grace/main.py`
//! (ship-blocker fix #3, `tests/test_ship_blockers.py::TestConfirmationAnswerIsStrict`).
//!
//! Strict on purpose: the utterance, once lowercased and stripped of
//! punctuation and filler, must reduce to a bare yes/no or one of a small
//! set of stock confirmation phrases - optionally preceded by that yes/no -
//! rather than merely containing one somewhere. Anything else is `None`,
//! which the caller (agent-loop resume logic) treats as cancelling the
//! pending action and reading the utterance as a new request.

const YES_WORDS: &[&str] = &["yes", "yeah", "yep", "yup", "sure", "ok", "okay", "affirmative", "correct"];
const NO_WORDS: &[&str] = &["no", "nope", "negative"];
const PROCEED_PHRASES: &[&str] = &["go ahead", "go on", "go for it", "do it", "confirm", "please do"];
const STOP_PHRASES: &[&str] = &[
    "dont",
    "dont do that",
    "do not do that",
    "cancel",
    "cancel that",
    "stop",
    "stop that",
    "never mind",
    "nevermind",
    "abort",
];
const NEGATION_WORDS: &[&str] = &["not", "no", "dont", "never"];
const LEADING_FILLER: &[&str] = &["grace", "um", "uh", "umm", "uhh"];
const TRAILING_FILLER: &[&str] = &["please"];

/// Strip everything that isn't a word character or whitespace, matching
/// Python's `re.sub(r"[^\w\s]", "", text)` (both operate on already-ASCII
/// voice transcripts in practice, so byte-wise ASCII punctuation stripping
/// is equivalent here).
fn strip_punctuation(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '_')
        .collect()
}

/// Read a yes/no out of an utterance. `None` means it wasn't a plain answer.
pub fn confirmation_answer(transcript: &str) -> Option<bool> {
    let text = strip_punctuation(&transcript.to_lowercase());
    let text = text.trim();
    if text.is_empty() {
        return None;
    }

    let mut words: Vec<&str> = text.split_whitespace().collect();
    while let Some(first) = words.first() {
        if LEADING_FILLER.contains(first) {
            words.remove(0);
        } else {
            break;
        }
    }
    while let Some(last) = words.last() {
        if TRAILING_FILLER.contains(last) {
            words.pop();
        } else {
            break;
        }
    }
    if words.is_empty() {
        return None;
    }

    let tail_words: Vec<&str>;
    if YES_WORDS.contains(&words[0]) {
        tail_words = words[1..].to_vec();
    } else if NO_WORDS.contains(&words[0]) {
        let tail_words = &words[1..];
        let tail = tail_words.join(" ");
        if tail.is_empty() || STOP_PHRASES.contains(&tail.as_str()) {
            return Some(false);
        }
        return None; // "no" followed by something else isn't a plain answer
    } else {
        tail_words = words.clone();
    }

    let tail = tail_words.join(" ");
    if tail.is_empty() && YES_WORDS.contains(&words[0]) {
        return Some(true);
    }
    if PROCEED_PHRASES.contains(&tail.as_str())
        && !tail_words.iter().any(|w| NEGATION_WORDS.contains(w))
    {
        return Some(true);
    }
    if STOP_PHRASES.contains(&tail.as_str()) {
        return Some(false);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unrelated_speech_is_not_read_as_an_answer() {
        for utterance in ["I am not sure", "okay open spotify", "click OK"] {
            assert_eq!(confirmation_answer(utterance), None, "{utterance}");
        }
    }

    #[test]
    fn plain_affirmatives_are_recognised() {
        for utterance in ["yes", "Yes.", "okay", "yes please", "go ahead"] {
            assert_eq!(confirmation_answer(utterance), Some(true), "{utterance}");
        }
    }

    #[test]
    fn plain_negatives_are_recognised() {
        for utterance in ["no", "don't"] {
            assert_eq!(confirmation_answer(utterance), Some(false), "{utterance}");
        }
    }

    #[test]
    fn leading_and_trailing_filler_is_stripped() {
        assert_eq!(confirmation_answer("um yes please"), Some(true));
        assert_eq!(confirmation_answer("grace yes"), Some(true));
    }

    #[test]
    fn tail_must_match_a_stock_phrase_exactly() {
        // "yes go ahead dont" isn't a plain answer: the tail ("go ahead
        // dont") doesn't exactly equal any PROCEED phrase, so it falls
        // through to `None` rather than fuzzy-matching "go ahead".
        assert_eq!(confirmation_answer("yes go ahead dont"), None);
        // "yes" followed by a tail that separately matches a STOP phrase
        // ("do not do that") resolves via the STOP-phrase check, giving
        // `Some(false)` - the yes/no words at the front of an utterance
        // don't lock in that reading if the rest of the sentence disagrees.
        assert_eq!(confirmation_answer("yes do not do that"), Some(false));
    }

    #[test]
    fn no_followed_by_unrelated_text_is_not_a_plain_answer() {
        assert_eq!(confirmation_answer("no I want something else"), None);
    }
}
