//! Ported from `src/grace/intent/router.py` (`CapabilityRouter`).
//!
//! Classifies incoming user voice requests into either a fast, deterministic
//! path, an agentic multi-step goal, or direct conversation. See the module
//! doc in the Python source for why the agentic loop is the default and the
//! fast path is the opt-in exception, not the other way round.

use regex::Regex;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskComplexity {
    FastPath,
    AgenticGoal,
    Conversation,
}

/// The subset of a parsed intent the router needs. Mirrors the two fields
/// of `grace.intent.parser.Intent` that `CapabilityRouter.classify` reads.
#[derive(Debug, Clone)]
pub struct ParsedIntentRef<'a> {
    pub tool: Option<&'a str>,
    pub is_conversation: bool,
}

impl<'a> From<&'a crate::intent::Intent> for ParsedIntentRef<'a> {
    fn from(intent: &'a crate::intent::Intent) -> Self {
        Self {
            tool: Some(intent.tool.as_str()),
            is_conversation: intent.is_conversation(),
        }
    }
}

/// Tools that complete in one deterministic pass. No screen state is read,
/// so there is nothing for the agentic loop to add.
pub const FAST_PATH_TOOLS: &[&str] = &[
    "adjust_volume",
    "lock_computer",
    "open_calculator",
    "open_app",
    "close_app",
    "open_file",
    "search_files",
    "delete_file",
    "cua_launch",
    "cua_list_windows",
    "cua_list_apps",
    "cua_press_key",
    "cua_scroll",
    "cua_activate",
    "undo",
    "describe_screen",
    "set_speech_rate",
];

/// Tools that inherently need to look at the screen and iterate.
pub const AGENTIC_TOOLS: &[&str] = &[
    "cua_click",
    "cua_type_text",
    "cua_drag",
    "cua_set_value",
    "cua_secondary_action",
    "read_pdf",
    "summarize_pdf",
];

/// Words that join a second clause onto the first.
const CLAUSE_SEPARATORS: &[&str] = &["and", "then", "also", "next", "after that", "plus"];

/// Longest utterance still treated as a single deterministic command.
const MAX_FAST_PATH_WORDS: usize = 8;

const CONVERSATION_KEYWORDS: &[&str] = &[
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "goodbye",
    "bye",
    "who are you",
    "what can you do",
    "how are you",
];

/// Whole-word / whole-phrase, case-insensitive matching - never a substring
/// match (the bug this router was written to fix: "open" used to match
/// "opening", "x" matched any word containing an x).
fn mentions(text: &str, phrases: &[&str]) -> bool {
    for phrase in phrases {
        let escaped: Vec<String> = phrase.split(' ').map(regex::escape).collect();
        let pattern = format!(r"\b{}\b", escaped.join(r"\s+"));
        // Built once per call (the phrase list is short and this runs at most
        // once per utterance); a static per-phrase cache would be premature
        // given the corpus's latency budget is dominated by the LLM call.
        let re = Regex::new(&format!("(?i){pattern}")).expect("valid regex");
        if re.is_match(text) {
            return true;
        }
    }
    false
}

static WORD_SPLIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

fn word_count(text: &str) -> usize {
    WORD_SPLIT
        .split(text.trim())
        .filter(|w| !w.is_empty())
        .count()
}

/// Why this utterance is more than one command, or `None` if it isn't.
fn not_atomic(text: &str) -> Option<String> {
    if mentions(text, CLAUSE_SEPARATORS) {
        return Some("the request has a second clause".to_string());
    }
    let words = word_count(text);
    if words > MAX_FAST_PATH_WORDS {
        return Some(format!("the request is {words} words, longer than one command"));
    }
    None
}

/// Determine task complexity path.
pub fn classify(prompt_text: &str, parsed_intent: Option<ParsedIntentRef>) -> TaskComplexity {
    let prompt_lower = prompt_text.to_lowercase();
    let prompt_lower = prompt_lower.trim();

    if let Some(intent) = &parsed_intent {
        let tool = intent.tool.unwrap_or("");

        if AGENTIC_TOOLS.contains(&tool) {
            return TaskComplexity::AgenticGoal;
        }

        if FAST_PATH_TOOLS.contains(&tool) {
            return match not_atomic(prompt_lower) {
                None => TaskComplexity::FastPath,
                Some(_reason) => TaskComplexity::AgenticGoal,
            };
        }

        if intent.is_conversation {
            return TaskComplexity::Conversation;
        }
    }

    if mentions(prompt_lower, CONVERSATION_KEYWORDS) {
        return TaskComplexity::Conversation;
    }

    TaskComplexity::AgenticGoal
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intent(tool: &str) -> ParsedIntentRef<'_> {
        ParsedIntentRef {
            tool: Some(tool),
            is_conversation: false,
        }
    }

    #[test]
    fn fast_path_tool_with_short_atomic_command_stays_fast_path() {
        assert_eq!(
            classify("open whatsapp", Some(intent("open_app"))),
            TaskComplexity::FastPath
        );
    }

    #[test]
    fn fast_path_tool_with_a_second_clause_goes_agentic() {
        assert_eq!(
            classify("open whatsapp and search for the pdf", Some(intent("open_app"))),
            TaskComplexity::AgenticGoal
        );
    }

    #[test]
    fn fast_path_tool_with_a_long_utterance_goes_agentic() {
        assert_eq!(
            classify(
                "please could you go ahead and open the whatsapp application for me now",
                Some(intent("open_app"))
            ),
            TaskComplexity::AgenticGoal
        );
    }

    #[test]
    fn agentic_tool_is_always_agentic_regardless_of_length() {
        assert_eq!(
            classify("click", Some(intent("cua_click"))),
            TaskComplexity::AgenticGoal
        );
    }

    #[test]
    fn conversation_intent_routes_to_conversation() {
        let intent = ParsedIntentRef {
            tool: None,
            is_conversation: true,
        };
        assert_eq!(classify("how are you", Some(intent)), TaskComplexity::Conversation);
    }

    #[test]
    fn no_intent_but_greeting_phrasing_routes_to_conversation() {
        assert_eq!(classify("hey there", None), TaskComplexity::Conversation);
    }

    #[test]
    fn substring_matches_do_not_falsely_trigger_conversation() {
        // The bug this router replaced: "opening" must not match "open", and
        // more to the point here, "hi" must not match inside "history".
        assert_eq!(classify("show me my history", None), TaskComplexity::AgenticGoal);
    }

    #[test]
    fn word_boundary_clause_separator_does_not_match_inside_a_word() {
        // "and" must not match inside "sandwich".
        assert_eq!(
            classify("open the sandwich shop app", Some(intent("open_app"))),
            TaskComplexity::FastPath
        );
    }

    #[test]
    fn no_intent_and_no_keyword_defaults_to_agentic() {
        assert_eq!(classify("do the thing with the file", None), TaskComplexity::AgenticGoal);
    }
}
