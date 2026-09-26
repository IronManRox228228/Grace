//! Ported from `src/grace/intent/parser.py`: validates and parses JSON
//! output from the intent model.

use crate::tools::all_tools;
use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::LazyLock;

pub static VALID_TOOLS: LazyLock<BTreeSet<&'static str>> =
    LazyLock::new(|| all_tools().iter().map(|t| t.name).collect());

#[derive(Debug, Clone, PartialEq)]
pub struct Intent {
    pub tool: String,
    pub params: Value,
    pub response: Option<String>,
}

impl Intent {
    pub fn new(tool: impl Into<String>, params: Value, response: Option<String>) -> Self {
        Self { tool: tool.into(), params, response }
    }

    /// Whether this intent requires the CUA bridge.
    pub fn needs_cua(&self) -> bool {
        self.tool.starts_with("cua_")
    }

    pub fn is_conversation(&self) -> bool {
        self.tool == "converse"
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum IntentParseError {
    #[error("Failed to parse JSON: {0}")]
    Json(String),
    #[error("Expected JSON object, got {0}")]
    NotAnObject(String),
    #[error("Missing required field: 'tool'")]
    MissingTool,
    #[error("Unknown tool '{0}'. Valid tools: {1:?}")]
    UnknownTool(String, Vec<&'static str>),
    #[error("'params' must be a dict, got {0}")]
    ParamsNotADict(String),
}

/// Strip markdown code fences and whitespace from raw JSON strings.
pub fn clean_json_fence(raw_json: &str) -> String {
    let cleaned = raw_json.trim();
    if !cleaned.starts_with("```") {
        return cleaned.to_string();
    }

    let mut opened = false;
    let mut content_lines: Vec<&str> = Vec::new();
    for line in cleaned.split('\n') {
        let stripped = line.trim();
        if !opened {
            if stripped.starts_with("```") {
                opened = true;
            }
            continue;
        }
        if stripped.starts_with("```") {
            break;
        }
        content_lines.push(line);
    }
    content_lines.join("\n").trim().to_string()
}

#[derive(Default)]
pub struct IntentParser {
    last_intent: Option<Intent>,
}

impl IntentParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn last_intent(&self) -> Option<&Intent> {
        self.last_intent.as_ref()
    }

    pub fn parse(&mut self, raw_json: &str) -> Result<Intent, IntentParseError> {
        let cleaned = clean_json_fence(raw_json);

        let data: Value =
            serde_json::from_str(&cleaned).map_err(|e| IntentParseError::Json(e.to_string()))?;

        let obj = data
            .as_object()
            .ok_or_else(|| IntentParseError::NotAnObject(json_type_name(&data)))?;

        let tool = obj
            .get("tool")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or(IntentParseError::MissingTool)?;

        if !VALID_TOOLS.contains(tool) {
            let mut valid: Vec<&'static str> = VALID_TOOLS.iter().copied().collect();
            valid.sort();
            return Err(IntentParseError::UnknownTool(tool.to_string(), valid));
        }

        let params = obj.get("params").cloned().unwrap_or(Value::Object(Default::default()));
        if !params.is_object() {
            return Err(IntentParseError::ParamsNotADict(json_type_name(&params)));
        }

        // top-level "response", falling back to params["response"]
        let response = obj
            .get("response")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| params.get("response").and_then(Value::as_str).filter(|s| !s.is_empty()))
            .map(str::to_string);

        let intent = Intent::new(tool.to_string(), params, response);
        self.last_intent = Some(intent.clone());
        Ok(intent)
    }

    /// Extract the verbal response text from an intent. For converse tools,
    /// returns `intent.response` or `params["response"]`. For other tools,
    /// returns an empty string (response generated separately).
    pub fn extract_response_text(&self, intent: &Intent) -> String {
        if intent.is_conversation() {
            if let Some(r) = &intent.response {
                return r.clone();
            }
            if let Some(r) = intent.params.get("response").and_then(Value::as_str) {
                return r.to_string();
            }
        }
        String::new()
    }
}

fn json_type_name(value: &Value) -> String {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strips_a_markdown_fence() {
        assert_eq!(clean_json_fence("```json\n{\"tool\": \"open_app\"}\n```"), "{\"tool\": \"open_app\"}");
        assert_eq!(clean_json_fence("  {\"tool\": \"open_app\"}  "), "{\"tool\": \"open_app\"}");
    }

    #[test]
    fn parses_a_valid_intent() {
        let mut parser = IntentParser::new();
        let intent = parser
            .parse(r#"{"tool": "open_app", "params": {"name": "Notepad"}}"#)
            .unwrap();
        assert_eq!(intent.tool, "open_app");
        assert_eq!(intent.params, json!({"name": "Notepad"}));
        assert!(!intent.needs_cua());
        assert!(!intent.is_conversation());
    }

    #[test]
    fn cua_tool_needs_cua() {
        let mut parser = IntentParser::new();
        let intent = parser.parse(r#"{"tool": "cua_click", "params": {}}"#).unwrap();
        assert!(intent.needs_cua());
    }

    #[test]
    fn rejects_invalid_json() {
        let mut parser = IntentParser::new();
        assert!(matches!(parser.parse("not json"), Err(IntentParseError::Json(_))));
    }

    #[test]
    fn rejects_unknown_tool() {
        let mut parser = IntentParser::new();
        assert!(matches!(
            parser.parse(r#"{"tool": "not_a_real_tool"}"#),
            Err(IntentParseError::UnknownTool(_, _))
        ));
    }

    #[test]
    fn rejects_missing_tool() {
        let mut parser = IntentParser::new();
        assert!(matches!(parser.parse(r#"{"params": {}}"#), Err(IntentParseError::MissingTool)));
    }

    #[test]
    fn rejects_non_dict_params() {
        let mut parser = IntentParser::new();
        assert!(matches!(
            parser.parse(r#"{"tool": "open_app", "params": "not a dict"}"#),
            Err(IntentParseError::ParamsNotADict(_))
        ));
    }

    #[test]
    fn response_can_come_from_top_level_or_params() {
        let mut parser = IntentParser::new();
        let a = parser
            .parse(r#"{"tool": "converse", "response": "hi", "params": {}}"#)
            .unwrap();
        assert_eq!(a.response.as_deref(), Some("hi"));

        let b = parser
            .parse(r#"{"tool": "converse", "params": {"response": "hey"}}"#)
            .unwrap();
        assert_eq!(b.response.as_deref(), Some("hey"));
    }

    #[test]
    fn extract_response_text_only_for_conversation() {
        let parser = IntentParser::new();
        let convo = Intent::new("converse", json!({}), Some("hi there".to_string()));
        assert_eq!(parser.extract_response_text(&convo), "hi there");

        let tool = Intent::new("open_app", json!({"name": "Notepad"}), None);
        assert_eq!(parser.extract_response_text(&tool), "");
    }

    #[test]
    fn last_intent_tracks_the_most_recent_parse() {
        let mut parser = IntentParser::new();
        assert!(parser.last_intent().is_none());
        parser.parse(r#"{"tool": "lock_computer"}"#).unwrap();
        assert_eq!(parser.last_intent().unwrap().tool, "lock_computer");
    }
}
