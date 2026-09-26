//! Ported from `src/grace/agent/ui_tars_parser.py`: parses native UI-TARS
//! model text output (Thought / Action format) into structured step data.

use regex::Regex;
use serde_json::{json, Value};
use std::sync::LazyLock;

/// A coordinate pair, with or without the box tokens and quoting the model uses.
const BOX_SRC: &str = r"(?:<\|box_start\|>)?\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?\s*(?:<\|box_end\|>)?";

fn coord_pattern(keywords: &str) -> String {
    let arg_names = "start_box|point|location|coordinate";
    format!(r#"(?i)(?:{keywords})\s*\(\s*(?:(?:{arg_names})\s*=\s*)?['"]?{BOX_SRC}['"]?\s*\)"#)
}

static BOX_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(&format!("(?i){BOX_SRC}")).unwrap());

/// A parsed step, as a JSON object shaped like the Python dict the original
/// returns (`thought`, `action`, `params`, `user_update`, `is_completed`,
/// optionally `final_response`). Kept as `Value` rather than a dedicated
/// struct because the shape genuinely varies per action and the loop only
/// ever consumes it as a map.
pub type ParsedStep = Value;

pub struct UiTarsParser;

impl UiTarsParser {
    pub fn parse_response(response_text: &str) -> Option<ParsedStep> {
        if response_text.is_empty() {
            return None;
        }
        let clean_text = response_text.trim();
        let mut thought = String::new();
        let mut action_str = clean_text.to_string();

        if clean_text.contains("Thought:") {
            if let Some(idx) = clean_text.find("Action:") {
                thought = clean_text[..idx].replace("Thought:", "").trim().to_string();
                action_str = clean_text[idx..].to_string();
            } else {
                thought = clean_text.replace("Thought:", "").trim().to_string();
                action_str = clean_text.to_string();
            }
        }

        if let Some(idx) = action_str.find("Action:") {
            action_str = action_str[idx + "Action:".len()..].trim().to_string();
        }

        for handler in [
            Self::parse_finished,
            Self::parse_drag,
            Self::parse_scroll,
            Self::parse_coordinate_click,
            Self::parse_named_click,
            Self::parse_type,
            Self::parse_hotkey,
            Self::parse_wait,
        ] {
            if let Some(result) = handler(&thought, &action_str) {
                return Some(result);
            }
        }

        None
    }

    fn parse_coordinate_click(thought: &str, action_str: &str) -> Option<ParsedStep> {
        let re = Regex::new(&coord_pattern(
            "left_double_click|double_click|left_click|right_click|middle_click|click|hover|mouse_move",
        ))
        .unwrap();
        let m = re.find(action_str)?;
        let caps = re.captures(action_str)?;
        let x: i64 = caps.get(1)?.as_str().parse().ok()?;
        let y: i64 = caps.get(2)?.as_str().parse().ok()?;
        let verb = m.as_str().split('(').next().unwrap_or("").trim().to_lowercase();

        let (action, mut params, label) = match verb.as_str() {
            "left_double_click" | "double_click" => ("cua_click", json!({"x": x, "y": y, "click_count": 2}), "Double-clicking"),
            "right_click" => {
                let params = json!({"x": x, "y": y, "button": "right"});
                return Some(json!({
                    "thought": if thought.is_empty() { format!("Right-clicking at ({x}, {y})") } else { thought.to_string() },
                    "action": "cua_secondary_action",
                    "params": params,
                    "user_update": format!("Right-clicking at ({x}, {y})..."),
                    "is_completed": false,
                }));
            }
            "hover" | "mouse_move" => ("cua_click", json!({"x": x, "y": y, "click_count": 1}), "Moving to"),
            _ => ("cua_click", json!({"x": x, "y": y, "click_count": 1}), "Clicking"),
        };
        let _ = &mut params;

        Some(json!({
            "thought": if thought.is_empty() { format!("{label} target at ({x}, {y})") } else { thought.to_string() },
            "action": action,
            "params": params,
            "user_update": format!("{label} at ({x}, {y})..."),
            "is_completed": false,
        }))
    }

    fn parse_named_click(thought: &str, action_str: &str) -> Option<ParsedStep> {
        let re = Regex::new(
            r#"(?i)(?:left_double_click|double_click|right_click|click)\s*\(\s*(?:target|name|label|text|element)\s*=\s*['"]([^'"]+)['"]\s*\)"#,
        )
        .unwrap();
        let caps = re.captures(action_str)?;
        let target_name = caps.get(1)?.as_str().to_string();
        let full = caps.get(0)?.as_str();
        let verb = full.split('(').next().unwrap_or("").trim().to_lowercase();

        if verb == "right_click" {
            return Some(json!({
                "thought": if thought.is_empty() { format!("Right-clicking '{target_name}'") } else { thought.to_string() },
                "action": "cua_secondary_action",
                "params": {"target_name": target_name},
                "user_update": format!("Right-clicking '{target_name}'..."),
                "is_completed": false,
            }));
        }

        let mut params = json!({"target_name": target_name});
        if verb == "left_double_click" || verb == "double_click" {
            params["click_count"] = json!(2);
        }

        Some(json!({
            "thought": if thought.is_empty() { format!("Clicking '{target_name}'") } else { thought.to_string() },
            "action": "cua_click",
            "params": params,
            "user_update": format!("Clicking '{target_name}'..."),
            "is_completed": false,
        }))
    }

    fn parse_type(thought: &str, action_str: &str) -> Option<ParsedStep> {
        let re = Regex::new(r#"(?i)type\s*\(\s*(?:content|text|input|value)?\s*=?\s*['"]([^'"]*)['"]\s*\)"#).unwrap();
        let caps = re.captures(action_str)?;
        let text = caps.get(1)?.as_str().to_string();
        Some(json!({
            "thought": if thought.is_empty() { format!("Typing text '{text}'") } else { thought.to_string() },
            "action": "cua_type_text",
            "params": {"text": text.clone()},
            "user_update": format!("Typing '{}'...", &text.chars().take(40).collect::<String>()),
            "is_completed": false,
        }))
    }

    fn parse_hotkey(thought: &str, action_str: &str) -> Option<ParsedStep> {
        let re = Regex::new(r#"(?i)(?:hotkey|press_key|press|key)\s*\(\s*(?:(?:key|name|content)\s*=\s*)?['"]([^'"]+)['"]\s*\)"#).unwrap();
        let caps = re.captures(action_str)?;
        let mut key_name = caps.get(1)?.as_str().trim().to_string();
        if key_name.contains(' ') && !key_name.contains('+') {
            key_name = key_name.split_whitespace().collect::<Vec<_>>().join("+");
        }
        Some(json!({
            "thought": if thought.is_empty() { format!("Pressing key '{key_name}'") } else { thought.to_string() },
            "action": "cua_press_key",
            "params": {"key": key_name.clone()},
            "user_update": format!("Pressing '{key_name}'..."),
            "is_completed": false,
        }))
    }

    fn parse_scroll(thought: &str, action_str: &str) -> Option<ParsedStep> {
        if !Regex::new(r"(?i)\bscroll\s*\(").unwrap().is_match(action_str) {
            return None;
        }
        let (x, y) = BOX_RE
            .captures(action_str)
            .map(|c| {
                (
                    c.get(1).and_then(|m| m.as_str().parse::<i64>().ok()).unwrap_or(0),
                    c.get(2).and_then(|m| m.as_str().parse::<i64>().ok()).unwrap_or(0),
                )
            })
            .unwrap_or((0, 0));

        let direction = Regex::new(r#"(?i)direction\s*=\s*['"]?(up|down|left|right)['"]?"#)
            .unwrap()
            .captures(action_str)
            .and_then(|c| c.get(1).map(|m| m.as_str().to_lowercase()))
            .unwrap_or_else(|| "down".to_string());

        let (scroll_x, scroll_y) = match direction.as_str() {
            "down" => (0, 500),
            "up" => (0, -500),
            "right" => (500, 0),
            "left" => (-500, 0),
            _ => (0, 500),
        };

        Some(json!({
            "thought": if thought.is_empty() { format!("Scrolling {direction}") } else { thought.to_string() },
            "action": "cua_scroll",
            "params": {"x": x, "y": y, "scrollX": scroll_x, "scrollY": scroll_y},
            "user_update": format!("Scrolling {direction}..."),
            "is_completed": false,
        }))
    }

    fn parse_drag(thought: &str, action_str: &str) -> Option<ParsedStep> {
        if !Regex::new(r"(?i)\b(?:drag|select)\s*\(").unwrap().is_match(action_str) {
            return None;
        }
        let points: Vec<(i64, i64)> = BOX_RE
            .captures_iter(action_str)
            .filter_map(|c| {
                Some((
                    c.get(1)?.as_str().parse().ok()?,
                    c.get(2)?.as_str().parse().ok()?,
                ))
            })
            .collect();
        if points.len() < 2 {
            return None;
        }
        let (x1, y1) = points[0];
        let (x2, y2) = points[1];
        Some(json!({
            "thought": if thought.is_empty() { "Dragging between two points".to_string() } else { thought.to_string() },
            "action": "cua_drag",
            "params": {"from_x": x1, "from_y": y1, "to_x": x2, "to_y": y2},
            "user_update": "Dragging...",
            "is_completed": false,
        }))
    }

    fn parse_wait(thought: &str, action_str: &str) -> Option<ParsedStep> {
        if !Regex::new(r"(?i)\b(?:wait|sleep)\s*\(").unwrap().is_match(action_str) {
            return None;
        }
        Some(json!({
            "thought": if thought.is_empty() { "Waiting for the screen to update".to_string() } else { thought.to_string() },
            "action": "cua_screenshot",
            "params": {},
            "user_update": "Waiting...",
            "is_completed": false,
        }))
    }

    fn parse_finished(thought: &str, action_str: &str) -> Option<ParsedStep> {
        let re = Regex::new(r#"(?i)(?:finished|complete|stop)\s*\(\s*(?:(?:response|message|content)\s*=\s*)?['"]?([^'"]*)['"]?\s*\)"#).unwrap();
        let caps = re.captures(action_str);
        let lower = action_str.to_lowercase();
        if caps.is_none() && !lower.contains("finished()") && !lower.contains("completed") {
            return None;
        }

        let response_msg = caps
            .as_ref()
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Goal completed.".to_string());

        Some(json!({
            "thought": if thought.is_empty() { "Task execution complete".to_string() } else { thought.to_string() },
            "action": "converse",
            "params": {"response": response_msg.clone()},
            "final_response": response_msg,
            "user_update": "Task completed.",
            "is_completed": true,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_box_token_click() {
        let step = UiTarsParser::parse_response(
            "Thought: clicking the button\nAction: click(start_box='<|box_start|>(345,678)<|box_end|>')",
        )
        .unwrap();
        assert_eq!(step["action"], "cua_click");
        assert_eq!(step["params"]["x"], 345);
        assert_eq!(step["params"]["y"], 678);
        assert_eq!(step["thought"], "clicking the button");
    }

    #[test]
    fn right_click_maps_to_secondary_action() {
        let step = UiTarsParser::parse_response("Action: right_click(point='(10,20)')").unwrap();
        assert_eq!(step["action"], "cua_secondary_action");
        assert_eq!(step["params"]["x"], 10);
    }

    #[test]
    fn double_click_sets_click_count_two() {
        let step = UiTarsParser::parse_response("Action: double_click(start_box='(1,2)')").unwrap();
        assert_eq!(step["action"], "cua_click");
        assert_eq!(step["params"]["click_count"], 2);
    }

    #[test]
    fn named_click_without_coordinates() {
        let step = UiTarsParser::parse_response("Action: click(target='Submit button')").unwrap();
        assert_eq!(step["action"], "cua_click");
        assert_eq!(step["params"]["target_name"], "Submit button");
    }

    #[test]
    fn type_action_parses_content() {
        let step = UiTarsParser::parse_response("Action: type(content='hello world')").unwrap();
        assert_eq!(step["action"], "cua_type_text");
        assert_eq!(step["params"]["text"], "hello world");
    }

    #[test]
    fn hotkey_with_space_becomes_plus_joined() {
        let step = UiTarsParser::parse_response("Action: hotkey('ctrl a')").unwrap();
        assert_eq!(step["action"], "cua_press_key");
        assert_eq!(step["params"]["key"], "ctrl+a");
    }

    #[test]
    fn scroll_defaults_to_down() {
        let step = UiTarsParser::parse_response("Action: scroll(start_box='(100,200)')").unwrap();
        assert_eq!(step["action"], "cua_scroll");
        assert_eq!(step["params"]["scrollY"], 500);
    }

    #[test]
    fn scroll_up_is_negative() {
        let step = UiTarsParser::parse_response("Action: scroll(start_box='(0,0)', direction='up')").unwrap();
        assert_eq!(step["params"]["scrollY"], -500);
    }

    #[test]
    fn drag_needs_two_points() {
        let step = UiTarsParser::parse_response("Action: drag(start_box='(1,2)', end_box='(3,4)')").unwrap();
        assert_eq!(step["action"], "cua_drag");
        assert_eq!(step["params"]["from_x"], 1);
        assert_eq!(step["params"]["to_x"], 3);
    }

    #[test]
    fn wait_maps_to_a_no_op_screenshot() {
        let step = UiTarsParser::parse_response("Action: wait()").unwrap();
        assert_eq!(step["action"], "cua_screenshot");
        assert_eq!(step["is_completed"], false);
    }

    #[test]
    fn finished_marks_completion_with_response() {
        let step = UiTarsParser::parse_response("Action: finished(content='All done here')").unwrap();
        assert_eq!(step["action"], "converse");
        assert_eq!(step["is_completed"], true);
        assert_eq!(step["final_response"], "All done here");
    }

    #[test]
    fn unrecognized_text_returns_none() {
        assert!(UiTarsParser::parse_response("gibberish with no action syntax at all").is_none());
    }

    #[test]
    fn empty_input_returns_none() {
        assert!(UiTarsParser::parse_response("").is_none());
    }
}
