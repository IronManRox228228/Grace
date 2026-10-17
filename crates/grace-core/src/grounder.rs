//! Ported from `src/grace/agent/grounder.py`: turns a described target into
//! screen coordinates via a vision-capable `LargeLanguageModel`.

use crate::models::{LargeLanguageModel, LlmRequest};
use regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;

/// Canonical vision grounding instruction for any vision model.
pub fn grounder_system_prompt() -> &'static str {
    "You are a GUI visual grounding assistant. You are shown a screenshot and asked to locate a single element.\n\
Locate the requested element and return its center coordinates.\n\
Respond in this format:\n\
Thought: <brief description of where the element is>\n\
Action: click(point=(x, y))\n\n\
If the element is not visible or cannot be found, respond with:\n\
Action: wait()"
}

/// Coordinate formats supported by the vision grounder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoordinateFormat {
    #[default]
    Norm1000,
    Norm1,
    Percent,
    Pixels,
}

impl CoordinateFormat {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "0-1000" | "norm_1000" | "1000" => Self::Norm1000,
            "0-1" | "norm_1" | "normalized" => Self::Norm1,
            "percent" | "percentage" | "0-100" | "100" => Self::Percent,
            "pixels" | "pixel" | "image_pixels" => Self::Pixels,
            _ => Self::Norm1000,
        }
    }
}


/// A located target, in screen coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundedPoint {
    pub x: i64,
    pub y: i64,
    pub thought: String,
}

static WAIT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\bwait\s*\(\s*\)").unwrap());
static CLICK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:click|move|point)\s*\(\s*(?:(?:point|start_box|target|location|coord|coordinates)\s*=\s*)?['\x22]?(?:<\|box_start\|>)?\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*(?:<\|box_end\|>)?['\x22]?\s*\)",
    )
    .unwrap()
});
static BOX_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)<\|box_start\|>\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*<\|box_end\|>").unwrap()
});
static COORD_PREFIX_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:point|coord|coordinate|coordinates|location)\s*[:=]\s*[\(\[]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\)\]]").unwrap()
});
static ACTION_BARE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)Action:\s*(?:click\s*)?[\(\[]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\)\]]").unwrap()
});

/// Parse raw text from a vision grounder into `(x, y, thought)`.
pub fn parse_grounder_response(response_text: &str) -> Option<(f64, f64, String)> {
    if response_text.is_empty() {
        return None;
    }
    let clean = response_text.trim();
    if clean.is_empty() {
        return None;
    }

    let mut thought = String::new();
    if clean.contains("Thought:") {
        if let Some(idx) = clean.find("Action:") {
            thought = clean[..idx].replace("Thought:", "").trim().to_string();
        } else if let Some(idx) = clean.find('\n') {
            thought = clean[..idx].replace("Thought:", "").trim().to_string();
        } else {
            thought = clean.replace("Thought:", "").trim().to_string();
        }
    }

    if WAIT_RE.is_match(clean) {
        return None;
    }

    // Try parsing JSON if braces present
    if let (Some(open), Some(close)) = (clean.find('{'), clean.rfind('}')) {
        if open < close {
            if let Ok(val) = serde_json::from_str::<Value>(&clean[open..=close]) {
                if let Some(obj) = val.as_object() {
                    let cand_thought = obj.get("thought").and_then(Value::as_str).unwrap_or(&thought).to_string();
                    if let Some(pt) = obj.get("point").or_else(|| obj.get("coordinates")).or_else(|| obj.get("click")).and_then(Value::as_array) {
                        if pt.len() >= 2 {
                            if let (Some(x), Some(y)) = (pt[0].as_f64(), pt[1].as_f64()) {
                                return Some((x, y, cand_thought));
                            }
                        }
                    }
                    if let (Some(x), Some(y)) = (obj.get("x").and_then(Value::as_f64), obj.get("y").and_then(Value::as_f64)) {
                        return Some((x, y, cand_thought));
                    }
                }
            }
        }
    }

    if let Some(caps) = CLICK_RE.captures(clean) {
        if let (Some(x), Some(y)) = (caps.get(1)?.as_str().parse().ok(), caps.get(2)?.as_str().parse().ok()) {
            return Some((x, y, thought));
        }
    }

    if let Some(caps) = BOX_RE.captures(clean) {
        if let (Some(x), Some(y)) = (caps.get(1)?.as_str().parse().ok(), caps.get(2)?.as_str().parse().ok()) {
            return Some((x, y, thought));
        }
    }

    if let Some(caps) = COORD_PREFIX_RE.captures(clean) {
        if let (Some(x), Some(y)) = (caps.get(1)?.as_str().parse().ok(), caps.get(2)?.as_str().parse().ok()) {
            return Some((x, y, thought));
        }
    }

    if let Some(caps) = ACTION_BARE_RE.captures(clean) {
        if let (Some(x), Some(y)) = (caps.get(1)?.as_str().parse().ok(), caps.get(2)?.as_str().parse().ok()) {
            return Some((x, y, thought));
        }
    }

    None
}

pub struct Grounder<'a> {
    llm: &'a mut dyn LargeLanguageModel,
    pub coordinate_format: CoordinateFormat,
    pub calls_made: u32,
}

impl<'a> Grounder<'a> {
    pub fn new(llm: &'a mut dyn LargeLanguageModel) -> Self {
        Self {
            llm,
            coordinate_format: CoordinateFormat::Norm1000,
            calls_made: 0,
        }
    }

    pub fn with_coordinate_format(mut self, format: CoordinateFormat) -> Self {
        self.coordinate_format = format;
        self
    }

    /// Find `description` in the screenshot and return screen coordinates.
    pub fn locate(
        &mut self,
        description: &str,
        png_bytes: &[u8],
        image_size: (u32, u32),
        screen_size: (u32, u32),
    ) -> Option<GroundedPoint> {
        if png_bytes.is_empty() || description.is_empty() {
            return None;
        }

        self.calls_made += 1;
        let request = LlmRequest {
            prompt: format!("Locate this element: {description}"),
            system_prompt: Some(grounder_system_prompt().to_string()),
            image_b64: Some(base64_encode(png_bytes)),
            model: None,
            temperature: 0.0,
            max_tokens: 2048,
        };

        let raw = self.llm.generate_text(&request).ok()??;
        let (x, y, thought) = parse_grounder_response(&raw)?;
        let (sx, sy) = scale_to_screen(x, y, image_size, screen_size, self.coordinate_format)?;

        Some(GroundedPoint { x: sx, y: sy, thought })
    }
}

/// Map a coordinate from the model back to the physical screen.
pub fn scale_to_screen(
    x: f64,
    y: f64,
    image_size: (u32, u32),
    screen_size: (u32, u32),
    format: CoordinateFormat,
) -> Option<(i64, i64)> {
    let (img_w, img_h) = (image_size.0 as f64, image_size.1 as f64);
    let (scr_w, scr_h) = (screen_size.0 as f64, screen_size.1 as f64);
    if img_w == 0.0 || img_h == 0.0 || scr_w == 0.0 || scr_h == 0.0 {
        return Some((x.round() as i64, y.round() as i64));
    }

    match format {
        CoordinateFormat::Norm1000 => {
            if x < 0.0 || x > 1000.0 || y < 0.0 || y > 1000.0 {
                return None;
            }
            Some(((x * scr_w / 1000.0).round() as i64, (y * scr_h / 1000.0).round() as i64))
        }
        CoordinateFormat::Norm1 => {
            if x < 0.0 || x > 1.0 || y < 0.0 || y > 1.0 {
                return None;
            }
            Some(((x * scr_w).round() as i64, (y * scr_h).round() as i64))
        }
        CoordinateFormat::Percent => {
            if x < 0.0 || x > 100.0 || y < 0.0 || y > 100.0 {
                return None;
            }
            Some(((x * scr_w / 100.0).round() as i64, (y * scr_h / 100.0).round() as i64))
        }
        CoordinateFormat::Pixels => {

            if x < 0.0 || y < 0.0 {
                return None;
            }
            if x > img_w || y > img_h {
                // Overflow net: if model emitted 0-1000 coordinates
                if (0.0..=1000.0).contains(&x) && (0.0..=1000.0).contains(&y) {
                    return Some(((x * scr_w / 1000.0).round() as i64, (y * scr_h / 1000.0).round() as i64));
                }
                return None;
            }
            Some(((x * scr_w / img_w).round() as i64, (y * scr_h / img_h).round() as i64))
        }
    }
}

/// Build a natural-language description from the planner's params.
pub fn describe_target(params: &Value) -> String {
    if let Some(described) = params.get("describe").and_then(Value::as_str) {
        if !described.is_empty() {
            return described.to_string();
        }
    }

    let name = params
        .get("target_name")
        .and_then(Value::as_str)
        .or_else(|| params.get("name").and_then(Value::as_str))
        .unwrap_or("");
    let role = params.get("role").and_then(Value::as_str).unwrap_or("");
    let frame = params.get("frame").and_then(Value::as_str).unwrap_or("");

    let mut parts = Vec::new();
    if !role.is_empty() {
        parts.push(role.to_string());
    }
    if !name.is_empty() {
        parts.push(format!("labelled '{name}'"));
    }
    match frame {
        "chrome" => parts.push("in the browser's toolbar".to_string()),
        "page" => parts.push("in the web page content".to_string()),
        _ => {}
    }

    if parts.is_empty() {
        name.to_string()
    } else {
        parts.join(" ")
    }
}

/// Minimal base64 encoder.
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | (b2 as u32);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 { ALPHABET[((n >> 6) & 0x3F) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { ALPHABET[(n & 0x3F) as usize] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ScriptedLlm;
    use serde_json::json;

    #[test]
    fn parse_grounder_response_action_click_point() {
        let (x, y, thought) = parse_grounder_response("Thought: found button\nAction: click(point=(469, 472))").unwrap();
        assert_eq!(x, 469.0);
        assert_eq!(y, 472.0);
        assert_eq!(thought, "found button");
    }

    #[test]
    fn parse_grounder_response_click_tuple() {
        let (x, y, _) = parse_grounder_response("Action: click((100, 200))").unwrap();
        assert_eq!(x, 100.0);
        assert_eq!(y, 200.0);
    }

    #[test]
    fn parse_grounder_response_json() {
        let (x, y, thought) = parse_grounder_response(r#"{"thought": "located", "point": [500, 500]}"#).unwrap();
        assert_eq!(x, 500.0);
        assert_eq!(y, 500.0);
        assert_eq!(thought, "located");
    }

    #[test]
    fn parse_grounder_response_wait_returns_none() {
        assert!(parse_grounder_response("Action: wait()").is_none());
        assert!(parse_grounder_response("Thought: not found\nAction: wait()").is_none());
    }

    #[test]
    fn parse_grounder_response_malformed_returns_none() {
        assert!(parse_grounder_response("I cannot locate that element.").is_none());
        assert!(parse_grounder_response("").is_none());
    }

    #[test]
    fn scale_to_screen_norm_1000() {
        let res = scale_to_screen(500.0, 500.0, (1280, 720), (1920, 1080), CoordinateFormat::Norm1000).unwrap();
        assert_eq!(res, (960, 540));
    }

    #[test]
    fn scale_to_screen_norm_1() {
        let res = scale_to_screen(0.5, 0.5, (1280, 720), (1920, 1080), CoordinateFormat::Norm1).unwrap();
        assert_eq!(res, (960, 540));
    }

    #[test]
    fn scale_to_screen_percent() {
        let res = scale_to_screen(50.0, 50.0, (1280, 720), (1920, 1080), CoordinateFormat::Percent).unwrap();
        assert_eq!(res, (960, 540));
    }

    #[test]
    fn scale_to_screen_pixels() {
        let res = scale_to_screen(640.0, 360.0, (1280, 720), (1920, 1080), CoordinateFormat::Pixels).unwrap();
        assert_eq!(res, (960, 540));
    }

    #[test]
    fn scale_to_screen_out_of_bounds_returns_none() {
        assert!(scale_to_screen(-5.0, 500.0, (1280, 720), (1920, 1080), CoordinateFormat::Norm1000).is_none());
        assert!(scale_to_screen(1050.0, 500.0, (1280, 720), (1920, 1080), CoordinateFormat::Norm1000).is_none());
        assert!(scale_to_screen(1.2, 0.5, (1280, 720), (1920, 1080), CoordinateFormat::Norm1).is_none());
        assert!(scale_to_screen(-1.0, 50.0, (1280, 720), (1920, 1080), CoordinateFormat::Percent).is_none());
        assert!(scale_to_screen(105.0, 50.0, (1280, 720), (1920, 1080), CoordinateFormat::Percent).is_none());
        assert!(scale_to_screen(2500.0, 500.0, (1280, 720), (1920, 1080), CoordinateFormat::Pixels).is_none());
    }


    #[test]
    fn describe_target_prefers_an_explicit_describe_field() {
        assert_eq!(describe_target(&json!({"describe": "the red button"})), "the red button");
    }

    #[test]
    fn describe_target_builds_from_role_name_and_frame() {
        assert_eq!(
            describe_target(&json!({"role": "button", "target_name": "Send", "frame": "page"})),
            "button labelled 'Send' in the web page content"
        );
    }

    #[test]
    fn describe_target_falls_back_to_bare_name() {
        assert_eq!(describe_target(&json!({"target_name": "Send"})), "labelled 'Send'");
        assert_eq!(describe_target(&json!({})), "");
    }

    #[test]
    fn locate_returns_none_for_empty_png_or_description() {
        let mut llm = ScriptedLlm::new(vec![]);
        let mut grounder = Grounder::new(&mut llm);
        assert!(grounder.locate("", &[1, 2, 3], (100, 100), (100, 100)).is_none());
        assert!(grounder.locate("a button", &[], (100, 100), (100, 100)).is_none());
    }

    #[test]
    fn locate_parses_and_scales_a_coordinate_response() {
        let mut llm = ScriptedLlm::new(vec![ScriptedLlm::text(
            "Thought: found it\nAction: click(point=(500,500))",
        )]);
        let mut grounder = Grounder::new(&mut llm);
        let point = grounder
            .locate("the button", &[1, 2, 3], (200, 200), (400, 400))
            .unwrap();
        assert_eq!(point.x, 200);
        assert_eq!(point.y, 200);
        assert_eq!(grounder.calls_made, 1);
    }

    #[test]
    fn locate_returns_none_when_the_model_cannot_see_the_element() {
        let mut llm = ScriptedLlm::new(vec![ScriptedLlm::text("Action: wait()")]);
        let mut grounder = Grounder::new(&mut llm);
        assert!(grounder.locate("the button", &[1, 2, 3], (200, 200), (400, 400)).is_none());
    }
}
