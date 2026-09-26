//! Ported from `src/grace/agent/grounder.py`: turns a described target into
//! screen coordinates via a vision-capable `LargeLanguageModel`, using
//! UI-TARS's native Thought/Action action space rather than Grace's tool
//! schema.

use crate::models::{LargeLanguageModel, LlmRequest};
use crate::ui_tars_parser::UiTarsParser;
use serde_json::Value;

/// The canonical UI-TARS grounding instruction. Deliberately not Grace's
/// tool schema: the model was trained against this action space.
fn grounder_system_prompt(width: u32, height: u32) -> String {
    format!(
        r#"You are a GUI grounding model. You are shown a screenshot and asked to locate a single element.

Respond in exactly this format and nothing else:

Thought: <one short sentence identifying the element>
Action: click(start_box='(x,y)')

x and y are pixel coordinates in the image you were given, measured from its top-left corner, where x is between 0 and {width} and y is between 0 and {height}. Do not normalise them to any other range. Point at the centre of the element. If the element is not visible in the screenshot, respond with Action: wait()."#
    )
}

/// A located target, in screen coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundedPoint {
    pub x: i64,
    pub y: i64,
    pub thought: String,
}

pub struct Grounder<'a> {
    llm: &'a mut dyn LargeLanguageModel,
    pub calls_made: u32,
}

impl<'a> Grounder<'a> {
    pub fn new(llm: &'a mut dyn LargeLanguageModel) -> Self {
        Self { llm, calls_made: 0 }
    }

    /// Find `description` in the screenshot and return screen coordinates.
    ///
    /// `image_size` is the size of the PNG actually sent; `screen_size` is
    /// the real desktop. The model answers in image space, so the result
    /// must be scaled back.
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
            system_prompt: Some(grounder_system_prompt(image_size.0, image_size.1)),
            image_b64: Some(base64_encode(png_bytes)),
            model: None,
            temperature: 0.0,
            max_tokens: 2048,
        };

        let raw = self.llm.generate_text(&request).ok()??;
        let parsed = UiTarsParser::parse_response(&raw)?;

        let x = parsed["params"].get("x").and_then(Value::as_i64)?;
        let y = parsed["params"].get("y").and_then(Value::as_i64)?;

        let (sx, sy) = scale_to_screen(x, y, image_size, screen_size);
        Some(GroundedPoint {
            x: sx,
            y: sy,
            thought: parsed.get("thought").and_then(Value::as_str).unwrap_or("").to_string(),
        })
    }
}

/// Map a coordinate in the sent image back to the physical screen.
///
/// The overflow branch is kept only as a net for a model that ignores the
/// instruction outright and emits normalised 0-1000 coordinates instead of
/// pixels.
pub fn scale_to_screen(x: i64, y: i64, image_size: (u32, u32), screen_size: (u32, u32)) -> (i64, i64) {
    let (img_w, img_h) = (image_size.0 as i64, image_size.1 as i64);
    let (scr_w, scr_h) = (screen_size.0 as i64, screen_size.1 as i64);
    if img_w == 0 || img_h == 0 || scr_w == 0 || scr_h == 0 {
        return (x, y);
    }

    if x > img_w || y > img_h {
        return (
            (x as f64 * scr_w as f64 / 1000.0).round() as i64,
            (y as f64 * scr_h as f64 / 1000.0).round() as i64,
        );
    }

    (
        (x as f64 * scr_w as f64 / img_w as f64).round() as i64,
        (y as f64 * scr_h as f64 / img_h as f64).round() as i64,
    )
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

/// Minimal base64 encoder (standard alphabet, no external dependency for
/// something this small and hot-path-irrelevant - grounding is a rare call).
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
    fn scale_to_screen_maps_pixel_coordinates_proportionally() {
        assert_eq!(scale_to_screen(640, 360, (1280, 720), (1920, 1080)), (960, 540));
    }

    #[test]
    fn scale_to_screen_zero_size_is_a_no_op() {
        assert_eq!(scale_to_screen(10, 20, (0, 0), (1920, 1080)), (10, 20));
    }

    #[test]
    fn scale_to_screen_out_of_bounds_is_treated_as_normalised() {
        // (1500, 800) exceeds a 1280x720 image, so it's read as 0-1000 space.
        let (x, y) = scale_to_screen(1500, 800, (1280, 720), (1920, 1080));
        assert_eq!(x, (1500.0_f64 * 1920.0 / 1000.0).round() as i64);
        assert_eq!(y, (800.0_f64 * 1080.0 / 1000.0).round() as i64);
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
            "Thought: found it\nAction: click(start_box='(100,100)')",
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
