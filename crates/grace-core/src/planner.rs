//! Ported from `src/grace/agent/planner.py`: decides *what* to do next,
//! from structure (the element graph) rather than pixels.

use crate::intent::clean_json_fence;
use crate::models::{LargeLanguageModel, LlmRequest, RateLimitError};
use crate::tools::format_tools_for_prompt;
use serde_json::Value;
use std::collections::BTreeMap;

/// One decision from the planner.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlannedStep {
    pub thought: String,
    pub action: String,
    pub params: Value,
    pub expect: String,
    pub is_completed: bool,
    pub final_response: String,
    pub user_update: String,
}

impl PlannedStep {
    /// True when the planner named a target it could not pin to an
    /// element. This is the only condition under which the grounder is
    /// invoked.
    pub fn needs_grounding(&self) -> bool {
        if !matches!(self.action.as_str(), "cua_click" | "cua_secondary_action" | "cua_set_value") {
            return false;
        }
        if self.params.get("element_id").map(|v| !v.is_null()).unwrap_or(false) {
            return false;
        }
        let has_x = self.params.get("x").map(|v| !v.is_null()).unwrap_or(false);
        let has_y = self.params.get("y").map(|v| !v.is_null()).unwrap_or(false);
        if has_x && has_y {
            return false;
        }
        let target_name = self.params.get("target_name").and_then(Value::as_str).filter(|s| !s.is_empty());
        let describe = self.params.get("describe").and_then(Value::as_str).filter(|s| !s.is_empty());
        target_name.is_some() || describe.is_some()
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "thought": self.thought,
            "action": self.action,
            "params": self.params,
            "expect": self.expect,
            "is_completed": self.is_completed,
            "final_response": self.final_response,
            "user_update": self.user_update,
        })
    }
}

/// The per-goal call cap was hit. The loop finishes with what it knows.
#[derive(Debug, Clone, thiserror::Error)]
#[error("Planner reached its {0}-call limit for this goal")]
pub struct PlannerBudgetExceeded(pub u32);

pub fn get_planner_system_prompt() -> String {
    format!(
        r#"You are the planner for Grace, a voice assistant that operates Windows 11 for a user who cannot use a keyboard or mouse.

You are given the user's goal, what is currently on screen, what you have already done, and whether the last step did what you expected. You decide the single next step.

Tools:
{}

The screen is described to you in one of two ways, and you will always get exactly one of them.

**An element list.** A JSON array of the controls the app reports, each with an `id`. Target them with `element_id`. This is exact - the id you name is the control that gets clicked.

**A marked screenshot.** Some apps report nothing useful to Windows, so instead you get an image with a red numbered badge on every target, and a legend naming them. The badge numbers are element ids: answer with `element_id` exactly as you would from a list. Read the image to decide *which* number; read the legend to check what it is.

You cannot click a position. There is no `x`/`y` - a coordinate you reason out from a picture lands somewhere arbitrary, and this is the single most common way a goal has failed. If what you need has no id and no badge:
- Use `target_name` to describe it in plain words ("the search box", "the Chemistry group in the chat list"). That hands it to a visual model that locates it properly.
- Or use the keyboard. Shortcuts and typing go to the focused window whatever the screen description shows, and are often the most reliable route in an app you cannot read.

About `frame`:
- `"chrome"` is the browser's own UI: address bar, tabs, bookmarks, back button.
- `"page"` is the content of the website itself.
- `"app"` is a normal desktop application.
A website's own search box is ALWAYS `frame: "page"`. The browser address bar is ALWAYS `frame: "chrome"`. Typing a site's search query into the address bar is a mistake - it searches the web instead of the site.

Rules:
- One step per response. Do not plan several actions at once.
- Before typing, make sure the field you want is focused - click it first.
- `expect` must describe something you will be able to *see* in the next screen description, e.g. "the YouTube search box is focused" or "video result links are listed".
- Set `is_completed: true` only when the screen description or window title shows the goal is actually achieved. Put the spoken answer in `final_response`, in plain sentences with no JSON or markdown.
- If the last step reports it did not do what you expected, do something different. Do not repeat the same failing action.

Respond with ONLY one JSON object, no code fences, no commentary:
{{"thought": "why this step", "action": "<tool name>", "params": {{...}}, "expect": "what should be true next", "is_completed": false, "user_update": "short phrase shown to the user", "final_response": ""}}"#,
        format_tools_for_prompt()
    )
}

pub fn build_prompt(
    goal: &str,
    elements_prompt: &str,
    history: &str,
    scratchpad: &str,
    expectation_note: &str,
    window_title: &str,
) -> String {
    let mut sections = vec![format!("### Goal\n{goal}")];
    if !window_title.is_empty() {
        sections.push(format!("### Active window\n{window_title}"));
    }
    sections.push(format!("### What is on screen\n{elements_prompt}"));
    if !history.is_empty() {
        sections.push(format!("### What you have already done\n{history}"));
    }
    if !scratchpad.is_empty() {
        sections.push(format!("### Data collected so far\n{scratchpad}"));
    }
    if !expectation_note.is_empty() {
        sections.push(format!("### Result of your last step\n{expectation_note}"));
    }
    sections.push("Decide the single next step. Respond with one JSON object only.".to_string());
    sections.join("\n\n")
}

/// Arguments for one `Planner::plan` call, grouped because the parameter
/// list otherwise repeats verbatim at every call site (as it does in
/// Python).
#[derive(Debug, Clone, Default)]
pub struct PlanArgs {
    pub goal: String,
    pub elements_prompt: String,
    pub history: String,
    pub scratchpad: String,
    pub expectation_note: String,
    pub window_title: String,
    pub image_b64: Option<String>,
    /// `None` for the configured planner model; `Some` for the escalation
    /// ladder's third rung.
    pub model: Option<String>,
}

/// Wraps the LLM with a per-goal call budget and JSON parsing.
pub struct Planner<'a> {
    llm: &'a mut dyn LargeLanguageModel,
    max_calls: Option<u32>,
    calls: u32,
    system_prompt: String,
}

impl<'a> Planner<'a> {
    /// `max_calls = 0` (or `None`) disables the cap entirely, matching
    /// `DEFAULT_MAX_CALLS = 0` in Python.
    pub fn new(llm: &'a mut dyn LargeLanguageModel, max_calls: u32) -> Self {
        Self {
            llm,
            max_calls: if max_calls > 0 { Some(max_calls) } else { None },
            calls: 0,
            system_prompt: get_planner_system_prompt(),
        }
    }

    pub fn is_unlimited(&self) -> bool {
        self.max_calls.is_none()
    }

    pub fn calls_made(&self) -> u32 {
        self.calls
    }

    pub fn calls_remaining(&self) -> Option<u32> {
        self.max_calls.map(|max| max.saturating_sub(self.calls))
    }

    pub fn reset(&mut self) {
        self.calls = 0;
    }

    /// Ask for the next step. Returns `Ok(None)` if the model gave nothing
    /// usable. `Err(PlannerBudgetExceeded)` only when a positive per-goal
    /// cap has been configured and is spent; `Err`s the rate limit straight
    /// through so the loop can tell the user.
    pub fn plan(&mut self, args: &PlanArgs) -> Result<Option<PlannedStep>, PlanError> {
        if let Some(max) = self.max_calls {
            if self.calls >= max {
                return Err(PlanError::BudgetExceeded(PlannerBudgetExceeded(max)));
            }
        }

        let prompt = build_prompt(
            &args.goal,
            &args.elements_prompt,
            &args.history,
            &args.scratchpad,
            &args.expectation_note,
            &args.window_title,
        );

        self.calls += 1;
        let request = LlmRequest {
            prompt,
            system_prompt: Some(self.system_prompt.clone()),
            image_b64: args.image_b64.clone(),
            model: args.model.clone(),
            temperature: 0.1,
            max_tokens: 8192,
        };

        let raw = self.llm.generate_text(&request).map_err(PlanError::RateLimited)?;
        let Some(raw) = raw else {
            return Ok(None);
        };

        Ok(parse_planned_step(&raw))
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum PlanError {
    #[error(transparent)]
    BudgetExceeded(#[from] PlannerBudgetExceeded),
    #[error(transparent)]
    RateLimited(#[from] RateLimitError),
}

/// Parse the planner's JSON, tolerating fences and surrounding prose.
pub fn parse_planned_step(text: &str) -> Option<PlannedStep> {
    if text.is_empty() {
        return None;
    }

    let cleaned = clean_json_fence(text);
    let data = load_json_object(&cleaned).or_else(|| {
        let start = text.find('{')?;
        let end = text.rfind('}')?;
        if end > start {
            load_json_object(&text[start..=end])
        } else {
            None
        }
    })?;

    let action = data.get("action").and_then(Value::as_str).filter(|s| !s.trim().is_empty())?;
    let action = action.trim().to_string();

    let params = data
        .get("params")
        .filter(|v| v.is_object())
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));

    let mut final_response = data
        .get("final_response")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if final_response.is_empty() && action == "converse" {
        if let Some(response) = params.get("response").and_then(Value::as_str) {
            final_response = response.to_string();
        }
    }

    Some(PlannedStep {
        thought: data.get("thought").and_then(Value::as_str).unwrap_or("").to_string(),
        action: action.clone(),
        params,
        expect: data.get("expect").and_then(Value::as_str).unwrap_or("").to_string(),
        is_completed: data.get("is_completed").and_then(Value::as_bool).unwrap_or(false),
        final_response,
        user_update: data
            .get("user_update")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("Running {action}…")),
    })
}

fn load_json_object(candidate: &str) -> Option<Value> {
    let value: Value = serde_json::from_str(candidate).ok()?;
    value.is_object().then_some(value)
}

/// A crude accounting map for tests that want to inspect call counts by
/// model name without a full mock framework.
pub type ModelCallCounts = BTreeMap<String, u32>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ScriptedLlm;
    use serde_json::json;

    #[test]
    fn parses_a_clean_json_response() {
        let step = parse_planned_step(
            r#"{"thought": "click it", "action": "cua_click", "params": {"element_id": 3}, "expect": "clicked", "is_completed": false}"#,
        )
        .unwrap();
        assert_eq!(step.action, "cua_click");
        assert_eq!(step.params, json!({"element_id": 3}));
    }

    #[test]
    fn parses_json_wrapped_in_a_fence_with_prose() {
        let step = parse_planned_step(
            "Sure, here you go:\n```json\n{\"action\": \"converse\", \"params\": {\"response\": \"hi\"}}\n```\nHope that helps!",
        );
        // Fenced text is stripped by clean_json_fence first; if that alone
        // doesn't parse, the {…} substring fallback catches the rest.
        assert!(step.is_some());
        assert_eq!(step.unwrap().action, "converse");
    }

    #[test]
    fn missing_action_is_not_a_step() {
        assert!(parse_planned_step(r#"{"thought": "hmm"}"#).is_none());
    }

    #[test]
    fn empty_text_is_not_a_step() {
        assert!(parse_planned_step("").is_none());
    }

    #[test]
    fn converse_without_final_response_falls_back_to_params_response() {
        let step = parse_planned_step(r#"{"action": "converse", "params": {"response": "All set."}}"#).unwrap();
        assert_eq!(step.final_response, "All set.");
    }

    #[test]
    fn needs_grounding_true_only_for_a_named_unresolved_click_target() {
        let mut step = PlannedStep {
            action: "cua_click".into(),
            params: json!({"target_name": "the search box"}),
            ..Default::default()
        };
        assert!(step.needs_grounding());

        step.params = json!({"element_id": 5});
        assert!(!step.needs_grounding());

        step.params = json!({"x": 10, "y": 20});
        assert!(!step.needs_grounding());

        step.action = "open_app".into();
        step.params = json!({"target_name": "x"});
        assert!(!step.needs_grounding());
    }

    #[test]
    fn planner_enforces_a_positive_call_budget() {
        let mut llm = ScriptedLlm::new(vec![ScriptedLlm::text(r#"{"action": "converse", "params": {}}"#)]);
        let mut planner = Planner::new(&mut llm, 1);
        let args = PlanArgs { goal: "test".into(), ..Default::default() };
        assert!(planner.plan(&args).unwrap().is_some());
        assert!(matches!(planner.plan(&args), Err(PlanError::BudgetExceeded(_))));
    }

    #[test]
    fn zero_budget_means_unlimited() {
        let mut llm = ScriptedLlm::new(vec![
            ScriptedLlm::text(r#"{"action": "converse", "params": {}}"#),
            ScriptedLlm::text(r#"{"action": "converse", "params": {}}"#),
        ]);
        let mut planner = Planner::new(&mut llm, 0);
        assert!(planner.is_unlimited());
        let args = PlanArgs { goal: "test".into(), ..Default::default() };
        assert!(planner.plan(&args).unwrap().is_some());
        assert!(planner.plan(&args).unwrap().is_some());
    }

    #[test]
    fn an_empty_llm_response_is_not_an_error() {
        let mut llm = ScriptedLlm::new(vec![Ok(None)]);
        let mut planner = Planner::new(&mut llm, 0);
        let args = PlanArgs { goal: "test".into(), ..Default::default() };
        assert_eq!(planner.plan(&args).unwrap(), None);
    }

    #[test]
    fn rate_limit_propagates_as_an_error() {
        let mut llm = ScriptedLlm::new(vec![Err(RateLimitError("slow down".into()))]);
        let mut planner = Planner::new(&mut llm, 0);
        let args = PlanArgs { goal: "test".into(), ..Default::default() };
        assert!(matches!(planner.plan(&args), Err(PlanError::RateLimited(_))));
    }

    #[test]
    fn reset_clears_the_call_count() {
        let mut llm = ScriptedLlm::new(vec![ScriptedLlm::text(r#"{"action": "converse", "params": {}}"#)]);
        let mut planner = Planner::new(&mut llm, 1);
        let args = PlanArgs { goal: "test".into(), ..Default::default() };
        planner.plan(&args).unwrap();
        assert_eq!(planner.calls_made(), 1);
        planner.reset();
        assert_eq!(planner.calls_made(), 0);
    }
}
