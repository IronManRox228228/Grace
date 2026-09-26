//! Ported from `src/grace/agent/loop.py`: observe -> plan -> (ground) ->
//! act -> verify, until the goal is done.
//!
//! Every ship-blocker/review-backlog behaviour PLAN.md §10.1 items 3, 8 and
//! `contract/README.md`'s two findings describe is ported deliberately,
//! not incidentally: the 30s pending-confirmation TTL, the completion guard
//! that re-checks the step claiming completion, the escalation ladder's
//! budget/rate-limit handling, and cancel-on-timeout (the last of these
//! lives in the turn engine that calls `expire_pending`/`cancel_pending`,
//! not here - see `grace_app.rs`).

use crate::dispatcher::Dispatcher;
use crate::events::EventSink;
use crate::grounder::{describe_target, GroundedPoint, Grounder};
use crate::intent::Intent;
use crate::memory::AgentMemory;
use crate::models::LargeLanguageModel;
use crate::perception::{observe_for_planner, PerceptionSource, ScreenSnapshot};
use crate::planner::{PlanArgs, PlanError, PlannedStep, Planner};
use crate::safety::evaluate as safety_evaluate;
use grace_contract::GraceEvent;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Tools whose execution constitutes actually doing something to the
/// desktop.
const INTERACTIVE_TOOLS: &[&str] = &[
    "cua_click", "cua_type_text", "cua_press_key", "cua_scroll",
    "cua_drag", "cua_launch", "cua_activate", "cua_set_value",
    "cua_secondary_action",
    "open_app", "close_app", "open_file", "delete_file",
    "read_pdf", "summarize_pdf", "search_files",
    "adjust_volume", "lock_computer", "open_calculator",
];

const ESCALATION_LADDER: [&str; 4] = ["normal", "reground", "stronger", "stop"];

/// How long a parked safety confirmation waits for an answer before it is
/// treated as abandoned (PLAN.md §10.1 item 3).
pub const PENDING_CONFIRMATION_TTL_SECONDS: u64 = 30;

pub const DEFAULT_MAX_SECONDS: f64 = 180.0;
pub const DEFAULT_MAX_CONSECUTIVE_PLAN_FAILURES: u32 = 3;
pub const DEFAULT_MAX_REPEATED_ACTIONS: u32 = 3;
pub const DEFAULT_MIN_ACTIONABLE: usize = crate::perception::DEFAULT_MIN_ACTIONABLE;

fn is_conversational_goal(goal: &str) -> bool {
    let lower = goal.to_lowercase();
    const PATTERNS: &[&str] = &[
        "what is", "what's", "who is", "where is", "why", "tell me", "explain",
        "how are", "how do", "how does", "hello", "hi", "hey", "thanks", "thank",
        "calculate",
    ];
    // Word-boundary-ish check: split on non-alphanumeric and look for any
    // pattern as a whole-word sequence. Ported informally from Python's
    // compiled `\b...\b` alternation - good enough given the same short,
    // curated word list and no adversarial input expected here.
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let joined = format!(" {} ", words.join(" "));
    PATTERNS.iter().any(|p| {
        let needle = format!(" {} ", p.replace('\'', "").replace(' ', " "));
        joined.contains(&needle) || joined.contains(&format!(" {p} "))
    })
}

/// A confirmation parked mid-turn, waiting for a yes/no.
pub enum Pending {
    /// A fast-path intent with no goal or memory behind it.
    Intent { intent: Intent, prompt: String, parked_at: Instant },
    /// A step mid-goal; resuming it continues the loop.
    Step { memory: AgentMemory, step: PlannedStep, prompt: String, parked_at: Instant },
}

impl Pending {
    pub fn prompt(&self) -> &str {
        match self {
            Pending::Intent { prompt, .. } => prompt,
            Pending::Step { prompt, .. } => prompt,
        }
    }

    fn parked_at(&self) -> Instant {
        match self {
            Pending::Intent { parked_at, .. } => *parked_at,
            Pending::Step { parked_at, .. } => *parked_at,
        }
    }
}

pub struct AgentLoopConfig {
    pub max_iterations: u32,
    pub max_consecutive_failures: u32,
    pub max_repeated_actions: u32,
    pub max_seconds: f64,
    pub planner_max_calls_per_goal: u32,
    pub stronger_model: Option<String>,
    pub min_actionable: usize,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: 0,
            max_consecutive_failures: DEFAULT_MAX_CONSECUTIVE_PLAN_FAILURES,
            max_repeated_actions: DEFAULT_MAX_REPEATED_ACTIONS,
            max_seconds: DEFAULT_MAX_SECONDS,
            planner_max_calls_per_goal: 0,
            stronger_model: Some("gemini-3.1-flash".to_string()),
            min_actionable: DEFAULT_MIN_ACTIONABLE,
        }
    }
}

/// Autonomous ReAct execution engine for complex multi-step tasks.
pub struct AgentLoop {
    config: AgentLoopConfig,
    pending: Option<Pending>,
}

impl AgentLoop {
    pub fn new(config: AgentLoopConfig) -> Self {
        Self { config, pending: None }
    }

    // -- safety resumption --------------------------------------------

    /// True when a step is parked waiting for the user to say yes or no.
    pub fn has_pending_confirmation(&mut self) -> bool {
        self.expire_pending();
        self.pending.is_some()
    }

    pub fn pending_prompt(&mut self) -> Option<String> {
        self.expire_pending();
        self.pending.as_ref().map(|p| p.prompt().to_string())
    }

    pub fn cancel_pending(&mut self) {
        self.pending = None;
    }

    /// Drop a parked confirmation once it has waited too long to still
    /// apply (ship-blocker #3): without a TTL a pending delete or app-close
    /// stays approvable forever, so an unrelated "yes" spoken in a much
    /// later exchange would silently confirm it.
    fn expire_pending(&mut self) {
        if let Some(pending) = &self.pending {
            if pending.parked_at().elapsed() > Duration::from_secs(PENDING_CONFIRMATION_TTL_SECONDS) {
                self.pending = None;
            }
        }
    }

    /// Hold a fast-path tool call until the user answers the question.
    pub fn park_intent(&mut self, intent: Intent, prompt: String) {
        self.pending = Some(Pending::Intent { intent, prompt, parked_at: Instant::now() });
    }

    /// Continue whatever stopped for a safety confirmation.
    pub fn resume_pending(
        &mut self,
        approved: bool,
        planner_llm: &mut dyn LargeLanguageModel,
        vision_llm: &mut dyn LargeLanguageModel,
        dispatcher: &mut Dispatcher,
        perception: &mut dyn PerceptionSource,
        sink: &mut dyn EventSink,
    ) -> Value {
        self.expire_pending();
        let Some(pending) = self.pending.take() else {
            return json!({"status": "error", "error": "Nothing is waiting for confirmation."});
        };

        match pending {
            Pending::Intent { intent, .. } => {
                if !approved {
                    return json!({"status": "ok", "final_response": "Alright, I won't do that.", "steps": []});
                }
                let result = dispatcher.execute_with_events(&intent, true, sink);
                json!({
                    "status": result.get("status").cloned().unwrap_or(json!("ok")),
                    "final_response": result.get("text").and_then(Value::as_str).unwrap_or("Done."),
                    "steps": [],
                })
            }
            Pending::Step { mut memory, step, .. } => {
                memory.resume_clock();
                if !approved {
                    memory.safety_pending = None;
                    memory.is_completed = true;
                    memory.final_response = Some("Alright, I won't do that.".to_string());
                    memory.add_step(step.thought.clone(), "converse", json!({}), json!({"status": "cancelled"}), Some("Cancelled".to_string()));
                    return build_result(&memory);
                }

                memory.safety_pending = None;
                let exec_result = self.dispatch(&step, dispatcher, true, sink);
                memory.add_step(step.thought.clone(), step.action.clone(), step.params.clone(), exec_result.clone(), Some(step.user_update.clone()));
                self.continue_loop(memory, Some(step), Some(exec_result), planner_llm, vision_llm, dispatcher, perception, sink)
            }
        }
    }

    // -- main entry point -----------------------------------------------

    pub fn run(
        &mut self,
        user_goal: &str,
        planner_llm: &mut dyn LargeLanguageModel,
        vision_llm: &mut dyn LargeLanguageModel,
        dispatcher: &mut Dispatcher,
        perception: &mut dyn PerceptionSource,
        sink: &mut dyn EventSink,
    ) -> Value {
        let memory = AgentMemory::new(user_goal, self.config.max_iterations, self.config.max_seconds);
        self.pending = None;
        self.continue_loop(memory, None, None, planner_llm, vision_llm, dispatcher, perception, sink)
    }

    /// Drive the loop until completion, the step cap, or a confirmation.
    fn continue_loop(
        &mut self,
        mut memory: AgentMemory,
        mut last_step: Option<PlannedStep>,
        mut last_result: Option<Value>,
        planner_llm: &mut dyn LargeLanguageModel,
        vision_llm: &mut dyn LargeLanguageModel,
        dispatcher: &mut Dispatcher,
        perception: &mut dyn PerceptionSource,
        sink: &mut dyn EventSink,
    ) -> Value {
        let mut planner = Planner::new(planner_llm, self.config.planner_max_calls_per_goal);
        let mut consecutive_failures: u32 = 0;
        let mut last_observation: Option<String> = None;
        let mut attempts: BTreeMap<String, u32> = BTreeMap::new();

        while !memory.is_completed && !memory.is_exceeded() {
            if memory.is_out_of_time() {
                return timeout_result(&memory);
            }

            let snapshot = perception.capture_snapshot();
            let view = observe_for_planner(&snapshot, self.config.min_actionable);

            if Some(&view.markdown) != last_observation.as_ref() {
                last_observation = Some(view.markdown.clone());
                attempts.clear();
            }

            let expectation_note = expectation_note(last_step.as_ref(), last_result.as_ref(), &snapshot);

            let plan_args = PlanArgs {
                goal: memory.user_goal.clone(),
                elements_prompt: view.markdown.clone(),
                history: memory.format_history_markdown(),
                scratchpad: memory.format_scratchpad_markdown(),
                expectation_note: expectation_note.clone(),
                window_title: snapshot.window_title.clone(),
                image_b64: view.image_b64.clone(),
                model: None,
            };

            let step = match planner.plan(&plan_args) {
                Ok(Some(step)) => step,
                Ok(None) => {
                    memory.set_scratchpad("last_error", json!("The previous plan could not be parsed."));
                    memory.add_step("", "converse", json!({}), json!({"status": "error", "error": "unparseable plan"}), Some("Rethinking…".to_string()));
                    last_step = None;
                    last_result = None;
                    consecutive_failures += 1;
                    if consecutive_failures >= self.config.max_consecutive_failures {
                        return plan_failure_result(&memory);
                    }
                    continue;
                }
                Err(PlanError::BudgetExceeded(_)) => return budget_result(&memory),
                Err(PlanError::RateLimited(e)) => {
                    return json!({
                        "status": "rate_limited",
                        "final_response": "I've hit my request limit for now. Please try again shortly.",
                        "steps": steps_json(&memory),
                        "_debug_rate_limit": e.0,
                    });
                }
            };
            consecutive_failures = 0;

            let signature = step_signature(&step);
            let attempt_count = *attempts.entry(signature.clone()).and_modify(|c| *c += 1).or_insert(1);
            let rung = rung(attempt_count, self.config.max_repeated_actions);

            if rung == "stop" {
                memory.rule_out(&format!("{} on this screen", step.action), "tried at every tier; nothing changed");
                return no_progress_result(&memory, &snapshot);
            }

            let mut step = step;
            if rung == "stronger" {
                match self.replan_stronger(&mut memory, &view, &snapshot, &expectation_note, &mut planner) {
                    Ok(Some(stronger)) => {
                        step = stronger;
                    }
                    Ok(None) => {}
                    Err(ReplanError::BudgetExceeded) => return budget_result(&memory),
                }
            }

            let is_safe_eval = safety_evaluate(&step.action, &step.params);
            if !is_safe_eval.is_safe {
                let prompt = is_safe_eval.confirmation_prompt.unwrap_or_default();
                memory.safety_pending = Some(json!({
                    "action": step.action, "params": step.params, "prompt": prompt,
                }));
                memory.pause_clock();
                let pending_step = step.to_json();
                self.pending = Some(Pending::Step { memory, step, prompt: prompt.clone(), parked_at: Instant::now() });
                return json!({
                    "status": "safety_confirmation_required",
                    "confirmation_prompt": prompt,
                    "pending_step": pending_step,
                    "steps": [],
                });
            }

            if step.needs_grounding() || rung == "reground" {
                self.ground_step(&mut step, &snapshot, rung == "reground", vision_llm);
            }

            let step_no = memory.current_iteration + 1;
            sink.emit(GraceEvent::ToolExecutionStarted {
                label: step.user_update.clone(),
                tool: Some(step.action.clone()),
                step: Some(step_no),
            });

            // Executed before the completion check so a final step that
            // also performs an action is never skipped.
            let exec_result = self.dispatch(&step, dispatcher, false, sink);

            if step.is_completed || step.action == "converse" {
                let (verified, hint) = verify_goal_completion(&memory, &snapshot, Some(&step), Some(&exec_result));
                if verified {
                    memory.is_completed = true;
                    memory.final_response = Some(final_response(&step, &memory));
                    memory.add_step(step.thought.clone(), step.action.clone(), step.params.clone(), exec_result, Some(step.user_update.clone()));
                    // No ToolExecutionFinished here - matches loop.py, whose
                    // emit for it sits after this "verified -> break" path,
                    // so a converse-completed goal never reaches it.
                    break;
                }
                memory.set_scratchpad("verification_hint", json!(hint));
                step.is_completed = false;
            }

            remember_outcome(&step, &exec_result, &mut memory);
            harvest(&exec_result, &mut memory);
            memory.add_step(step.thought.clone(), step.action.clone(), step.params.clone(), exec_result.clone(), Some(step.user_update.clone()));
            sink.emit(GraceEvent::ToolExecutionFinished {
                tool: Some(step.action.clone()),
                status: Some(exec_result.get("status").and_then(Value::as_str).unwrap_or("ok").to_string()),
            });
            last_step = Some(step);
            last_result = Some(exec_result);
        }

        if memory.is_exceeded() && !memory.is_completed {
            return json!({
                "status": "max_iterations_reached",
                "final_response": "I reached the step limit before fully completing the goal.",
                "steps": steps_json(&memory),
            });
        }

        build_result(&memory)
    }

    fn dispatch(&self, step: &PlannedStep, dispatcher: &mut Dispatcher, confirmed: bool, sink: &mut dyn EventSink) -> Value {
        if step.action == "converse" {
            return json!({"status": "ok"});
        }
        let intent = Intent::new(step.action.clone(), step.params.clone(), None);
        dispatcher.execute_with_events(&intent, confirmed, sink)
    }

    fn ground_step(&self, step: &mut PlannedStep, snapshot: &ScreenSnapshot, force: bool, vision_llm: &mut dyn LargeLanguageModel) {
        let description = describe_target(&step.params);
        let Some(png) = &snapshot.png_bytes else { return };
        if png.is_empty() {
            return;
        }
        let image_size = (
            if snapshot.image_width > 0 { snapshot.image_width } else { snapshot.width },
            if snapshot.image_height > 0 { snapshot.image_height } else { snapshot.height },
        );

        let mut grounder = Grounder::new(vision_llm);
        let point: Option<GroundedPoint> = grounder.locate(&description, png, image_size, (snapshot.width, snapshot.height));
        if let Some(point) = point {
            let obj = step.params.as_object_mut().unwrap();
            obj.insert("x".to_string(), json!(point.x));
            obj.insert("y".to_string(), json!(point.y));
            if force {
                obj.remove("element_id");
                obj.remove("element_index");
            }
        }
    }

    fn replan_stronger(
        &self,
        memory: &mut AgentMemory,
        view: &crate::perception::Observation,
        snapshot: &ScreenSnapshot,
        expectation_note: &str,
        planner: &mut Planner,
    ) -> Result<Option<PlannedStep>, ReplanError> {
        let args = PlanArgs {
            goal: memory.user_goal.clone(),
            elements_prompt: view.markdown.clone(),
            history: memory.format_history_markdown(),
            scratchpad: memory.format_scratchpad_markdown(),
            expectation_note: expectation_note.to_string(),
            window_title: snapshot.window_title.clone(),
            image_b64: view.image_b64.clone(),
            model: self.config.stronger_model.clone(),
        };
        match planner.plan(&args) {
            Ok(step) => Ok(step),
            Err(PlanError::BudgetExceeded(_)) => Err(ReplanError::BudgetExceeded),
            // Rate limit and any other planning failure on this OPTIONAL
            // attempt: keep the original step rather than failing the goal.
            Err(PlanError::RateLimited(_)) => Ok(None),
        }
    }
}

enum ReplanError {
    BudgetExceeded,
}

fn expectation_note(last_step: Option<&PlannedStep>, last_result: Option<&Value>, snapshot: &ScreenSnapshot) -> String {
    let (Some(last_step), Some(last_result)) = (last_step, last_result) else {
        return String::new();
    };
    if last_result.is_null() {
        return String::new();
    }

    let mut lines = vec![format!(
        "You ran `{}` and expected: {}",
        last_step.action,
        if last_step.expect.is_empty() { "(nothing stated)" } else { &last_step.expect }
    )];

    let inner = last_result.get("result").filter(|v| v.is_object());
    let reason = failure_reason(last_result);

    if let Some(reason) = &reason {
        lines.push(format!("IT FAILED: {reason}"));
        lines.push("Do not repeat this action unchanged. Try a different element or a different approach.".to_string());
    } else {
        let status = last_result.get("status").and_then(Value::as_str).unwrap_or("ok");
        let message = inner.and_then(|i| i.get("message")).and_then(Value::as_str).unwrap_or(status);
        lines.push(format!("The tool reported: {message}"));
    }

    let verified = inner.and_then(|i| i.get("verified")).and_then(Value::as_bool);
    let evidence = inner.and_then(|i| i.get("evidence")).and_then(Value::as_str).unwrap_or("");
    let sent = inner.and_then(|i| i.get("sent")).and_then(Value::as_bool).unwrap_or(false);
    match verified {
        Some(true) => lines.push(format!("EFFECT CONFIRMED: {evidence}")),
        Some(false) if reason.is_none() => {
            lines.push(format!("NO EFFECT: {evidence}"));
            lines.push("The input was sent but changed nothing. Try a different target.".to_string());
        }
        None if sent => {
            lines.push(format!("EFFECT UNKNOWN: {evidence}"));
            lines.push("Do not treat that as failure. Look at the screen description below and judge for yourself.".to_string());
        }
        _ => {}
    }

    if !snapshot.window_title.is_empty() {
        lines.push(format!("The active window is now: {}", snapshot.window_title));
    }

    if let Some(focused) = snapshot.graph.focused() {
        lines.push(format!(
            "Keyboard focus is on [{}] {} '{}' (frame={})",
            focused.id,
            focused.role,
            if focused.name.is_empty() { &focused.placeholder } else { &focused.name },
            focused.frame
        ));
    } else if last_step.action == "cua_click" {
        lines.push("Nothing currently has keyboard focus.".to_string());
    }

    lines.push("Judge from the screen description above whether your expectation actually came true.".to_string());
    lines.join("\n")
}

/// Keep what this step settled, in a form that outlives the history window.
/// Only definite outcomes are recorded (`verified is None` is deliberately
/// not written down as either).
fn remember_outcome(step: &PlannedStep, exec_result: &Value, memory: &mut AgentMemory) {
    if let Some(reason) = failure_reason(exec_result) {
        memory.rule_out(&format!("`{}` with {}", step.action, short_params(&step.params)), &reason);
        return;
    }
    let inner = exec_result.get("result").filter(|v| v.is_object());
    let verified = inner.and_then(|i| i.get("verified")).and_then(Value::as_bool);
    if verified == Some(true) && !step.expect.is_empty() {
        memory.establish(step.expect.clone());
    } else if verified == Some(false) {
        let evidence = inner.and_then(|i| i.get("evidence")).and_then(Value::as_str).unwrap_or("it changed nothing");
        memory.rule_out(&format!("`{}` with {}", step.action, short_params(&step.params)), evidence);
    }
}

/// Move useful tool output into the scratchpad.
fn harvest(exec_result: &Value, memory: &mut AgentMemory) {
    if exec_result.get("status").and_then(Value::as_str) != Some("ok") {
        return;
    }
    let res = exec_result.get("result").filter(|v| v.is_object()).unwrap_or(exec_result);
    if let Some(windows) = res.get("windows").and_then(Value::as_array) {
        let titles: Vec<Value> = windows
            .iter()
            .filter_map(|w| w.get("title"))
            .filter(|t| t.as_str().map(|s| !s.is_empty()).unwrap_or(false))
            .cloned()
            .collect();
        memory.set_scratchpad("open_windows", Value::Array(titles));
    }
    if let Some(apps) = res.get("apps").and_then(Value::as_array) {
        let names: Vec<Value> = apps
            .iter()
            .filter_map(|a| a.get("name"))
            .filter(|n| n.as_str().map(|s| !s.is_empty()).unwrap_or(false))
            .cloned()
            .collect();
        memory.set_scratchpad("open_apps", Value::Array(names));
    }
    for (key, target) in [("text", "latest_extracted_text"), ("summary", "latest_summary"), ("files", "found_files")] {
        if let Some(value) = exec_result.get(key) {
            memory.set_scratchpad(target, value.clone());
        }
    }
}

fn final_response(step: &PlannedStep, memory: &AgentMemory) -> String {
    if !step.final_response.is_empty() && step.final_response != step.thought {
        return step.final_response.clone();
    }
    if let Some(Value::Array(wins)) = memory.get_scratchpad("open_windows") {
        let names: Vec<String> = wins.iter().take(5).filter_map(|v| v.as_str().map(String::from)).collect();
        return format!("The open windows are: {}.", names.join(", "));
    }
    if let Some(Value::Array(apps)) = memory.get_scratchpad("open_apps") {
        let names: Vec<String> = apps.iter().take(5).filter_map(|v| v.as_str().map(String::from)).collect();
        return format!("The open applications are: {}.", names.join(", "));
    }
    "Goal completed.".to_string()
}

fn steps_json(memory: &AgentMemory) -> Value {
    Value::Array(memory.steps_taken.iter().map(StepRecordJson::to_json).collect())
}

trait StepRecordJson {
    fn to_json(&self) -> Value;
}
impl StepRecordJson for crate::memory::StepRecord {
    fn to_json(&self) -> Value {
        crate::memory::StepRecord::to_json(self)
    }
}

fn build_result(memory: &AgentMemory) -> Value {
    json!({"status": "ok", "final_response": memory.final_response, "steps": steps_json(memory)})
}

fn budget_result(memory: &AgentMemory) -> Value {
    let response = memory.final_response.clone().unwrap_or_else(|| {
        if let Some(text) = memory.get_scratchpad("latest_extracted_text") {
            truncate_display(text, 400)
        } else if !memory.steps_taken.is_empty() {
            "I got part of the way through that, but I've used up my planning budget for this request.".to_string()
        } else {
            "I wasn't able to work out how to do that.".to_string()
        }
    });
    json!({"status": "planner_budget_exceeded", "final_response": response, "steps": steps_json(memory)})
}

fn no_progress_result(memory: &AgentMemory, snapshot: &ScreenSnapshot) -> Value {
    let app = if snapshot.window_title.is_empty() { "that window".to_string() } else { snapshot.window_title.clone() };
    let response = if !snapshot.has_elements() {
        format!("I can't read anything inside {app} - it doesn't report its contents to Windows, so I can't see what to click. I've stopped rather than keep guessing.")
    } else {
        format!("I've tried the same thing several times in {app} without getting anywhere, so I've stopped.")
    };
    json!({"status": "no_progress", "final_response": memory.final_response.clone().unwrap_or(response), "steps": steps_json(memory)})
}

fn timeout_result(memory: &AgentMemory) -> Value {
    let where_clause = memory.steps_taken.last().map(|s| format!(" The last thing I did was `{}`.", s.action)).unwrap_or_default();
    let response = memory.final_response.clone().unwrap_or_else(|| {
        format!("I spent {:.0} seconds on that without finishing, so I've stopped.{where_clause}", memory.elapsed_seconds())
    });
    json!({"status": "timed_out", "final_response": response, "steps": steps_json(memory)})
}

fn plan_failure_result(memory: &AgentMemory) -> Value {
    let response = memory.final_response.clone().unwrap_or_else(|| {
        "I'm having trouble planning that right now - I can't reach my language model. Please check the connection and try again.".to_string()
    });
    json!({"status": "planner_failed", "final_response": response, "steps": steps_json(memory)})
}

/// Block a claimed completion that no action could have produced. See
/// Python's docstring on `_verify_goal_completion` for the full reasoning;
/// ported verbatim: a conversational goal is trusted outright, a goal with
/// no interactive step yet is refused, a blind window's claim is trusted
/// (no channel to check it against), and otherwise the LAST interactive
/// step - including the one making the claim right now, via
/// `claiming_step`/`claiming_result` - must not itself have failed.
fn verify_goal_completion(
    memory: &AgentMemory,
    snapshot: &ScreenSnapshot,
    claiming_step: Option<&PlannedStep>,
    claiming_result: Option<&Value>,
) -> (bool, Option<String>) {
    if is_conversational_goal(&memory.user_goal) {
        return (true, None);
    }

    let mut interactive: Vec<(&str, &Value)> = memory
        .steps_taken
        .iter()
        .filter(|s| INTERACTIVE_TOOLS.contains(&s.action.as_str()))
        .map(|s| (s.action.as_str(), &s.result))
        .collect();

    let claim_entry;
    if let Some(step) = claiming_step {
        if INTERACTIVE_TOOLS.contains(&step.action.as_str()) {
            claim_entry = claiming_result.cloned().unwrap_or(json!({}));
            interactive.push((step.action.as_str(), &claim_entry));
        }
    }

    if interactive.is_empty() {
        return (
            false,
            Some(
                "Goal observation check: no desktop interaction has been performed yet. Inspect what is on screen and execute the next action."
                    .to_string(),
            ),
        );
    }

    if snapshot.is_blind(DEFAULT_MIN_ACTIONABLE) {
        return (true, None);
    }

    let (last_action, last_result) = interactive.last().unwrap();
    if let Some(failure) = failure_reason(last_result) {
        return (
            false,
            Some(format!(
                "Goal observation check: your last action (`{last_action}`) reported failure - {failure} - so the goal cannot be complete. Address that before finishing."
            )),
        );
    }

    (true, None)
}

/// Why a dispatch failed, or `None` if it did not. One definition of "this
/// failed" across the dispatcher's `{"status": "error"}` envelope and the
/// handler's `{"ok": false}` body.
fn failure_reason(result: &Value) -> Option<String> {
    if result.is_null() {
        return None;
    }
    let inner = result.get("result").filter(|v| v.is_object());
    let status_is_error = result.get("status").and_then(Value::as_str) == Some("error");
    let ok_is_false = inner.and_then(|i| i.get("ok")).and_then(Value::as_bool) == Some(false);
    if !status_is_error && !ok_is_false {
        return None;
    }
    result
        .get("error")
        .or_else(|| inner.and_then(|i| i.get("error")))
        .or_else(|| inner.and_then(|i| i.get("message")))
        .and_then(Value::as_str)
        .map(String::from)
        .or_else(|| Some("no reason given".to_string()))
}

/// Identifies "the same action again". The window the step names is
/// excluded: its handle and title wobble between snapshots of the same
/// application.
fn step_signature(step: &PlannedStep) -> String {
    let mut params = step.params.clone();
    if let Some(obj) = params.as_object_mut() {
        obj.remove("window");
    }
    format!("{}:{}", step.action, params)
}

/// Enough of a step's parameters to recognise it again, and no more.
fn short_params(params: &Value) -> String {
    let mut interesting = params.clone();
    if let Some(obj) = interesting.as_object_mut() {
        obj.remove("window");
    }
    let text = interesting.to_string();
    if text.chars().count() <= 90 {
        text
    } else {
        format!("{}…", text.chars().take(87).collect::<String>())
    }
}

/// Which strategy this attempt gets. With the default `before_stop=3`, an
/// action is attempted normally, then re-grounded, then re-planned by a
/// stronger model, and only then abandoned.
fn rung(attempt: u32, before_stop: u32) -> &'static str {
    let before_stop = before_stop.max(1);
    if attempt >= before_stop + 1 {
        return "stop";
    }
    let index = attempt.min(ESCALATION_LADDER.len() as u32).saturating_sub(1);
    ESCALATION_LADDER[index as usize]
}

fn truncate_display(value: &Value, limit: usize) -> String {
    let s = value.as_str().map(String::from).unwrap_or_else(|| value.to_string());
    if s.chars().count() <= limit {
        s
    } else {
        s.chars().take(limit).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatcher::{CuaBridge, Dispatcher, SystemActions};
    use crate::models::{RateLimitError, ScriptedLlm};
    use crate::perception::{ElementGraph, ElementNode, ScriptedPerception};

    #[derive(Default)]
    struct NoopActions;
    impl SystemActions for NoopActions {
        fn open_url(&mut self, _url: &str) -> bool { true }
        fn launch_app(&mut self, _name: &str) -> Value { json!({"status": "ok"}) }
        fn close_app(&mut self, _name: &str) -> Result<(), String> { Ok(()) }
        fn search_files(&mut self, _query: &str) -> Vec<String> { vec![] }
        fn open_file(&mut self, _name: &str) -> Result<(), String> { Ok(()) }
        fn adjust_volume(&mut self, _amount: i64, _mode: &str) -> Result<i64, String> { Ok(50) }
        fn lock_computer(&mut self) -> Result<(), String> { Ok(()) }
        fn open_calculator(&mut self) -> Result<(), String> { Ok(()) }
        fn delete_file(&mut self, name: &str) -> Result<String, String> { Ok(name.to_string()) }
        fn press_undo(&mut self) -> Result<(), String> { Ok(()) }
        fn describe_screen(&mut self) -> (String, Vec<String>) { (String::new(), vec![]) }
        fn set_speech_rate(&mut self, _speed: f32) {}
        fn play_success_earcon(&mut self) {}
        fn play_error_earcon(&mut self) {}
    }

    struct AlwaysOkCua;
    impl CuaBridge for AlwaysOkCua {
        fn is_ready(&self) -> bool { true }
        fn perform(&mut self, _action: &str, _params: &Value) -> Value { json!({"ok": true, "sent": true, "verified": true, "evidence": "clicked"}) }
    }

    fn blind_snapshot() -> ScreenSnapshot {
        ScreenSnapshot {
            graph: ElementGraph { elements: vec![], window_title: "Terminal".into(), is_browser: false, sources: vec![] },
            window_title: "Terminal".into(),
            width: 1920,
            height: 1080,
            png_bytes: None,
            image_width: 0,
            image_height: 0,
        }
    }

    fn rich_snapshot() -> ScreenSnapshot {
        let elements: Vec<ElementNode> = (1..=10)
            .map(|i| ElementNode {
                id: i,
                role: "button".into(),
                name: format!("Button {i}"),
                rect: (0, 0, 10, 10),
                center: (5, 5),
                enabled: true,
                ..Default::default()
            })
            .collect();
        ScreenSnapshot {
            graph: ElementGraph { elements, window_title: "Notepad".into(), is_browser: false, sources: vec!["uia".into()] },
            window_title: "Notepad".into(),
            width: 1920,
            height: 1080,
            png_bytes: None,
            image_width: 0,
            image_height: 0,
        }
    }

    #[test]
    fn a_conversational_goal_completes_immediately_via_converse() {
        let mut planner_llm = ScriptedLlm::new(vec![ScriptedLlm::text(
            r#"{"action": "converse", "params": {"response": "Hi there!"}, "is_completed": true, "final_response": "Hi there!"}"#,
        )]);
        let mut vision_llm = ScriptedLlm::new(vec![]);
        let mut actions = NoopActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let mut perception = ScriptedPerception::new(vec![blind_snapshot()]);

        let mut loop_ = AgentLoop::new(AgentLoopConfig::default());
        let mut sink = crate::events::RecordingEventSink::default();
        let result = loop_.run("say hello", &mut planner_llm, &mut vision_llm, &mut dispatcher, &mut perception, &mut sink);
        assert_eq!(result["status"], "ok");
        assert_eq!(result["final_response"], "Hi there!");
    }

    #[test]
    fn safety_confirmation_parks_the_step_and_pauses_the_clock() {
        let mut planner_llm = ScriptedLlm::new(vec![ScriptedLlm::text(
            r#"{"action": "delete_file", "params": {"name": "tax_return.pdf"}, "user_update": "Deleting..."}"#,
        )]);
        let mut vision_llm = ScriptedLlm::new(vec![]);
        let mut actions = NoopActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let mut perception = ScriptedPerception::new(vec![rich_snapshot()]);

        let mut loop_ = AgentLoop::new(AgentLoopConfig::default());
        let mut sink = crate::events::RecordingEventSink::default();
        let result = loop_.run("delete my tax return", &mut planner_llm, &mut vision_llm, &mut dispatcher, &mut perception, &mut sink);
        assert_eq!(result["status"], "safety_confirmation_required");
        assert!(loop_.has_pending_confirmation());
    }

    #[test]
    fn cancel_pending_clears_a_parked_confirmation() {
        let mut loop_ = AgentLoop::new(AgentLoopConfig::default());
        loop_.park_intent(Intent::new("delete_file", json!({"name": "x"}), None), "sure?".into());
        assert!(loop_.has_pending_confirmation());
        loop_.cancel_pending();
        assert!(!loop_.has_pending_confirmation());
    }

    #[test]
    fn resuming_a_parked_intent_with_no_approval_does_not_dispatch() {
        let mut planner_llm = ScriptedLlm::new(vec![]);
        let mut vision_llm = ScriptedLlm::new(vec![]);
        let mut actions = NoopActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let mut perception = ScriptedPerception::new(vec![]);

        let mut loop_ = AgentLoop::new(AgentLoopConfig::default());
        loop_.park_intent(Intent::new("lock_computer", json!({}), None), "Lock now?".into());
        let mut sink = crate::events::RecordingEventSink::default();
        let result = loop_.resume_pending(false, &mut planner_llm, &mut vision_llm, &mut dispatcher, &mut perception, &mut sink);
        assert_eq!(result["final_response"], "Alright, I won't do that.");
        assert!(!loop_.has_pending_confirmation());
    }

    #[test]
    fn resuming_a_parked_intent_with_approval_dispatches_it() {
        let mut planner_llm = ScriptedLlm::new(vec![]);
        let mut vision_llm = ScriptedLlm::new(vec![]);
        let mut actions = NoopActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let mut perception = ScriptedPerception::new(vec![]);

        let mut loop_ = AgentLoop::new(AgentLoopConfig::default());
        loop_.park_intent(Intent::new("lock_computer", json!({}), None), "Lock now?".into());
        let mut sink = crate::events::RecordingEventSink::default();
        let result = loop_.resume_pending(true, &mut planner_llm, &mut vision_llm, &mut dispatcher, &mut perception, &mut sink);
        assert_eq!(result["final_response"], "I've locked your computer.");
    }

    #[test]
    fn pending_confirmation_expires_after_the_ttl() {
        let mut loop_ = AgentLoop::new(AgentLoopConfig::default());
        loop_.pending = Some(Pending::Intent {
            intent: Intent::new("lock_computer", json!({}), None),
            prompt: "sure?".into(),
            parked_at: Instant::now() - Duration::from_secs(PENDING_CONFIRMATION_TTL_SECONDS + 1),
        });
        assert!(!loop_.has_pending_confirmation());
    }

    #[test]
    fn pending_confirmation_within_the_ttl_still_applies() {
        let mut loop_ = AgentLoop::new(AgentLoopConfig::default());
        loop_.pending = Some(Pending::Intent {
            intent: Intent::new("lock_computer", json!({}), None),
            prompt: "sure?".into(),
            parked_at: Instant::now() - Duration::from_secs(PENDING_CONFIRMATION_TTL_SECONDS - 5),
        });
        assert!(loop_.has_pending_confirmation());
    }

    #[test]
    fn an_unparseable_plan_is_retried_up_to_the_consecutive_failure_cap() {
        let mut planner_llm = ScriptedLlm::new(vec![Ok(None), Ok(None), Ok(None)]);
        let mut vision_llm = ScriptedLlm::new(vec![]);
        let mut actions = NoopActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let mut perception = ScriptedPerception::new(vec![blind_snapshot()]);

        let mut config = AgentLoopConfig::default();
        config.max_consecutive_failures = 3;
        let mut loop_ = AgentLoop::new(config);
        let mut sink = crate::events::RecordingEventSink::default();
        let result = loop_.run("do a thing", &mut planner_llm, &mut vision_llm, &mut dispatcher, &mut perception, &mut sink);
        assert_eq!(result["status"], "planner_failed");
    }

    #[test]
    fn a_rate_limited_plan_call_surfaces_immediately() {
        let mut planner_llm = ScriptedLlm::new(vec![Err(RateLimitError("quota".into()))]);
        let mut vision_llm = ScriptedLlm::new(vec![]);
        let mut actions = NoopActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let mut perception = ScriptedPerception::new(vec![blind_snapshot()]);

        let mut loop_ = AgentLoop::new(AgentLoopConfig::default());
        let mut sink = crate::events::RecordingEventSink::default();
        let result = loop_.run("do a thing", &mut planner_llm, &mut vision_llm, &mut dispatcher, &mut perception, &mut sink);
        assert_eq!(result["status"], "rate_limited");
    }

    #[test]
    fn repeating_the_same_action_against_an_unchanging_view_eventually_stops() {
        // Every plan call asks to click the same element again; the view
        // never changes (one repeated snapshot), so the repeated-action
        // ladder should escalate through reground/stronger and then stop.
        let repeat_step = r#"{"action": "cua_click", "params": {"element_id": 1}, "user_update": "Clicking...", "expect": "nothing"}"#;
        // 5 responses: iteration 1 (normal), iteration 2 (reground - no extra
        // planner call), iteration 3 (stronger - the ladder's third rung asks
        // ONE EXTRA planner call for the re-plan attempt, so this attempt
        // consumes two responses), iteration 4 (its own plan() call to
        // compute the signature/rung that turns out to be "stop").
        let mut planner_llm = ScriptedLlm::new(vec![
            ScriptedLlm::text(repeat_step),
            ScriptedLlm::text(repeat_step),
            ScriptedLlm::text(repeat_step),
            ScriptedLlm::text(repeat_step), // the "stronger" rung's extra re-plan call
            ScriptedLlm::text(repeat_step),
        ]);
        let mut vision_llm = ScriptedLlm::new(vec![]);
        let mut actions = NoopActions::default();
        let mut cua = AlwaysOkCua;
        let mut dispatcher = Dispatcher::new(Some(&mut cua), &mut actions);
        // Same rich snapshot every time - the view truly never changes.
        let snapshot = rich_snapshot();
        let mut perception = ScriptedPerception::new(vec![snapshot.clone(), snapshot.clone(), snapshot]);

        let mut config = AgentLoopConfig::default();
        config.max_repeated_actions = 3;
        let mut loop_ = AgentLoop::new(config);
        let mut sink = crate::events::RecordingEventSink::default();
        let result = loop_.run("click the button repeatedly", &mut planner_llm, &mut vision_llm, &mut dispatcher, &mut perception, &mut sink);
        assert_eq!(result["status"], "no_progress");
    }

    #[test]
    fn completion_guard_rejects_a_claim_with_no_interaction_yet() {
        let memory = AgentMemory::new("open notepad and create a document", 0, 0.0);
        let snapshot = rich_snapshot();
        let claiming_step = PlannedStep { action: "converse".into(), is_completed: true, ..Default::default() };
        let (verified, hint) = verify_goal_completion(&memory, &snapshot, Some(&claiming_step), Some(&json!({"status": "ok"})));
        assert!(!verified);
        assert!(hint.unwrap().contains("no desktop interaction"));
    }

    #[test]
    fn completion_guard_rejects_when_the_claiming_step_itself_failed() {
        let mut memory = AgentMemory::new("send the message", 0, 0.0);
        let claiming_step = PlannedStep { action: "cua_click".into(), is_completed: true, ..Default::default() };
        let claiming_result = json!({"status": "error", "error": "wrong_focus"});
        let snapshot = rich_snapshot();
        let (verified, hint) = verify_goal_completion(&memory, &snapshot, Some(&claiming_step), Some(&claiming_result));
        assert!(!verified);
        assert!(hint.unwrap().contains("wrong_focus"));
        let _ = &mut memory;
    }

    #[test]
    fn completion_guard_accepts_a_conversational_goal_outright() {
        let memory = AgentMemory::new("hello there", 0, 0.0);
        let snapshot = blind_snapshot();
        let (verified, hint) = verify_goal_completion(&memory, &snapshot, None, None);
        assert!(verified);
        assert!(hint.is_none());
    }

    #[test]
    fn completion_guard_trusts_a_blind_window_it_cannot_check() {
        let mut memory = AgentMemory::new("send the message", 0, 0.0);
        memory.add_step("", "cua_click", json!({}), json!({"status": "ok"}), None);
        let snapshot = blind_snapshot(); // 0 actionable elements -> blind
        let (verified, _) = verify_goal_completion(&memory, &snapshot, None, None);
        assert!(verified);
    }

    #[test]
    fn escalation_ladder_progresses_through_every_rung_before_stopping() {
        assert_eq!(rung(1, 3), "normal");
        assert_eq!(rung(2, 3), "reground");
        assert_eq!(rung(3, 3), "stronger");
        assert_eq!(rung(4, 3), "stop");
    }

    #[test]
    fn step_signature_ignores_the_window_parameter() {
        let a = PlannedStep { action: "cua_click".into(), params: json!({"element_id": 1, "window": {"title": "A"}}), ..Default::default() };
        let b = PlannedStep { action: "cua_click".into(), params: json!({"element_id": 1, "window": {"title": "B"}}), ..Default::default() };
        assert_eq!(step_signature(&a), step_signature(&b));
    }
}
