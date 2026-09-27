//! Ported from `src/grace/tools/dispatcher.py`: routes parsed intents to
//! CUA or hardcoded tool implementations.
//!
//! The outermost side-effecting boundary in the backend, and - per the
//! module's own docstring - the place the safety check belongs: it used to
//! live only inside the agent loop, which meant `delete_file`, `close_app`
//! and `lock_computer` (all in `CapabilityRouter::FAST_PATH_TOOLS`) reached
//! the desktop unconfirmed whenever the router judged the request simple
//! enough for the fast path. `confirmation_required` is checked in
//! `Dispatcher::execute` here, same as Python, so that gap cannot reopen by
//! one call site forgetting to ask.
//!
//! Every OS-touching action (launching a process, killing one, adjusting
//! volume, locking the workstation, moving a file to the Recycle Bin,
//! pressing Ctrl+Z, reading a PDF) is behind the `SystemActions` trait, so
//! this module's own logic - parameter validation, alias resolution, query
//! sanitisation, the exact error/label text - is tested without touching a
//! real desktop. See `grace-win` for the (not yet implemented) real
//! `SystemActions`.

use crate::events::EventSink;
use crate::intent::Intent;
use crate::safety::evaluate;
use grace_contract::GraceEvent;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::LazyLock;

/// One entry per handler `SystemActions` implements for a `cua_*` tool.
static CUA_LABELS: LazyLock<BTreeMap<&'static str, &'static str>> = LazyLock::new(|| {
    BTreeMap::from([
        ("click", "Clicking…"),
        ("type_text", "Typing…"),
        ("press_key", "Pressing key…"),
        ("drag", "Dragging…"),
        ("scroll", "Scrolling…"),
        ("screenshot", "Taking screenshot…"),
        ("list_windows", "Listing windows…"),
        ("list_apps", "Listing applications…"),
        ("get_window", "Reading window…"),
        ("activate", "Switching window…"),
        ("launch", "Launching application…"),
        ("text", "Reading screen…"),
        ("set_value", "Setting value…"),
        ("secondary_action", "Right-clicking…"),
    ])
});

static TOOL_LABELS: LazyLock<BTreeMap<&'static str, &'static str>> = LazyLock::new(|| {
    BTreeMap::from([
        ("open_app", "Opening application…"),
        ("close_app", "Closing application…"),
        ("search_files", "Searching files…"),
        ("open_file", "Opening file…"),
        ("read_pdf", "Reading document…"),
        ("summarize_pdf", "Summarizing document…"),
        ("adjust_volume", "Adjusting volume…"),
        ("lock_computer", "Locking computer…"),
        ("open_calculator", "Opening calculator…"),
        ("delete_file", "Moving file to recycle bin…"),
        ("undo", "Undoing action…"),
        ("describe_screen", "Inspecting screen…"),
        ("set_speech_rate", "Adjusting speech rate…"),
    ])
});

/// True when `value` is an http(s) URL - the only scheme this port ever
/// hands to the OS's default browser opener unconfirmed (R15). Anything
/// else (`file://`, a UNC path, a custom registered scheme) is the same
/// class of unconfirmed local-open `os.startfile` is.
fn is_http_url(value: &str) -> bool {
    let lower = value.to_lowercase();
    let rest = lower.strip_prefix("http://").or_else(|| lower.strip_prefix("https://"));
    rest.is_some_and(|r| !r.is_empty())
}

static WEBSITE_ALIASES: LazyLock<BTreeMap<&'static str, &'static str>> = LazyLock::new(|| {
    BTreeMap::from([
        ("youtube", "https://www.youtube.com"),
        ("youtube.com", "https://www.youtube.com"),
        ("google", "https://www.google.com"),
        ("google.com", "https://www.google.com"),
        ("gmail", "https://mail.google.com"),
        ("reddit", "https://www.reddit.com"),
        ("reddit.com", "https://www.reddit.com"),
        ("github", "https://github.com"),
        ("github.com", "https://github.com"),
        ("twitter", "https://x.com"),
        ("x", "https://x.com"),
        ("x.com", "https://x.com"),
        ("amazon", "https://www.amazon.com"),
        ("amazon.com", "https://www.amazon.com"),
        ("wikipedia", "https://www.wikipedia.org"),
        ("netflix", "https://www.netflix.com"),
        ("chatgpt", "https://chatgpt.com"),
    ])
});

/// The refusal to return from `Dispatcher::execute`, or `None` to go ahead.
/// Split out so a replay harness can apply the same decision without a
/// second copy of it.
pub fn confirmation_required(intent: &Intent, confirmed: bool) -> Option<Value> {
    if confirmed {
        return None;
    }
    let eval = evaluate(&intent.tool, &intent.params);
    if eval.is_safe {
        return None;
    }
    let prompt = eval.confirmation_prompt.unwrap_or_default();
    Some(json!({
        "status": "confirmation_required",
        "confirmation_prompt": prompt,
        "text": prompt,
        "tool": intent.tool,
        "params": intent.params,
    }))
}

/// What a dispatch handler needs to touch the real desktop/OS. A concrete
/// implementation lives in `grace-win` (not yet real - see PORT_STATUS.md);
/// tests use a fake that records calls and returns scripted results.
pub trait SystemActions: Send {
    fn open_url(&mut self, url: &str) -> bool;
    /// Launch an installed application by name/query. Returns Grace's own
    /// result envelope (mirrors `AppIndexer.launch`'s `{"status", "text"}`
    /// shape) so `grace-win`'s ported `plan_launch` decision is exposed
    /// unchanged.
    fn launch_app(&mut self, name: &str) -> Value;
    fn close_app(&mut self, name: &str) -> Result<(), String>;
    /// Searches `Documents` for files matching `query` with one of the
    /// given extensions. Returns matching absolute paths.
    fn search_files(&mut self, query: &str) -> Vec<String>;
    fn open_file(&mut self, name: &str) -> Result<(), String>;
    fn adjust_volume(&mut self, amount: i64, mode: &str) -> Result<i64, String>;
    fn lock_computer(&mut self) -> Result<(), String>;
    fn open_calculator(&mut self) -> Result<(), String>;
    fn delete_file(&mut self, name: &str) -> Result<String, String>;
    fn press_undo(&mut self) -> Result<(), String>;
    fn describe_screen(&mut self) -> (String, Vec<String>);
    fn set_speech_rate(&mut self, speed: f32);
    fn play_success_earcon(&mut self);
    fn play_error_earcon(&mut self);
}

/// What the `cua_*` bridge needs. A concrete implementation wraps
/// `ComputerUse`/`grace-win`'s `DesktopActuator`; tests use a fake.
pub trait CuaBridge: Send {
    fn is_ready(&self) -> bool;
    /// `action` is the tool name with its `cua_` prefix stripped (e.g.
    /// `"click"`, `"type_text"`). Returns Grace's own result envelope.
    fn perform(&mut self, action: &str, params: &Value) -> Value;
}

/// Strip everything cmd.exe (or a future shell path) would re-parse as its
/// own syntax from a search query. Ported from `_search_files`'s
/// sanitisation regex.
fn sanitize_query(query: &str) -> String {
    query
        .chars()
        .filter(|c| !"\"'`;$|&<>(){}![]\\".contains(*c))
        .collect::<String>()
        .trim()
        .to_string()
}

fn tool_started_label(tool: &str, params: &Value) -> String {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let query = params.get("query").and_then(Value::as_str).unwrap_or("");
    match tool {
        "open_app" if !name.is_empty() => format!("Opening {name}…"),
        "close_app" if !name.is_empty() => format!("Closing {name}…"),
        "search_files" if !query.is_empty() => format!("Searching for {query}…"),
        "open_file" if !name.is_empty() => format!("Opening {name}…"),
        "delete_file" if !name.is_empty() => format!("Moving {name} to recycle bin…"),
        _ => TOOL_LABELS.get(tool).map(|s| s.to_string()).unwrap_or_else(|| format!("{tool}…")),
    }
}

/// Routes intents to CUA or hardcoded tool implementations.
pub struct Dispatcher<'a> {
    cua: Option<&'a mut dyn CuaBridge>,
    actions: &'a mut dyn SystemActions,
    /// Every `ToolExecutionStarted`/`ToolExecutionFinished` label emitted,
    /// in order - a test-observable stand-in for the real WS emission
    /// (`grace-backend::WsEventServer` in the wired-up shell).
    pub emitted_labels: Vec<String>,
    last_search_results: Vec<String>,
}

impl<'a> Dispatcher<'a> {
    pub fn new(cua: Option<&'a mut dyn CuaBridge>, actions: &'a mut dyn SystemActions) -> Self {
        Self { cua, actions, emitted_labels: Vec::new(), last_search_results: Vec::new() }
    }

    /// Execute a parsed intent, with no event emission - for callers (the
    /// agent loop) that emit their own wrapping
    /// `ToolExecutionStarted`/`Finished` and don't want the dispatcher's
    /// inner pair too... except Python's dispatcher emits unconditionally,
    /// so callers that DO want the faithful double emission should use
    /// `execute_with_events` instead. This overload exists for tests and
    /// call sites that don't care about the WS surface at all.
    pub fn execute(&mut self, intent: &Intent, confirmed: bool) -> Value {
        let mut sink = crate::events::RecordingEventSink::default();
        self.execute_with_events(intent, confirmed, &mut sink)
    }

    /// Execute a parsed intent. This is the outermost side-effecting
    /// boundary - safety is enforced here regardless of which path (fast
    /// path or agent loop) produced the intent. Emits
    /// `ToolExecutionStarted{label}`/`ToolExecutionFinished{}` (no `tool`/
    /// `step`/`status` fields - those are the agent loop's own, separate
    /// emission around its call into this method) exactly where
    /// `dispatcher.py`'s `_emit_tool_started`/`_emit_tool_finished` do.
    pub fn execute_with_events(&mut self, intent: &Intent, confirmed: bool, sink: &mut dyn EventSink) -> Value {
        if let Some(refusal) = confirmation_required(intent, confirmed) {
            return refusal;
        }
        self.dispatch(intent, sink)
    }

    fn dispatch(&mut self, intent: &Intent, sink: &mut dyn EventSink) -> Value {
        let tool = intent.tool.as_str();
        let params = &intent.params;

        if let Some(action) = tool.strip_prefix("cua_") {
            return self.execute_cua(action, params, sink);
        }

        let label = tool_started_label(tool, params);
        self.emitted_labels.push(label.clone());
        sink.emit(GraceEvent::ToolExecutionStarted { label, tool: None, step: None });
        let result = match tool {
            "converse" => self.converse(params),
            "open_app" => self.open_app(params),
            "close_app" => self.close_app(params),
            "search_files" => self.search_files(params),
            "open_file" => self.open_file(params),
            "read_pdf" => json!({"status": "error", "error": "read_pdf is not implemented in this port yet"}),
            "summarize_pdf" => json!({"status": "error", "error": "summarize_pdf is not implemented in this port yet"}),
            "adjust_volume" => self.adjust_volume(params),
            "lock_computer" => self.lock_computer(),
            "open_calculator" => self.open_calculator(),
            "delete_file" => self.delete_file(params),
            "undo" => self.undo(),
            "describe_screen" => self.describe_screen(),
            "set_speech_rate" => self.set_speech_rate(params),
            other => json!({"status": "error", "error": format!("Unknown tool: {other}")}),
        };
        self.emitted_labels.push("<finished>".to_string());
        sink.emit(GraceEvent::ToolExecutionFinished { tool: None, status: None });
        result
    }

    fn execute_cua(&mut self, action: &str, params: &Value, sink: &mut dyn EventSink) -> Value {
        let Some(cua) = self.cua.as_deref_mut() else {
            return json!({"status": "error", "error": "Computer use not available"});
        };
        if !cua.is_ready() {
            return json!({"status": "error", "error": "Computer use not available"});
        }

        let label = CUA_LABELS.get(action).map(|s| s.to_string()).unwrap_or_else(|| format!("{action}…"));
        self.emitted_labels.push(label.clone());
        sink.emit(GraceEvent::ToolExecutionStarted { label, tool: None, step: None });
        let result = cua.perform(action, params);
        self.emitted_labels.push("<finished>".to_string());
        sink.emit(GraceEvent::ToolExecutionFinished { tool: None, status: None });

        if let Some(error) = result.get("error") {
            let error_text = error.as_str().unwrap_or_default();
            return json!({
                "status": "error",
                "error": error_text,
                "text": format!("Action error: {error_text}"),
                "result": result,
            });
        }
        json!({"status": "ok", "result": result})
    }

    fn converse(&self, params: &Value) -> Value {
        let response = params.get("response").and_then(Value::as_str).unwrap_or("");
        json!({"status": "ok", "text": response})
    }

    fn open_app(&mut self, params: &Value) -> Value {
        let name = params.get("name").and_then(Value::as_str).unwrap_or("").trim().to_string();
        let url = params.get("url").and_then(Value::as_str).unwrap_or("").trim().to_string();
        if name.is_empty() && url.is_empty() {
            return json!({"status": "error", "error": "Missing 'name' or 'url' parameter"});
        }

        // An explicit `url` used to reach `open_url` completely unvalidated -
        // unlike the name-inference branch below, which only ever builds an
        // http(s) string in the first place. `open_app` is a fast-path tool
        // (no confirmation gate), so a model-supplied `file://` URI, a UNC
        // path, or a registered custom scheme there was a known local-RCE/
        // credential-leak class (R15). Only http(s) is allowed through.
        let mut target_url = String::new();
        if !url.is_empty() {
            if !is_http_url(&url) {
                return json!({
                    "status": "error",
                    "error": format!("Refusing to open '{url}': only http/https links are allowed."),
                });
            }
            target_url = url;
        } else {
            let lower_name = name.to_lowercase();
            if let Some(known) = WEBSITE_ALIASES.get(lower_name.as_str()) {
                target_url = known.to_string();
            } else if lower_name.starts_with("http://")
                || lower_name.starts_with("https://")
                || lower_name.starts_with("www.")
                || [".com", ".org", ".net", ".io", ".edu", ".gov"].iter().any(|ext| lower_name.contains(ext))
            {
                let candidate = if lower_name.starts_with("http") { name.clone() } else { format!("https://{name}") };
                if is_http_url(&candidate) {
                    target_url = candidate;
                }
            }
        }

        if !target_url.is_empty() {
            self.actions.open_url(&target_url);
            let display_name = if name.is_empty() { target_url.clone() } else { name };
            return json!({"status": "ok", "text": format!("I've opened {display_name} in your browser.")});
        }

        self.actions.launch_app(&name)
    }

    fn close_app(&mut self, params: &Value) -> Value {
        let name = params.get("name").and_then(Value::as_str).unwrap_or("");
        if name.is_empty() {
            return json!({"status": "error", "error": "Missing 'name' parameter"});
        }
        match self.actions.close_app(name) {
            Ok(()) => json!({"status": "ok", "text": format!("I've closed {name}.")}),
            Err(e) => json!({"status": "error", "error": e, "text": format!("I couldn't find {name} to close it.")}),
        }
    }

    fn search_files(&mut self, params: &Value) -> Value {
        let query = params.get("query").and_then(Value::as_str).unwrap_or("");
        if query.is_empty() {
            return json!({"status": "error", "error": "Missing 'query' parameter"});
        }
        let clean_query = sanitize_query(query);
        if clean_query.is_empty() {
            return json!({"status": "error", "error": "Invalid query after sanitization"});
        }

        self.last_search_results = self.actions.search_files(&clean_query);
        if !self.last_search_results.is_empty() {
            let listing = self
                .last_search_results
                .iter()
                .enumerate()
                .map(|(i, path)| format!("  {}. {path}", i + 1))
                .collect::<Vec<_>>()
                .join("\n");
            json!({
                "status": "ok",
                "text": format!("Here are the matching files:\n{listing}"),
                "files": self.last_search_results,
            })
        } else {
            json!({"status": "ok", "text": format!("I couldn't find any files matching '{query}'."), "files": []})
        }
    }

    fn open_file(&mut self, params: &Value) -> Value {
        let name = params.get("name").and_then(Value::as_str).unwrap_or("");
        if name.is_empty() {
            return json!({"status": "error", "error": "Missing 'name' parameter"});
        }
        match self.actions.open_file(name) {
            Ok(()) => json!({"status": "ok", "text": format!("I've opened {name}.")}),
            Err(e) => json!({"status": "error", "error": e, "text": format!("I couldn't open {name}. {e}")}),
        }
    }

    fn adjust_volume(&mut self, params: &Value) -> Value {
        let amount = params.get("amount").and_then(Value::as_i64).unwrap_or(0);
        let mode = params.get("mode").and_then(Value::as_str).unwrap_or("increase");
        match self.actions.adjust_volume(amount, mode) {
            Ok(new_percent) => json!({"status": "ok", "text": format!("Volume set to {new_percent} percent."), "volume": new_percent}),
            Err(e) => json!({"status": "error", "error": e, "text": format!("Could not adjust volume: {e}")}),
        }
    }

    fn lock_computer(&mut self) -> Value {
        match self.actions.lock_computer() {
            Ok(()) => json!({"status": "ok", "text": "I've locked your computer."}),
            Err(_) => json!({"status": "error", "error": "lock failed", "text": "I couldn't lock your computer."}),
        }
    }

    fn open_calculator(&mut self) -> Value {
        match self.actions.open_calculator() {
            Ok(()) => json!({"status": "ok", "text": "I've opened the calculator."}),
            Err(_) => json!({"status": "error", "error": "launch failed", "text": "I couldn't open the calculator."}),
        }
    }

    fn delete_file(&mut self, params: &Value) -> Value {
        let name = params.get("name").and_then(Value::as_str).unwrap_or("");
        if name.is_empty() {
            return json!({"status": "error", "error": "Missing 'name' parameter"});
        }
        match self.actions.delete_file(name) {
            Ok(basename) => json!({
                "status": "ok",
                "text": format!("I've moved '{basename}' to the Recycle Bin."),
                "action": "delete_file",
            }),
            Err(e) => json!({"status": "error", "error": e, "text": format!("I couldn't delete '{name}'. {e}")}),
        }
    }

    fn undo(&mut self) -> Value {
        match self.actions.press_undo() {
            Ok(()) => {
                self.actions.play_success_earcon();
                json!({"status": "ok", "action": "undo", "text": "Undone."})
            }
            Err(e) => {
                self.actions.play_error_earcon();
                json!({"status": "error", "error": e, "text": format!("Could not undo: {e}")})
            }
        }
    }

    fn describe_screen(&mut self) -> Value {
        let (title, controls) = self.actions.describe_screen();
        let text = if !controls.is_empty() {
            let controls_str = controls.iter().map(|n| format!("'{n}'")).collect::<Vec<_>>().join(", ");
            format!("You are looking at {title}. Key controls: {controls_str}.")
        } else if !title.is_empty() {
            format!("The active window is {title}.")
        } else {
            "I cannot detect an active application on screen.".to_string()
        };
        json!({"status": "ok", "action": "describe_screen", "text": text, "window_title": title, "controls": controls})
    }

    fn set_speech_rate(&mut self, params: &Value) -> Value {
        let rate_val = params.get("rate").cloned().unwrap_or_else(|| json!("1.0"));
        let speed = match &rate_val {
            Value::String(s) => {
                let lower = s.to_lowercase();
                let lower = lower.trim();
                if lower.contains("slow") {
                    0.8
                } else if lower.contains("fast") || lower.contains("quick") {
                    1.25
                } else if lower.contains("normal") || lower.contains("default") || lower.contains("reset") {
                    1.0
                } else {
                    extract_first_number(lower).unwrap_or(1.0)
                }
            }
            Value::Number(n) => n.as_f64().unwrap_or(1.0) as f32,
            _ => 1.0,
        };
        let speed = speed.clamp(0.5, 2.5);
        self.actions.set_speech_rate(speed);
        self.actions.play_success_earcon();
        json!({"status": "ok", "action": "set_speech_rate", "text": format!("Speech rate set to {speed:.2}x."), "speed": speed})
    }
}

/// A `SystemActions` that succeeds at everything with plausible defaults,
/// touching nothing real. Useful wherever a test or the harness needs the
/// dispatcher's own decision logic and hardcoded response text (which
/// doesn't depend on `SystemActions` at all for most tools) without caring
/// about the underlying action.
#[derive(Default)]
pub struct NoopSystemActions;

impl SystemActions for NoopSystemActions {
    fn open_url(&mut self, _url: &str) -> bool {
        true
    }
    fn launch_app(&mut self, name: &str) -> Value {
        json!({"status": "ok", "text": format!("I've opened {name}.")})
    }
    fn close_app(&mut self, _name: &str) -> Result<(), String> {
        Ok(())
    }
    fn search_files(&mut self, _query: &str) -> Vec<String> {
        Vec::new()
    }
    fn open_file(&mut self, _name: &str) -> Result<(), String> {
        Ok(())
    }
    fn adjust_volume(&mut self, amount: i64, mode: &str) -> Result<i64, String> {
        Ok(if mode == "set" || mode == "percent" { amount.clamp(0, 100) } else { 50 })
    }
    fn lock_computer(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn open_calculator(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn delete_file(&mut self, name: &str) -> Result<String, String> {
        Ok(name.to_string())
    }
    fn press_undo(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn describe_screen(&mut self) -> (String, Vec<String>) {
        (String::new(), Vec::new())
    }
    fn set_speech_rate(&mut self, _speed: f32) {}
    fn play_success_earcon(&mut self) {}
    fn play_error_earcon(&mut self) {}
}

fn extract_first_number(text: &str) -> Option<f32> {
    let re = regex::Regex::new(r"[-+]?(?:\d*\.\d+|\d+)").unwrap();
    re.find(text).and_then(|m| m.as_str().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct FakeActions {
        opened_urls: Vec<String>,
        launch_result: Option<Value>,
        closed: Vec<String>,
        fail_close: bool,
        search_results: Vec<String>,
        speech_rate: Option<f32>,
        success_earcons: u32,
        error_earcons: u32,
    }

    impl SystemActions for FakeActions {
        fn open_url(&mut self, url: &str) -> bool {
            self.opened_urls.push(url.to_string());
            true
        }
        fn launch_app(&mut self, name: &str) -> Value {
            self.launch_result.clone().unwrap_or_else(|| json!({"status": "ok", "text": format!("I've opened {name}.")}))
        }
        fn close_app(&mut self, name: &str) -> Result<(), String> {
            self.closed.push(name.to_string());
            if self.fail_close { Err("not found".to_string()) } else { Ok(()) }
        }
        fn search_files(&mut self, _query: &str) -> Vec<String> {
            self.search_results.clone()
        }
        fn open_file(&mut self, _name: &str) -> Result<(), String> {
            Ok(())
        }
        fn adjust_volume(&mut self, amount: i64, mode: &str) -> Result<i64, String> {
            Ok(if mode == "set" { amount } else { 50 })
        }
        fn lock_computer(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn open_calculator(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn delete_file(&mut self, name: &str) -> Result<String, String> {
            Ok(name.to_string())
        }
        fn press_undo(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn describe_screen(&mut self) -> (String, Vec<String>) {
            ("Notepad".to_string(), vec![])
        }
        fn set_speech_rate(&mut self, speed: f32) {
            self.speech_rate = Some(speed);
        }
        fn play_success_earcon(&mut self) {
            self.success_earcons += 1;
        }
        fn play_error_earcon(&mut self) {
            self.error_earcons += 1;
        }
    }

    struct FakeCua {
        ready: bool,
        result: Value,
    }

    impl CuaBridge for FakeCua {
        fn is_ready(&self) -> bool {
            self.ready
        }
        fn perform(&mut self, _action: &str, _params: &Value) -> Value {
            self.result.clone()
        }
    }

    #[test]
    fn confirmation_required_tools_are_refused_unless_confirmed() {
        let intent = Intent::new("delete_file", json!({"name": "x.txt"}), None);
        let refusal = confirmation_required(&intent, false).unwrap();
        assert_eq!(refusal["status"], "confirmation_required");
        assert!(confirmation_required(&intent, true).is_none());
    }

    #[test]
    fn execute_refuses_before_touching_any_action() {
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("lock_computer", json!({}), None);
        let result = dispatcher.execute(&intent, false);
        assert_eq!(result["status"], "confirmation_required");
    }

    #[test]
    fn confirmed_lock_computer_goes_ahead() {
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("lock_computer", json!({}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "ok");
    }

    #[test]
    fn open_app_with_a_known_website_alias_opens_a_url_not_the_indexer() {
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("open_app", json!({"name": "youtube"}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "ok");
        assert_eq!(actions.opened_urls, vec!["https://www.youtube.com".to_string()]);
    }

    #[test]
    fn open_app_with_an_ordinary_name_goes_to_the_app_indexer() {
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("open_app", json!({"name": "Notepad"}), None);
        dispatcher.execute(&intent, true);
        assert!(actions.opened_urls.is_empty());
    }

    #[test]
    fn open_app_with_an_http_url_opens_it() {
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("open_app", json!({"url": "https://example.com/page"}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "ok");
        assert_eq!(actions.opened_urls, vec!["https://example.com/page".to_string()]);
    }

    #[test]
    fn open_app_refuses_a_file_url() {
        // R15: the `url` param used to reach `open_url` completely
        // unvalidated - a `file://` URI there is a known local-RCE class.
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("open_app", json!({"url": "file:///C:/Windows/System32/cmd.exe"}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "error");
        assert!(actions.opened_urls.is_empty());
    }

    #[test]
    fn open_app_refuses_a_unc_path_as_url() {
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("open_app", json!({"url": r"\\attacker\share\x"}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "error");
        assert!(actions.opened_urls.is_empty());
    }

    #[test]
    fn open_app_refuses_a_custom_uri_scheme_as_url() {
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("open_app", json!({"url": "myapp://do-something-bad"}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "error");
        assert!(actions.opened_urls.is_empty());
    }

    #[test]
    fn open_app_missing_both_name_and_url_is_an_error() {
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("open_app", json!({}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "error");
    }

    #[test]
    fn search_files_sanitises_shell_characters() {
        let mut actions = FakeActions { search_results: vec!["C:/Documents/report.pdf".into()], ..Default::default() };
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("search_files", json!({"query": "report; rm -rf"}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "ok");
        assert_eq!(result["files"][0], "C:/Documents/report.pdf");
    }

    #[test]
    fn search_files_purely_unsafe_query_is_rejected() {
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("search_files", json!({"query": ";;;"}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "error");
    }

    #[test]
    fn cua_tool_without_a_bridge_is_an_error() {
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("cua_click", json!({"element_id": 1}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "error");
    }

    #[test]
    fn cua_tool_error_result_preserves_the_inner_envelope() {
        let mut actions = FakeActions::default();
        let mut cua = FakeCua { ready: true, result: json!({"error": "wrong_focus", "status": "wrong_focus"}) };
        let mut dispatcher = Dispatcher::new(Some(&mut cua), &mut actions);
        let intent = Intent::new("cua_click", json!({"element_id": 1}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "error");
        assert_eq!(result["error"], "wrong_focus");
        assert_eq!(result["result"]["status"], "wrong_focus");
    }

    #[test]
    fn cua_tool_ok_result_is_wrapped() {
        let mut actions = FakeActions::default();
        let mut cua = FakeCua { ready: true, result: json!({"clicked": true}) };
        let mut dispatcher = Dispatcher::new(Some(&mut cua), &mut actions);
        let intent = Intent::new("cua_click", json!({"element_id": 1}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "ok");
        assert_eq!(result["result"]["clicked"], true);
    }

    #[test]
    fn undo_plays_success_earcon_on_ok() {
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("undo", json!({}), None);
        dispatcher.execute(&intent, true);
        assert_eq!(actions.success_earcons, 1);
    }

    #[test]
    fn set_speech_rate_parses_words_and_numbers() {
        for (input, expected) in [("slow", 0.8_f32), ("fast", 1.25), ("1.4", 1.4), ("normal", 1.0)] {
            let mut actions = FakeActions::default();
            let mut dispatcher = Dispatcher::new(None, &mut actions);
            let intent = Intent::new("set_speech_rate", json!({"rate": input}), None);
            dispatcher.execute(&intent, true);
            drop(dispatcher);
            assert!((actions.speech_rate.unwrap() - expected).abs() < 0.001, "{input} -> {expected}");
        }
    }

    #[test]
    fn set_speech_rate_clamps_extreme_values() {
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("set_speech_rate", json!({"rate": "10.0"}), None);
        dispatcher.execute(&intent, true);
        assert_eq!(actions.speech_rate, Some(2.5));
    }

    #[test]
    fn unknown_tool_is_an_error() {
        let mut actions = FakeActions::default();
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("not_a_tool", json!({}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "error");
    }

    #[test]
    fn close_app_failure_reports_could_not_find() {
        let mut actions = FakeActions { fail_close: true, ..Default::default() };
        let mut dispatcher = Dispatcher::new(None, &mut actions);
        let intent = Intent::new("close_app", json!({"name": "Ghost"}), None);
        let result = dispatcher.execute(&intent, true);
        assert_eq!(result["status"], "error");
        assert!(result["text"].as_str().unwrap().contains("couldn't find Ghost"));
    }
}
