//! Wires the real `grace_core` turn engine to a `WsEventServer`, so
//! `GRACE_BACKEND=rust` can run an actual activation turn end-to-end through
//! the real event contract, not just bind a socket.
//!
//! **Honest scope**: there is no real STT, LLM, or Windows-automation
//! backend wired up in this phase (see PORT_STATUS.md - `grace-models`'s
//! `HttpLlm` needs a running llama.cpp server, and `grace-audio`/`grace-win`
//! have no real capture/dispatch yet). `run_demo_turn` runs ONE turn with a
//! fixed, hardcoded transcript through every real decision-making component
//! this port has (`IntentParser`, `CapabilityRouter`, `Dispatcher`,
//! `AgentLoop`, the response generator), talking to a real local llama.cpp
//! server if `LLAMA_SERVER_URL`/the default `http://127.0.0.1:8080` has one
//! running, and falling back to a scripted response otherwise so the
//! wiring is still demonstrable with nothing else running. This is a
//! deliberate, temporary stand-in for real speech input - it proves the
//! WebSocket -> turn engine -> dispatcher -> response generator -> WebSocket
//! pipeline end-to-end, which is what "GRACE_BACKEND=rust runs a full turn"
//! actually means until real STT exists to produce the transcript instead.

use crate::server::WsEventServer;
use grace_core::agent_loop::{AgentLoop, AgentLoopConfig};
use grace_core::dispatcher::{Dispatcher, NoopSystemActions};
use grace_core::events::EventSink;
use grace_core::feedback::{Earcon, EarconPlayer};
use grace_core::grace_app::{handle_activation_turn, ListenOutcome, RecordingWakeWordControl, TurnContext};
use grace_core::intent::IntentParser;
use grace_core::models::{LargeLanguageModel, LlmRequest, RateLimitError, ScriptedTts};
use grace_core::perception::ScriptedPerception;
use grace_models::HttpLlm;
use std::sync::Arc;

/// Adapts `Arc<WsEventServer>` to `EventSink`, so the real turn engine can
/// emit straight to connected WebSocket clients.
struct WsSink(Arc<WsEventServer>);

impl EventSink for WsSink {
    fn emit(&mut self, event: grace_contract::GraceEvent) {
        self.0.emit(event);
    }
}

/// A `LargeLanguageModel` that tries a real local llama.cpp server first
/// (via `HttpLlm`) and falls back to a fixed scripted response if nothing is
/// listening - so this demo turn runs identically whether or not a model
/// server happens to be up, and never blocks indefinitely on one.
struct DemoLlm {
    http: HttpLlm,
    fallback: Vec<String>,
    fallback_index: usize,
}

impl LargeLanguageModel for DemoLlm {
    fn generate_text(&mut self, request: &LlmRequest) -> Result<Option<String>, RateLimitError> {
        match self.http.generate_text(request) {
            Ok(Some(text)) => Ok(Some(text)),
            _ => {
                let text = self.fallback.get(self.fallback_index).cloned();
                self.fallback_index += 1;
                Ok(text)
            }
        }
    }
}

/// A silent `EarconPlayer` - the demo turn has no real speaker output any
/// more than it has a real microphone, so earcon requests are just dropped.
struct SilentEarcons;
impl EarconPlayer for SilentEarcons {
    fn play(&mut self, _earcon: Earcon) {}
}

/// Runs one activation turn with the fixed transcript `"open the
/// calculator"` and emits its real event sequence to every client connected
/// to `server`. See this module's doc comment for exactly what "real" means
/// here.
pub fn run_demo_turn(server: &Arc<WsEventServer>) {
    let llama_url =
        std::env::var("LLAMA_SERVER_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());

    let mut planner_llm = DemoLlm {
        http: HttpLlm::new(llama_url.clone()),
        fallback: vec![r#"{"tool": "open_calculator", "params": {}}"#.to_string()],
        fallback_index: 0,
    };
    let mut vision_llm = DemoLlm { http: HttpLlm::new(llama_url), fallback: vec![], fallback_index: 0 };
    let mut actions = NoopSystemActions;
    let mut dispatcher = Dispatcher::new(None, &mut actions);
    let mut perception = ScriptedPerception::new(vec![]);
    let mut agent_loop = AgentLoop::new(AgentLoopConfig::default());
    let mut tts = ScriptedTts::default();
    let mut wake_word = RecordingWakeWordControl::default();
    let mut earcons = SilentEarcons;
    let mut sink = WsSink(Arc::clone(server));
    let mut intent_parser = IntentParser::new();
    let mut last_spoken_text = String::new();

    let mut ctx = TurnContext {
        wake_word: &mut wake_word,
        earcons: &mut earcons,
        sink: &mut sink,
        intent_parser: &mut intent_parser,
        planner_llm: &mut planner_llm,
        vision_llm: &mut vision_llm,
        dispatcher: &mut dispatcher,
        perception: &mut perception,
        agent_loop: &mut agent_loop,
        tts: &mut tts,
    };

    handle_activation_turn(&mut ctx, ListenOutcome::Transcript("open the calculator".to_string()), 0.0, &|_| {}, INTENT_SYSTEM_PROMPT, &mut last_spoken_text);
}

/// A minimal intent-parsing instruction. Not the full prompt
/// `intent/prompt.py` builds (that lives in `src/grace/`, which this port
/// cannot import and does not duplicate here) - close enough for a demo
/// turn against a real local model, and irrelevant to the fallback path
/// (`DemoLlm` ignores the prompt when it returns the scripted response).
const INTENT_SYSTEM_PROMPT: &str = "Convert the user's request into one JSON object: {\"tool\": \"<tool name>\", \"params\": {...}}.";
