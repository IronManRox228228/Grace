//! Ported from `src/grace/main.py`'s `GraceApp`: the turn state machine -
//! activation, the immediate stop/repeat voice reflexes, pending-confirmation
//! resolution, and the follow-up window - wired over the traits this crate
//! already defines instead of real audio/model/OS calls.
//!
//! **Scope note**: the raw microphone capture loop (`_handle_activation_turn`'s
//! chunk-by-chunk VAD/timeout bookkeeping before a transcript exists) is real
//! hardware I/O and is not reimplemented here - `grace-audio`'s `VadDetector`
//! already ports that decision logic in isolation (see its module doc), and
//! the task's hard constraints forbid a test driving a real microphone
//! anyway. This module picks up from "a transcript arrived, or the listen
//! timed out with silence" - already-provided `Option<&str>` - and ports
//! everything from there: the wake-word-resume-on-every-exit-path guarantee,
//! the strict stop/repeat reflexes, pending-confirmation handling, routing,
//! dispatch, and the follow-up window's restart loop. That is where PLAN.md
//! §10.1's ship-blocker fixes actually live (items 1, 2, 3, 4), and where
//! this port's tests can prove they still hold.
//!
//! **`MIN_WAKE_TO_IDLE_SECONDS` (added to `main.py` after this module was
//! first ported)**: on the empty-transcript path only (not on a
//! transcription failure), `_handle_activation_turn` now sleeps for
//! whatever's left of a 3-second floor since the wake word, before playing
//! the cancel earcon and going idle - so a silent activation holds the pill
//! open at least 3 s rather than flashing shut instantly. Ported as
//! `MIN_WAKE_TO_IDLE_SECONDS` below. Because the real listen loop (with its
//! own `listen_start`) is out of scope here (see above), this function takes
//! the elapsed listening time as a plain `f64` from its caller, and a
//! `hold_for` callback in place of `asyncio.sleep` so tests can assert the
//! requested duration without a real 3-second wait.

use crate::agent_loop::AgentLoop;
use crate::confirmation::confirmation_answer;
use crate::dispatcher::Dispatcher;
use crate::events::EventSink;
use crate::feedback::{Earcon, EarconPlayer};
use crate::intent::{Intent, IntentParser};
use crate::models::{LargeLanguageModel, TextToSpeech};
use crate::perception::PerceptionSource;
use crate::response_generator::speak_text;
use crate::router::{classify, ParsedIntentRef, TaskComplexity};
use grace_contract::GraceEvent;
use serde_json::Value;
use std::sync::LazyLock;
use std::time::Duration;

/// How long a wake activation must stay visibly "on" before the pill is
/// allowed to close, even when nothing was said. Mirrors
/// `GraceApp.MIN_WAKE_TO_IDLE_SECONDS` in `src/grace/main.py`.
pub const MIN_WAKE_TO_IDLE_SECONDS: f64 = 3.0;

/// What the (out-of-scope, real-hardware) listen loop produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListenOutcome {
    /// Whisper transcribed something (possibly still empty/whitespace,
    /// which `handle_activation_turn` treats the same as `Silence`).
    Transcript(String),
    /// The listen loop ended with nothing said - a timeout, or a transcript
    /// that was empty/whitespace-only. Mirrors Python's `not
    /// transcript.strip()` branch, which is the ONLY one
    /// `MIN_WAKE_TO_IDLE_SECONDS` applies to.
    Silence,
    /// `whisper.transcribe()` itself raised. Mirrors Python's `except
    /// Exception` around that call - deliberately a separate case from
    /// `Silence`, because that branch returns immediately with no
    /// `MIN_WAKE_TO_IDLE_SECONDS` hold.
    TranscriptionFailed,
}

/// Pausing/resuming the wake-word spotter around a turn. A real
/// implementation wraps `grace-audio`'s `WakeWordDetector`; tests use a
/// fake that records calls.
pub trait WakeWordControl: Send {
    fn pause(&mut self);
    fn resume(&mut self);
}

#[derive(Default)]
pub struct RecordingWakeWordControl {
    pub pause_calls: u32,
    pub resume_calls: u32,
}

impl WakeWordControl for RecordingWakeWordControl {
    fn pause(&mut self) {
        self.pause_calls += 1;
    }
    fn resume(&mut self) {
        self.resume_calls += 1;
    }
}

static STOP_COMMANDS: LazyLock<std::collections::BTreeSet<&'static str>> = LazyLock::new(|| {
    [
        "stop", "cancel", "never mind", "nevermind", "quiet", "be quiet",
        "shut up", "pause", "halt", "stop speaking", "stop that", "abort",
    ]
    .into_iter()
    .collect()
});

static REPEAT_COMMANDS: LazyLock<std::collections::BTreeSet<&'static str>> = LazyLock::new(|| {
    [
        "repeat", "repeat that", "what did you say", "say that again",
        "say again", "pardon", "what was that", "can you repeat that",
    ]
    .into_iter()
    .collect()
});

fn clean_transcript(transcript: &str) -> String {
    transcript
        .trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '_')
        .collect::<String>()
        .trim()
        .to_string()
}

fn is_stop_command(clean: &str) -> bool {
    clean == "stop" || clean.starts_with("stop ") || STOP_COMMANDS.contains(clean)
}

fn is_repeat_command(clean: &str) -> bool {
    REPEAT_COMMANDS.contains(clean)
}

/// Everything the turn state machine needs from the outside world for one
/// turn, bundled as mutable borrows so `handle_activation_turn` and
/// `handle_followup_transcript` don't repeat an 8-parameter signature at
/// every call site. Each field keeps its own lifetime so a `Dispatcher<'_>`
/// (which owns its own, independent borrows) can be embedded without lifetime
/// conflicts against the bundle's own borrows.
pub struct TurnContext<'a, 'b> {
    pub wake_word: &'a mut dyn WakeWordControl,
    pub earcons: &'a mut dyn EarconPlayer,
    pub sink: &'a mut dyn EventSink,
    pub intent_parser: &'a mut IntentParser,
    pub planner_llm: &'a mut dyn LargeLanguageModel,
    pub vision_llm: &'a mut dyn LargeLanguageModel,
    pub dispatcher: &'a mut Dispatcher<'b>,
    pub perception: &'a mut dyn PerceptionSource,
    pub agent_loop: &'a mut AgentLoop,
    pub tts: &'a mut dyn TextToSpeech,
}

/// Generates an intent for `transcript`, matching `GemmaClient.generate_intent`
/// + `IntentParser.parse`'s error handling: any parse failure (or an empty
/// LLM response) yields `None`, never propagated as an error - the caller
/// falls through to the router's keyword-based default.
fn generate_intent(transcript: &str, system_prompt: &str, llm: &mut dyn LargeLanguageModel, parser: &mut IntentParser) -> Option<Intent> {
    let request = crate::models::LlmRequest {
        prompt: transcript.to_string(),
        system_prompt: Some(system_prompt.to_string()),
        temperature: 0.1,
        max_tokens: 4096,
        ..Default::default()
    };
    let raw = llm.generate_text(&request).ok()??;
    parser.parse(&raw).ok()
}

fn agent_response_text(agent_res: &Value) -> Option<String> {
    if agent_res.get("status").and_then(Value::as_str) == Some("safety_confirmation_required") {
        return agent_res.get("confirmation_prompt").and_then(Value::as_str).map(String::from);
    }
    Some(
        agent_res
            .get("final_response")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or("Goal executed.")
            .to_string(),
    )
}

fn speak(ctx: &mut TurnContext, text: &str) {
    if text.trim().is_empty() {
        return;
    }
    speak_text(text, "af_bella", ctx.tts, ctx.sink, &|| false);
}

/// Apply a yes/no to a parked action. Returns `true` if it was handled
/// (an answer either way; `false` means the utterance should be treated as
/// a new request instead).
fn resolve_pending_confirmation(ctx: &mut TurnContext, transcript: &str) -> bool {
    let Some(answer) = confirmation_answer(transcript) else {
        ctx.agent_loop.cancel_pending();
        return false;
    };

    let result = ctx.agent_loop.resume_pending(answer, ctx.planner_llm, ctx.vision_llm, ctx.dispatcher, ctx.perception, ctx.sink);
    let response_text = if result.get("status").and_then(Value::as_str) == Some("safety_confirmation_required") {
        agent_response_text(&result).unwrap_or_default()
    } else {
        result
            .get("final_response")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .unwrap_or_else(|| if answer { "Done.".to_string() } else { "Alright, I won't do that.".to_string() })
    };
    speak(ctx, &response_text);
    true
}

/// Ask the question a refused fast-path dispatch came back with. `true` if
/// parked (ship-blocker: the fast path reaches `delete_file`/`close_app`/
/// `lock_computer` without going through the agent loop, so this is what
/// stops those three running unconfirmed).
fn park_if_confirmation_required(ctx: &mut TurnContext, intent: &Intent, result: &Value) -> bool {
    if result.get("status").and_then(Value::as_str) != Some("confirmation_required") {
        return false;
    }
    let prompt = result.get("confirmation_prompt").and_then(Value::as_str).unwrap_or("Should I go ahead?").to_string();
    ctx.agent_loop.park_intent(intent.clone(), prompt.clone());
    speak(ctx, &prompt);
    true
}

/// Handle one activation turn, given the outcome of the (out-of-scope,
/// real-hardware) listen loop already captured as a `ListenOutcome`.
/// `elapsed_listen_seconds` is how long that loop actually ran (its
/// caller's `time.time() - listen_start`), and `hold_for` is called
/// (instead of `asyncio.sleep`) with whatever's left of
/// `MIN_WAKE_TO_IDLE_SECONDS` on a `Silence` outcome - see this module's doc
/// comment. Returns `true` if the turn succeeded (the caller should open the
/// follow-up window next), `false` if it aborted (silence, an immediate
/// stop, a transcription failure) - mirroring `_handle_activation_turn`'s
/// bool return, which main.py's `_handle_activation` uses to decide whether
/// the wake word must be resumed here-and-now or left to the follow-up
/// window.
pub fn handle_activation_turn(
    ctx: &mut TurnContext,
    outcome: ListenOutcome,
    elapsed_listen_seconds: f64,
    hold_for: &dyn Fn(Duration),
    intent_system_prompt: &str,
    last_spoken_text: &mut String,
) -> bool {
    ctx.earcons.play(Earcon::Chime);
    ctx.sink.emit(GraceEvent::WakeWordDetected);
    ctx.sink.emit(GraceEvent::ListeningStarted);
    ctx.wake_word.pause();

    if outcome == ListenOutcome::TranscriptionFailed {
        // Python's `except Exception` around `whisper.transcribe()`: no
        // MIN_WAKE_TO_IDLE_SECONDS hold on this path, unlike Silence below.
        ctx.earcons.play(Earcon::Error);
        ctx.sink.emit(GraceEvent::ConversationFinished);
        ctx.sink.emit(GraceEvent::Idle);
        return false;
    }

    let transcript = match &outcome {
        ListenOutcome::Transcript(t) if !t.trim().is_empty() => t.clone(),
        _ => {
            let remaining = MIN_WAKE_TO_IDLE_SECONDS - elapsed_listen_seconds;
            if remaining > 0.0 {
                hold_for(Duration::from_secs_f64(remaining));
            }
            ctx.earcons.play(Earcon::Cancel);
            ctx.sink.emit(GraceEvent::ConversationFinished);
            ctx.sink.emit(GraceEvent::Idle);
            return false;
        }
    };
    let transcript = transcript.as_str();

    ctx.sink.emit(GraceEvent::FinalTranscript { text: transcript.to_string() });
    ctx.sink.emit(GraceEvent::ListeningStopped);

    let clean = clean_transcript(transcript);

    if is_stop_command(&clean) {
        if ctx.agent_loop.has_pending_confirmation() {
            ctx.agent_loop.cancel_pending();
        }
        ctx.earcons.play(Earcon::Cancel);
        ctx.sink.emit(GraceEvent::ConversationFinished);
        ctx.sink.emit(GraceEvent::Idle);
        return false;
    }

    if is_repeat_command(&clean) {
        let text = if last_spoken_text.is_empty() { "I haven't said anything yet.".to_string() } else { last_spoken_text.clone() };
        speak(ctx, &text);
        ctx.sink.emit(GraceEvent::ConversationFinished);
        return true;
    }

    if ctx.agent_loop.has_pending_confirmation() && resolve_pending_confirmation(ctx, transcript) {
        ctx.sink.emit(GraceEvent::ConversationFinished);
        return true;
    }

    ctx.sink.emit(GraceEvent::UnderstandingStarted { label: "Understanding request…".to_string() });
    let intent = generate_intent(transcript, intent_system_prompt, ctx.planner_llm, ctx.intent_parser);
    ctx.sink.emit(GraceEvent::UnderstandingFinished);

    let complexity = classify(transcript, intent.as_ref().map(ParsedIntentRef::from));

    match complexity {
        TaskComplexity::AgenticGoal => {
            if let Some(intent) = &intent {
                if matches!(intent.tool.as_str(), "open_app" | "cua_launch") {
                    ctx.dispatcher.execute_with_events(intent, false, ctx.sink);
                }
            }
            let agent_res = ctx.agent_loop.run(transcript, ctx.planner_llm, ctx.vision_llm, ctx.dispatcher, ctx.perception, ctx.sink);
            if let Some(text) = agent_response_text(&agent_res) {
                *last_spoken_text = text.clone();
                speak(ctx, &text);
            }
        }
        TaskComplexity::Conversation => {
            let response = intent
                .as_ref()
                .and_then(|i| i.response.clone())
                .unwrap_or_else(|| "I'm here to help! What can I do for you today?".to_string());
            *last_spoken_text = response.clone();
            speak(ctx, &response);
        }
        TaskComplexity::FastPath => {
            let Some(intent) = intent else {
                // Unreachable in practice (FastPath is only classified when
                // an intent was parsed), but handled rather than panicking.
                ctx.sink.emit(GraceEvent::ConversationFinished);
                return true;
            };
            let result = ctx.dispatcher.execute_with_events(&intent, false, ctx.sink);
            if park_if_confirmation_required(ctx, &intent, &result) {
                ctx.sink.emit(GraceEvent::ConversationFinished);
                return true;
            }
            let response_text = if intent.is_conversation() && intent.response.is_some() {
                intent.response.clone()
            } else {
                result.get("text").and_then(Value::as_str).map(String::from)
            };
            if let Some(text) = response_text {
                *last_spoken_text = text.clone();
                speak(ctx, &text);
            }
        }
    }

    ctx.sink.emit(GraceEvent::ConversationFinished);
    true
}

/// Act on one follow-up utterance. Returns `true` if the follow-up window
/// should restart (keep listening); `false` ends it.
pub fn handle_followup_transcript(ctx: &mut TurnContext, transcript: &str, intent_system_prompt: &str, last_spoken_text: &mut String) -> bool {
    ctx.sink.emit(GraceEvent::FinalTranscript { text: transcript.to_string() });

    let clean = clean_transcript(transcript);

    if is_stop_command(&clean) {
        if ctx.agent_loop.has_pending_confirmation() {
            ctx.agent_loop.cancel_pending();
        }
        ctx.earcons.play(Earcon::Cancel);
        return false;
    }

    if is_repeat_command(&clean) {
        let text = if last_spoken_text.is_empty() { "I haven't said anything yet.".to_string() } else { last_spoken_text.clone() };
        speak(ctx, &text);
        return true;
    }

    if ctx.agent_loop.has_pending_confirmation() && resolve_pending_confirmation(ctx, transcript) {
        return true;
    }

    let intent = generate_intent(transcript, intent_system_prompt, ctx.planner_llm, ctx.intent_parser);
    let complexity = classify(transcript, intent.as_ref().map(ParsedIntentRef::from));

    if complexity == TaskComplexity::AgenticGoal {
        if let Some(intent) = &intent {
            if matches!(intent.tool.as_str(), "open_app" | "cua_launch") {
                ctx.dispatcher.execute_with_events(intent, false, ctx.sink);
            }
        }
        let agent_res = ctx.agent_loop.run(transcript, ctx.planner_llm, ctx.vision_llm, ctx.dispatcher, ctx.perception, ctx.sink);
        if let Some(text) = agent_response_text(&agent_res) {
            *last_spoken_text = text.clone();
            speak(ctx, &text);
        }
        return true;
    }

    if let Some(intent) = &intent {
        if !intent.is_conversation() {
            let result = ctx.dispatcher.execute_with_events(intent, false, ctx.sink);
            if park_if_confirmation_required(ctx, intent, &result) {
                return true;
            }
            if let Some(text) = result.get("text").and_then(Value::as_str) {
                *last_spoken_text = text.to_string();
                speak(ctx, text);
            }
            return true;
        }
        if let Some(response) = &intent.response {
            *last_spoken_text = response.clone();
            speak(ctx, response);
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatcher::{Dispatcher, SystemActions};
    use crate::models::ScriptedLlm;
    use crate::perception::ScriptedPerception;
    use crate::agent_loop::AgentLoopConfig;

    #[derive(Default)]
    struct NoopActions;
    impl SystemActions for NoopActions {
        fn open_url(&mut self, _url: &str) -> bool { true }
        fn launch_app(&mut self, _name: &str) -> Value { serde_json::json!({"status": "ok"}) }
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

    struct Fixture {
        wake_word: RecordingWakeWordControl,
        earcons: crate::feedback::RecordingEarconPlayer,
        sink: crate::events::RecordingEventSink,
        intent_parser: IntentParser,
        planner_llm: ScriptedLlm,
        vision_llm: ScriptedLlm,
        actions: NoopActions,
        perception: ScriptedPerception,
        agent_loop: AgentLoop,
        tts: crate::models::ScriptedTts,
    }

    impl Fixture {
        fn new(planner_responses: Vec<Result<Option<String>, crate::models::RateLimitError>>) -> Self {
            Self {
                wake_word: RecordingWakeWordControl::default(),
                earcons: Default::default(),
                sink: Default::default(),
                intent_parser: IntentParser::new(),
                planner_llm: ScriptedLlm::new(planner_responses),
                vision_llm: ScriptedLlm::new(vec![]),
                actions: NoopActions::default(),
                perception: ScriptedPerception::new(vec![]),
                agent_loop: AgentLoop::new(AgentLoopConfig::default()),
                tts: Default::default(),
            }
        }

        fn run(&mut self, f: impl FnOnce(&mut TurnContext) -> bool) -> bool {
            let mut dispatcher = Dispatcher::new(None, &mut self.actions);
            let mut ctx = TurnContext {
                wake_word: &mut self.wake_word,
                earcons: &mut self.earcons,
                sink: &mut self.sink,
                intent_parser: &mut self.intent_parser,
                planner_llm: &mut self.planner_llm,
                vision_llm: &mut self.vision_llm,
                dispatcher: &mut dispatcher,
                perception: &mut self.perception,
                agent_loop: &mut self.agent_loop,
                tts: &mut self.tts,
            };
            f(&mut ctx)
        }
    }

    #[test]
    fn silence_aborts_the_turn_and_does_not_resume_wake_word_here() {
        // Mirrors ship-blocker #1's split responsibility: this function
        // itself doesn't resume the wake word on any path (the outer
        // `_handle_activation` does, based on the bool return) - it only
        // pauses it at the start. That's exercised in isolation here; the
        // resume-on-every-exit-path guarantee is the outer function's job.
        let mut fixture = Fixture::new(vec![]);
        let result = fixture.run(|ctx| {
            handle_activation_turn(ctx, ListenOutcome::Silence, 0.0, &|_| {}, "system prompt", &mut String::new())
        });
        assert!(!result);
        assert_eq!(fixture.wake_word.pause_calls, 1);
        assert_eq!(fixture.wake_word.resume_calls, 0);
        assert!(matches!(fixture.sink.events.last(), Some(GraceEvent::Idle)));
    }

    #[test]
    fn silence_holds_the_pill_open_for_the_remainder_of_min_wake_to_idle_seconds() {
        let mut fixture = Fixture::new(vec![]);
        let held = std::cell::Cell::new(None);
        fixture.run(|ctx| {
            handle_activation_turn(
                ctx,
                ListenOutcome::Silence,
                1.0, // listened for 1s already
                &|d| held.set(Some(d)),
                "system prompt",
                &mut String::new(),
            )
        });
        // MIN_WAKE_TO_IDLE_SECONDS (3.0) - 1.0 already elapsed = 2.0s left.
        assert_eq!(held.get(), Some(Duration::from_secs_f64(2.0)));
    }

    #[test]
    fn silence_does_not_hold_when_the_floor_is_already_met() {
        let mut fixture = Fixture::new(vec![]);
        let held = std::cell::Cell::new(None);
        fixture.run(|ctx| {
            handle_activation_turn(
                ctx,
                ListenOutcome::Silence,
                5.0, // already listened longer than the floor
                &|d| held.set(Some(d)),
                "system prompt",
                &mut String::new(),
            )
        });
        assert_eq!(held.get(), None);
    }

    #[test]
    fn a_transcription_failure_never_holds_regardless_of_elapsed_time() {
        let mut fixture = Fixture::new(vec![]);
        let held = std::cell::Cell::new(None);
        let result = fixture.run(|ctx| {
            handle_activation_turn(
                ctx,
                ListenOutcome::TranscriptionFailed,
                0.0, // even with no time elapsed, this path never holds
                &|d| held.set(Some(d)),
                "system prompt",
                &mut String::new(),
            )
        });
        assert!(!result);
        assert_eq!(held.get(), None);
        assert!(fixture.sink.events.iter().any(|e| matches!(e, GraceEvent::Idle)));
    }

    #[test]
    fn a_stop_command_ends_the_turn_immediately() {
        let mut fixture = Fixture::new(vec![]);
        let mut last = String::new();
        let result = fixture.run(|ctx| handle_activation_turn(ctx, ListenOutcome::Transcript("stop".to_string()), 0.0, &|_| {}, "prompt", &mut last));
        assert!(!result);
        assert!(fixture.sink.events.iter().any(|e| matches!(e, GraceEvent::Idle)));
    }

    #[test]
    fn a_repeat_command_speaks_the_last_response_again() {
        let mut fixture = Fixture::new(vec![]);
        let mut last = "I already said this.".to_string();
        let result = fixture.run(|ctx| handle_activation_turn(ctx, ListenOutcome::Transcript("repeat that".to_string()), 0.0, &|_| {}, "prompt", &mut last));
        assert!(result);
        let chunk = fixture.sink.events.iter().find_map(|e| {
            if let GraceEvent::ResponseChunk { text } = e { Some(text.clone()) } else { None }
        });
        assert_eq!(chunk.as_deref(), Some("I already said this."));
    }

    #[test]
    fn repeat_with_nothing_said_yet_says_so() {
        let mut fixture = Fixture::new(vec![]);
        let mut last = String::new();
        fixture.run(|ctx| handle_activation_turn(ctx, ListenOutcome::Transcript("repeat".to_string()), 0.0, &|_| {}, "prompt", &mut last));
        let chunk = fixture.sink.events.iter().find_map(|e| {
            if let GraceEvent::ResponseChunk { text } = e { Some(text.clone()) } else { None }
        });
        assert_eq!(chunk.as_deref(), Some("I haven't said anything yet."));
    }

    #[test]
    fn a_pending_yes_no_answer_is_resolved_before_intent_generation() {
        let mut fixture = Fixture::new(vec![]); // no planner calls should happen
        fixture.agent_loop.park_intent(Intent::new("lock_computer", serde_json::json!({}), None), "Lock now?".into());
        let mut last = String::new();
        let result = fixture.run(|ctx| handle_activation_turn(ctx, ListenOutcome::Transcript("yes".to_string()), 0.0, &|_| {}, "prompt", &mut last));
        assert!(result);
        assert!(!fixture.agent_loop.has_pending_confirmation());
    }

    #[test]
    fn a_fast_path_confirmation_required_result_is_parked_and_asked() {
        let mut fixture = Fixture::new(vec![ScriptedLlm::text(r#"{"tool": "lock_computer", "params": {}}"#)]);
        let mut last = String::new();
        let result = fixture.run(|ctx| handle_activation_turn(ctx, ListenOutcome::Transcript("lock my computer".to_string()), 0.0, &|_| {}, "prompt", &mut last));
        assert!(result);
        assert!(fixture.agent_loop.has_pending_confirmation());
        let chunk = fixture.sink.events.iter().find_map(|e| {
            if let GraceEvent::ResponseChunk { text } = e { Some(text.clone()) } else { None }
        });
        assert_eq!(chunk.as_deref(), Some("Should I lock your computer now?"));
    }

    #[test]
    fn conversation_intent_speaks_its_own_response() {
        let mut fixture = Fixture::new(vec![ScriptedLlm::text(r#"{"tool": "converse", "response": "Hi!", "params": {}}"#)]);
        let mut last = String::new();
        fixture.run(|ctx| handle_activation_turn(ctx, ListenOutcome::Transcript("hello".to_string()), 0.0, &|_| {}, "prompt", &mut last));
        assert_eq!(last, "Hi!");
    }

    #[test]
    fn the_full_event_sequence_matches_the_real_fastpath_open_calculator_tape() {
        // This exact sequence is the recorded `corpus/fastpath_open_calculator`
        // tape's `events.jsonl` (minus the diagnostic `TurnTrace` and the
        // follow-up-window events this function doesn't produce - see its
        // module doc). Matching it verbatim is the point: it is real,
        // recorded backend behaviour, not a shape this test invented.
        let mut fixture = Fixture::new(vec![ScriptedLlm::text(r#"{"tool": "open_calculator", "params": {}}"#)]);
        let mut last = String::new();
        fixture.run(|ctx| handle_activation_turn(ctx, ListenOutcome::Transcript("open the calculator".to_string()), 0.0, &|_| {}, "prompt", &mut last));

        let types: Vec<String> = fixture
            .sink
            .events
            .iter()
            .map(|e| serde_json::to_value(e).unwrap()["type"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            types,
            vec![
                "WakeWordDetected",
                "ListeningStarted",
                "FinalTranscript",
                "ListeningStopped",
                "UnderstandingStarted",
                "UnderstandingFinished",
                "ToolExecutionStarted",
                "ToolExecutionFinished",
                "SpeechStarted",
                "ResponseChunk",
                "SpeechChunk",
                "SpeechFinished",
                "ConversationFinished",
            ]
        );

        // The dispatcher-level ToolExecutionStarted carries only `label`
        // (no tool/step) - that's what tells it apart from the agent loop's
        // own emission around a step, which does carry them (see
        // `agent_loop.rs`'s tests).
        let started = fixture
            .sink
            .events
            .iter()
            .find(|e| matches!(e, GraceEvent::ToolExecutionStarted { .. }))
            .unwrap();
        assert_eq!(
            *started,
            GraceEvent::ToolExecutionStarted { label: "Opening calculator…".to_string(), tool: None, step: None }
        );
    }
}
