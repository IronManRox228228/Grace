//! Concrete model backends. The traits themselves
//! (`LargeLanguageModel`, `TextToSpeech`, `SpeechToText`) live in
//! `grace_core::models`, not here, so `grace-core`'s pure orchestration
//! logic (planner, grounder, agent loop, response generator) can depend on
//! the trait without depending on any concrete backend.
//!
//! Per PLAN.md §11, the intended runtimes are `sherpa-rs`/sherpa-onnx
//! (STT/TTS/VAD/keyword spotting), llama.cpp (server or bindings) and `ort`
//! (ONNX Runtime). Native bindings for those are NOT wired up in this phase
//! - see PORT_STATUS.md for why and what's next. What IS here: `HttpLlm`, a
//! real (not stubbed) HTTP client matching the non-streaming request shape
//! `src/grace/llm/gemma_client.py`'s `GemmaClient.chat(..., stream=False)`
//! sends to a local llama.cpp server's OpenAI-compatible endpoint - the
//! offline-first path PLAN.md's CPU-floor stance actually targets. The
//! Gemini cloud streaming path is not implemented (optional per PLAN.md
//! §1.2: "no required dependency on cloud LLMs").
//!
//! A deliberate simplification, noted in PORT_STATUS.md: every trait method
//! here is synchronous. Python's async is used for I/O concurrency;
//! nothing this port grades (event order, decision points, budgets) depends
//! on that concurrency, so the orchestration logic in `grace-core` is
//! ordinary synchronous Rust. `HttpLlm` uses `reqwest::blocking`
//! accordingly - a real network call, just not a `.await`ed one.

use grace_core::models::{LargeLanguageModel, LlmRequest, RateLimitError};
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub use grace_core::models::{ScriptedLlm, ScriptedStt, ScriptedTts, SpeechToText, TextToSpeech};

/// One transcription result. Mirrors the shape `stt.jsonl` tape entries and
/// `FinalTranscript` events carry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    pub text: String,
}

/// One candidate element the decision model scores, as a short summary -
/// never the whole tree per element (PLAN.md §3's "split of
/// responsibilities").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementCandidate {
    pub id: String,
    pub name: String,
    pub control_type: String,
    pub parent_path: String,
    pub window_title: String,
}

/// Element-selection scoring (GLiNER2.5-Decide / Laya in PLAN.md §3). Not
/// implemented against a real model in this phase; kept as a trait so a
/// future ONNX/`ort`-backed scorer has a fixed seam to implement.
pub trait ElementScorer: Send {
    fn score(&mut self, instruction: &str, candidates: &[ElementCandidate]) -> anyhow::Result<Vec<f32>>;
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: &'static str,
    content: ChatContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ChatContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ContentPart {
    Text { r#type: &'static str, text: String },
    Image { r#type: &'static str, image_url: ImageUrl },
}

#[derive(Debug, Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
    stream: bool,
    cache_prompt: bool,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChatChoiceMessage {
    #[serde(default)]
    content: Option<String>,
}

/// An HTTP-backed `LargeLanguageModel` talking to a local llama.cpp
/// server's OpenAI-compatible `/v1/chat/completions` endpoint, non-streaming
/// (`"stream": false`) - the same request shape
/// `GemmaClient.chat(messages, stream=False)` sends in
/// `src/grace/llm/gemma_client.py:195-231`, including the `image_b64` ->
/// `image_url` data-URI reshaping for a multimodal grounding call.
pub struct HttpLlm {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl HttpLlm {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("reqwest client builds"),
        }
    }

    fn build_messages(request: &LlmRequest) -> Vec<ChatMessage> {
        let mut messages = Vec::new();
        if let Some(system) = &request.system_prompt {
            messages.push(ChatMessage { role: "system", content: ChatContent::Text(system.clone()) });
        }
        let content = match &request.image_b64 {
            Some(image) => ChatContent::Parts(vec![
                ContentPart::Text { r#type: "text", text: request.prompt.clone() },
                ContentPart::Image {
                    r#type: "image_url",
                    image_url: ImageUrl { url: format!("data:image/png;base64,{image}") },
                },
            ]),
            None => ChatContent::Text(request.prompt.clone()),
        };
        messages.push(ChatMessage { role: "user", content });
        messages
    }
}

impl LargeLanguageModel for HttpLlm {
    fn generate_text(&mut self, request: &LlmRequest) -> Result<Option<String>, RateLimitError> {
        let payload = ChatRequest {
            messages: Self::build_messages(request),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: false,
            cache_prompt: true,
        };

        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = match self.client.post(&url).json(&payload).send() {
            Ok(resp) => resp,
            // Mirrors `GemmaClient.chat`'s non-streaming branch: any transport
            // failure is logged (here: swallowed - callers don't have a
            // logger handle) and returns `None`, never an error the loop
            // would treat as a rate limit.
            Err(_) => return Ok(None),
        };

        if response.status().as_u16() == 429 {
            let body = response.text().unwrap_or_default();
            return Err(RateLimitError(format!("HTTP 429: {}", &body[..body.len().min(150)])));
        }
        if !response.status().is_success() {
            return Ok(None);
        }

        match response.json::<ChatResponse>() {
            Ok(parsed) => Ok(parsed
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.message.content)),
            Err(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_messages_includes_system_and_user() {
        let req = LlmRequest {
            prompt: "hello".into(),
            system_prompt: Some("be nice".into()),
            ..LlmRequest::new("hello")
        };
        let messages = HttpLlm::build_messages(&req);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
    }

    #[test]
    fn build_messages_reshapes_an_image_into_parts() {
        let req = LlmRequest {
            image_b64: Some("Zm9v".into()),
            ..LlmRequest::new("locate the button")
        };
        let messages = HttpLlm::build_messages(&req);
        match &messages.last().unwrap().content {
            ChatContent::Parts(parts) => assert_eq!(parts.len(), 2),
            ChatContent::Text(_) => panic!("expected multimodal parts"),
        }
    }

    #[test]
    fn unreachable_server_returns_ok_none_not_an_error() {
        // Mirrors GemmaClient.chat's non-streaming branch: a transport
        // failure against a local llama-server that isn't running must read
        // to the caller exactly like Python's `None` return, not as a rate
        // limit or a panic.
        let mut llm = HttpLlm::new("http://127.0.0.1:1");
        let result = llm.generate_text(&LlmRequest::new("hi"));
        assert_eq!(result, Ok(None));
    }
}
