//! Model-edge traits shared by `planner.rs`, `grounder.rs`,
//! `response_generator.rs` and `agent_loop.rs`. Defined here (rather than in
//! `grace-models`) so this crate's pure orchestration logic can depend on
//! the trait without depending on any concrete backend - `grace-models`
//! depends on `grace-core` and implements these traits, not the other way
//! round.
//!
//! Concrete backends (a real HTTP client, sherpa-onnx, llama.cpp) live in
//! `grace-models`; see that crate for what's implemented vs. stubbed.

/// One request to the tool-calling/planner/grounder LLM. Mirrors the
/// keyword arguments `GemmaClient.generate_text` takes in
/// `src/grace/llm/gemma_client.py`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LlmRequest {
    pub prompt: String,
    pub system_prompt: Option<String>,
    /// Base64-encoded image bytes, when this call includes a screenshot
    /// (the grounder always sets this; the planner only in blind mode).
    pub image_b64: Option<String>,
    /// `None` uses the backend's configured default model; `Some` is the
    /// escalation ladder's "ask a stronger model for this one call" path.
    pub model: Option<String>,
    pub temperature: f32,
    pub max_tokens: u32,
}

impl LlmRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self { prompt: prompt.into(), temperature: 0.1, max_tokens: 2048, ..Default::default() }
    }
}

/// Raised only for a rate limit - mirrors `RateLimitError` in
/// `gemma_client.py`, the one exception `generate_text` does NOT swallow
/// into a `None` return.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("rate limited: {0}")]
pub struct RateLimitError(pub String);

/// The tool-calling/planner/grounder LLM boundary.
///
/// `Ok(None)` is deliberately not an error type: it mirrors
/// `GemmaClient.generate_text` catching every exception except
/// `RateLimitError` and returning `None` (see `contract/README.md`'s "the
/// agent loop can spin without bound" finding - reproduced deliberately, not
/// fixed, by the callers of this trait). `Err` is reserved for the one case
/// Python does not swallow.
pub trait LargeLanguageModel: Send {
    fn generate_text(&mut self, request: &LlmRequest) -> Result<Option<String>, RateLimitError>;
}

/// A scripted `LargeLanguageModel` for tests: returns queued responses in
/// order, mirroring how `src/grace/harness/replay.py` pins
/// `_stream_gemini_response`. Lives here (not `grace-models`) so every test
/// in this crate can use it without a dependency that would be circular.
pub struct ScriptedLlm {
    responses: std::collections::VecDeque<Result<Option<String>, RateLimitError>>,
    pub requests_seen: Vec<LlmRequest>,
}

impl ScriptedLlm {
    pub fn new(responses: Vec<Result<Option<String>, RateLimitError>>) -> Self {
        Self { responses: responses.into(), requests_seen: Vec::new() }
    }

    pub fn text(text: impl Into<String>) -> Result<Option<String>, RateLimitError> {
        Ok(Some(text.into()))
    }
}

impl LargeLanguageModel for ScriptedLlm {
    fn generate_text(&mut self, request: &LlmRequest) -> Result<Option<String>, RateLimitError> {
        self.requests_seen.push(request.clone());
        self.responses.pop_front().unwrap_or(Ok(None))
    }
}

/// Text-to-speech: one sentence in, WAV bytes out (or `None` on synthesis
/// failure - mirrors Kokoro's `synthesize` returning `Optional[bytes]`).
pub trait TextToSpeech: Send {
    fn synthesize(&mut self, sentence: &str, voice: &str) -> Option<Vec<u8>>;
}

/// A scripted `TextToSpeech` for tests: returns non-empty bytes for any
/// sentence unless told to fail specific ones.
pub struct ScriptedTts {
    pub fail_sentences: std::collections::BTreeSet<String>,
    pub synthesized: Vec<(String, String)>,
}

impl Default for ScriptedTts {
    fn default() -> Self {
        Self { fail_sentences: Default::default(), synthesized: Vec::new() }
    }
}

impl TextToSpeech for ScriptedTts {
    fn synthesize(&mut self, sentence: &str, voice: &str) -> Option<Vec<u8>> {
        self.synthesized.push((sentence.to_string(), voice.to_string()));
        if self.fail_sentences.contains(sentence) {
            None
        } else {
            Some(format!("wav:{sentence}").into_bytes())
        }
    }
}

/// Speech-to-text: whole-utterance transcription (Grace has no streaming
/// transcript - see the event contract's `FinalTranscript` doc).
pub trait SpeechToText: Send {
    fn transcribe(&mut self, pcm16_mono: &[u8]) -> String;
}

/// A scripted `SpeechToText` for tests: returns queued transcripts in order.
pub struct ScriptedStt {
    transcripts: std::collections::VecDeque<String>,
}

impl ScriptedStt {
    pub fn new(transcripts: Vec<&str>) -> Self {
        Self { transcripts: transcripts.into_iter().map(String::from).collect() }
    }
}

impl SpeechToText for ScriptedStt {
    fn transcribe(&mut self, _pcm16_mono: &[u8]) -> String {
        self.transcripts.pop_front().unwrap_or_default()
    }
}
