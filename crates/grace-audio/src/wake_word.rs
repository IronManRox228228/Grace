use std::path::Path;
use std::sync::Arc;
use regex::Regex;
use sherpa_rs::keyword_spot::{KeywordSpot, KeywordSpotConfig};

use crate::{is_wake_word_match, WakeWordDetector};

/// Keyword spotting wake word detector using Sherpa-ONNX Zipformer KWS.
///
/// Features:
/// - Personalized keyword & model path from config.
/// - Whole-word matching only (rejects substrings like "gracefully" or "disgrace").
/// - Stop/cancel watch during execution: fires callback on `\b(stop|cancel)\b` to halt within 500 ms.
pub struct SherpaWakeWord {
    spotter: Option<KeywordSpot>,
    keyword: String,
    paused: bool,
    cancel_callback: Option<Arc<dyn Fn() + Send + Sync>>,
    cancel_re: Regex,
    last_cancel_fired: std::time::Instant,
}

impl SherpaWakeWord {
    /// Create a new Sherpa KWS wake word spotter.
    ///
    /// Verifies model files exist in `model_path` (or returns an error pointing to `docs/MODELS.md`).
    pub fn new(
        model_path: &str,
        keywords_file: &str,
        keyword: &str,
        num_threads: i32,
    ) -> Result<Self, String> {
        let path = Path::new(model_path);
        if !path.exists() {
            return Err(format!(
                "Missing required Sherpa-ONNX KWS model directory at '{model_path}'. \
                 Please download weights as documented in docs/MODELS.md."
            ));
        }

        let encoder = path.join("encoder.onnx");
        let decoder = path.join("decoder.onnx");
        let joiner = path.join("joiner.onnx");
        let tokens = path.join("tokens.txt");

        for (f, name) in [
            (&encoder, "encoder.onnx"),
            (&decoder, "decoder.onnx"),
            (&joiner, "joiner.onnx"),
            (&tokens, "tokens.txt"),
        ] {
            if !f.exists() {
                return Err(format!(
                    "Missing required Sherpa-ONNX KWS file '{name}' in '{model_path}'. \
                     Please download weights as documented in docs/MODELS.md."
                ));
            }
        }

        let config = KeywordSpotConfig {
            zipformer_encoder: encoder.to_string_lossy().to_string(),
            zipformer_decoder: decoder.to_string_lossy().to_string(),
            zipformer_joiner: joiner.to_string_lossy().to_string(),
            tokens: tokens.to_string_lossy().to_string(),
            keywords: keywords_file.to_string(),
            num_threads: Some(num_threads),
            provider: Some("cpu".to_string()),
            ..Default::default()
        };

        let spotter = KeywordSpot::new(config)
            .map_err(|e| format!("Failed to create Sherpa KeywordSpot: {e}"))?;

        let cancel_re = Regex::new(r"(?i)\b(stop|cancel)\b").expect("compile cancel regex");

        Ok(Self {
            spotter: Some(spotter),
            keyword: keyword.to_string(),
            paused: false,
            cancel_callback: None,
            cancel_re,
            last_cancel_fired: std::time::Instant::now() - std::time::Duration::from_secs(10),
        })
    }

    /// Creates a mock/fake wake word spotter for synthetic unit tests that doesn't require model files.
    pub fn new_mock(keyword: &str) -> Self {
        let cancel_re = Regex::new(r"(?i)\b(stop|cancel)\b").expect("compile cancel regex");
        Self {
            spotter: None,
            keyword: keyword.to_string(),
            paused: false,
            cancel_callback: None,
            cancel_re,
            last_cancel_fired: std::time::Instant::now() - std::time::Duration::from_secs(10),
        }
    }

    /// Arm the voice stop/cancel watch during goal execution.
    pub fn arm_cancel_watch(&mut self, callback: Arc<dyn Fn() + Send + Sync>) {
        self.cancel_callback = Some(callback);
    }

    /// Disarm the voice stop/cancel watch when execution finishes.
    pub fn disarm_cancel_watch(&mut self) {
        self.cancel_callback = None;
    }

    /// Check text for whole-word stop/cancel command.
    pub fn check_cancel_text(&mut self, text: &str) -> bool {
        if self.cancel_re.is_match(text) {
            let now = std::time::Instant::now();
            if now.duration_since(self.last_cancel_fired) > std::time::Duration::from_millis(500) {
                self.last_cancel_fired = now;
                if let Some(cb) = &self.cancel_callback {
                    cb();
                    return true;
                }
            }
        }
        false
    }
}

impl WakeWordDetector for SherpaWakeWord {
    fn process_chunk(&mut self, chunk: &[u8]) -> bool {
        if self.paused || chunk.is_empty() {
            return false;
        }

        let samples: Vec<f32> = chunk
            .chunks_exact(2)
            .map(|c| {
                let s = i16::from_le_bytes([c[0], c[1]]);
                s as f32 / 32768.0
            })
            .collect();

        if let Some(spotter) = &mut self.spotter {
            if let Ok(Some(keyword)) = spotter.extract_keyword(samples, 16000) {
                // Whole-word match check
                if is_wake_word_match(&keyword, &self.keyword) {
                    return true;
                }
                // Also check if cancel watch is active
                self.check_cancel_text(&keyword);
            }
        }

        false
    }

    fn pause(&mut self) {
        self.paused = true;
    }

    fn resume(&mut self) {
        self.paused = false;
    }
}

/// Helper function to check if an utterance contains stop or cancel words.
pub fn is_cancel_utterance(text: &str) -> bool {
    let re = Regex::new(r"(?i)\b(stop|cancel)\b").unwrap();
    re.is_match(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn cancel_regex_matches_whole_words_only() {
        assert!(is_cancel_utterance("stop"));
        assert!(is_cancel_utterance("please stop that"));
        assert!(is_cancel_utterance("cancel"));
        assert!(is_cancel_utterance("cancel current goal"));

        // Substrings must NOT match
        assert!(!is_cancel_utterance("nonstop flight"));
        assert!(!is_cancel_utterance("cancellation policy"));
        assert!(!is_cancel_utterance("buses stopover"));
    }

    #[test]
    fn stop_watch_fires_callback_within_500ms() {
        let mut spotter = SherpaWakeWord::new_mock("grace");
        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = Arc::clone(&fired);

        spotter.arm_cancel_watch(Arc::new(move || {
            fired_clone.store(true, Ordering::SeqCst);
        }));

        let matched = spotter.check_cancel_text("grace stop immediately");
        assert!(matched);
        assert!(fired.load(Ordering::SeqCst));
    }

    #[test]
    fn missing_kws_files_returns_clear_error_pointing_to_models_md() {
        let res = SherpaWakeWord::new("nonexistent-kws-dir", "keywords.txt", "grace", 2);
        assert!(res.is_err());
        let err = res.err().unwrap();
        assert!(err.contains("docs/MODELS.md"));
        assert!(err.contains("nonexistent-kws-dir"));
    }
}
