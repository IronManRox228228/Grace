use std::path::Path;
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};

/// Voice Activity Detection powered by Sherpa-ONNX Silero VAD v5.
pub struct SherpaSileroVad {
    vad: SileroVad,
    pub sample_rate: u32,
}

impl SherpaSileroVad {
    /// Initialise Silero VAD from the given ONNX model file.
    ///
    /// If the model file is missing, returns an informative error pointing to `docs/MODELS.md`.
    pub fn new(model_path: &str, threshold: f32) -> Result<Self, String> {
        let p = Path::new(model_path);
        if !p.exists() {
            return Err(format!(
                "Missing required Sherpa-ONNX Silero VAD model at '{model_path}'. \
                 Please download weights as documented in docs/MODELS.md."
            ));
        }

        let config = SileroVadConfig {
            model: model_path.to_string(),
            threshold,
            min_silence_duration: 0.25,
            min_speech_duration: 0.25,
            max_speech_duration: 30.0,
            sample_rate: 16000,
            window_size: 512,
            provider: Some("cpu".to_string()),
            num_threads: Some(2),
            debug: false,
        };

        let vad = SileroVad::new(config, 30.0).map_err(|e| format!("Failed to create SileroVad: {e}"))?;

        Ok(Self {
            vad,
            sample_rate: 16000,
        })
    }

    /// Feed a 16-bit mono 16 kHz PCM audio chunk. Returns `true` if a speech boundary has been detected.
    pub fn process_chunk(&mut self, chunk: &[u8]) -> bool {
        if chunk.is_empty() {
            return false;
        }

        // Convert 16-bit PCM little-endian bytes to f32 samples in [-1.0, 1.0]
        let samples: Vec<f32> = chunk
            .chunks_exact(2)
            .map(|c| {
                let s = i16::from_le_bytes([c[0], c[1]]);
                s as f32 / 32768.0
            })
            .collect();

        self.vad.accept_waveform(samples);
        if !self.vad.is_speech() {
            return false;
        }
        !self.vad.is_empty()
    }

    /// Reset internal state between turns.
    pub fn reset(&mut self) {
        self.vad.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_file_returns_clear_error_pointing_to_models_md() {
        let res = SherpaSileroVad::new("nonexistent/silero_vad.onnx", 0.5);
        assert!(res.is_err());
        let err = res.err().unwrap();
        assert!(err.contains("docs/MODELS.md"));
        assert!(err.contains("nonexistent/silero_vad.onnx"));
    }
}
