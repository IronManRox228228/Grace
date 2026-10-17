use std::ffi::{CStr, CString};
use std::mem;
use std::path::Path;
use std::ptr::null;
use grace_core::models::SpeechToText;
use sherpa_rs::moonshine::{MoonshineConfig, MoonshineRecognizer};
use sherpa_rs::sherpa_rs_sys;

/// Offline speech-to-text recognizer supporting Parakeet-TDT and Moonshine Base via Sherpa-ONNX.
pub enum SherpaAsrEngine {
    Moonshine(MoonshineRecognizer),
    Parakeet {
        recognizer: *const sherpa_rs_sys::SherpaOnnxOfflineRecognizer,
    },
    Mock {
        canned_response: String,
    },
}

unsafe impl Send for SherpaAsrEngine {}

impl Drop for SherpaAsrEngine {
    fn drop(&mut self) {
        if let SherpaAsrEngine::Parakeet { recognizer } = self {
            if !recognizer.is_null() {
                unsafe {
                    sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizer(*recognizer);
                }
            }
        }
    }
}

pub struct SherpaAsr {
    engine: SherpaAsrEngine,
    sample_rate: u32,
}

impl SherpaAsr {
    /// Initialise Moonshine Base ASR (optimized for CPU Floor tier).
    pub fn new_moonshine(model_dir: &str, num_threads: i32) -> Result<Self, String> {
        let dir = Path::new(model_dir);
        if !dir.exists() {
            return Err(format!(
                "Missing required Sherpa-ONNX Moonshine model directory at '{model_dir}'. \
                 Please download weights as documented in docs/MODELS.md."
            ));
        }

        let preprocessor = dir.join("preprocess.onnx");
        let encoder = dir.join("encode.int8.onnx");
        let uncached_decoder = dir.join("uncached_decode.int8.onnx");
        let cached_decoder = dir.join("cached_decode.int8.onnx");
        let tokens = dir.join("tokens.txt");

        for (f, name) in [
            (&preprocessor, "preprocess.onnx"),
            (&encoder, "encode.int8.onnx"),
            (&uncached_decoder, "uncached_decode.int8.onnx"),
            (&cached_decoder, "cached_decode.int8.onnx"),
            (&tokens, "tokens.txt"),
        ] {
            if !f.exists() {
                return Err(format!(
                    "Missing required Moonshine file '{name}' in '{model_dir}'. \
                     Please download weights as documented in docs/MODELS.md."
                ));
            }
        }

        let config = MoonshineConfig {
            preprocessor: preprocessor.to_string_lossy().to_string(),
            encoder: encoder.to_string_lossy().to_string(),
            uncached_decoder: uncached_decoder.to_string_lossy().to_string(),
            cached_decoder: cached_decoder.to_string_lossy().to_string(),
            tokens: tokens.to_string_lossy().to_string(),
            num_threads: Some(num_threads),
            provider: Some("cpu".to_string()),
            debug: false,
        };

        let rec = MoonshineRecognizer::new(config)
            .map_err(|e| format!("Failed to create MoonshineRecognizer: {e}"))?;

        Ok(Self {
            engine: SherpaAsrEngine::Moonshine(rec),
            sample_rate: 16000,
        })
    }

    /// Initialise Parakeet ASR (NeMo CTC int8, high accuracy pick).
    pub fn new_parakeet(model_dir: &str, num_threads: i32) -> Result<Self, String> {
        let dir = Path::new(model_dir);
        if !dir.exists() {
            return Err(format!(
                "Missing required Sherpa-ONNX Parakeet model directory at '{model_dir}'. \
                 Please download weights as documented in docs/MODELS.md."
            ));
        }

        let model = if dir.join("model.int8.onnx").exists() {
            dir.join("model.int8.onnx")
        } else {
            dir.join("model.onnx")
        };
        let tokens = dir.join("tokens.txt");

        if !model.exists() || !tokens.exists() {
            return Err(format!(
                "Missing required Parakeet model files (model.int8.onnx, tokens.txt) in '{model_dir}'. \
                 Please download weights as documented in docs/MODELS.md."
            ));
        }

        let model_c = CString::new(model.to_string_lossy().to_string()).unwrap();
        let tokens_c = CString::new(tokens.to_string_lossy().to_string()).unwrap();
        let provider_c = CString::new("cpu").unwrap();

        unsafe {
            let model_config = sherpa_rs_sys::SherpaOnnxOfflineModelConfig {
                debug: 0,
                num_threads,
                provider: provider_c.as_ptr(),
                tokens: tokens_c.as_ptr(),
                nemo_ctc: sherpa_rs_sys::SherpaOnnxOfflineNemoEncDecCtcModelConfig {
                    model: model_c.as_ptr(),
                },
                model_type: null(),
                modeling_unit: null(),
                bpe_vocab: null(),
                telespeech_ctc: null(),
                sense_voice: mem::zeroed(),
                moonshine: mem::zeroed(),
                fire_red_asr: mem::zeroed(),
                dolphin: mem::zeroed(),
                zipformer_ctc: mem::zeroed(),
                canary: mem::zeroed(),
                paraformer: mem::zeroed(),
                tdnn: mem::zeroed(),
                transducer: mem::zeroed(),
                whisper: mem::zeroed(),
            };

            let config = sherpa_rs_sys::SherpaOnnxOfflineRecognizerConfig {
                decoding_method: null(),
                feat_config: sherpa_rs_sys::SherpaOnnxFeatureConfig {
                    sample_rate: 16000,
                    feature_dim: 80,
                },
                hotwords_file: null(),
                hotwords_score: 0.0,
                lm_config: sherpa_rs_sys::SherpaOnnxOfflineLMConfig {
                    model: null(),
                    scale: 0.0,
                },
                max_active_paths: 0,
                model_config,
                rule_fars: null(),
                rule_fsts: null(),
                blank_penalty: 0.0,
                hr: mem::zeroed(),
            };

            let recognizer = sherpa_rs_sys::SherpaOnnxCreateOfflineRecognizer(&config);
            if recognizer.is_null() {
                return Err("Failed to create SherpaOnnx Parakeet OfflineRecognizer".to_string());
            }

            Ok(Self {
                engine: SherpaAsrEngine::Parakeet { recognizer },
                sample_rate: 16000,
            })
        }
    }

    /// Create a synthetic mock ASR for tests without model files.
    pub fn new_mock(canned_response: &str) -> Self {
        Self {
            engine: SherpaAsrEngine::Mock {
                canned_response: canned_response.to_string(),
            },
            sample_rate: 16000,
        }
    }
}

impl SpeechToText for SherpaAsr {
    fn transcribe(&mut self, pcm16_mono: &[u8]) -> String {
        if pcm16_mono.is_empty() {
            return String::new();
        }

        let samples: Vec<f32> = pcm16_mono
            .chunks_exact(2)
            .map(|c| {
                let s = i16::from_le_bytes([c[0], c[1]]);
                s as f32 / 32768.0
            })
            .collect();

        match &mut self.engine {
            SherpaAsrEngine::Moonshine(rec) => {
                let res = rec.transcribe(self.sample_rate, &samples);
                res.text.trim().to_string()
            }
            SherpaAsrEngine::Parakeet { recognizer } => unsafe {
                let stream = sherpa_rs_sys::SherpaOnnxCreateOfflineStream(*recognizer);
                sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                    stream,
                    self.sample_rate as i32,
                    samples.as_ptr(),
                    samples.len() as i32,
                );
                sherpa_rs_sys::SherpaOnnxDecodeOfflineStream(*recognizer, stream);
                let res_ptr = sherpa_rs_sys::SherpaOnnxGetOfflineStreamResult(stream);
                let text = if !res_ptr.is_null() {
                    let cstr = CStr::from_ptr((*res_ptr).text);
                    let s = cstr.to_string_lossy().into_owned();
                    sherpa_rs_sys::SherpaOnnxDestroyOfflineRecognizerResult(res_ptr);
                    s
                } else {
                    String::new()
                };
                sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
                text.trim().to_string()
            },
            SherpaAsrEngine::Mock { canned_response } => canned_response.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_dir_reports_models_md() {
        let res = SherpaAsr::new_moonshine("nonexistent-moonshine-dir", 2);
        assert!(res.is_err());
        let err = res.err().unwrap();
        assert!(err.contains("docs/MODELS.md"));

        let res2 = SherpaAsr::new_parakeet("nonexistent-parakeet-dir", 2);
        assert!(res2.is_err());
        let err2 = res2.err().unwrap();
        assert!(err2.contains("docs/MODELS.md"));
    }

    #[test]
    fn mock_asr_transcribes_canned_text() {
        let mut asr = SherpaAsr::new_mock("open the calculator");
        let fake_pcm = vec![0u8; 3200];
        let text = asr.transcribe(&fake_pcm);
        assert_eq!(text, "open the calculator");
    }
}
