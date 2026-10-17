use std::io::Cursor;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use grace_core::models::TextToSpeech;
use sherpa_rs::tts::{KokoroTts, KokoroTtsConfig};
use sherpa_rs::OnnxConfig;

use crate::capture::AudioPump;

/// Kokoro-82M offline text-to-speech engine using Sherpa-ONNX.
pub struct SherpaKokoroTts {
    tts: Option<KokoroTts>,
    speed: f32,
}

impl SherpaKokoroTts {
    /// Initialise Kokoro TTS from the given model directory.
    pub fn new(model_dir: &str, speed: f32, num_threads: i32) -> Result<Self, String> {
        let dir = Path::new(model_dir);
        if !dir.exists() {
            return Err(format!(
                "Missing required Sherpa-ONNX Kokoro model directory at '{model_dir}'. \
                 Please download weights as documented in docs/MODELS.md."
            ));
        }

        let model = dir.join("model.onnx");
        let voices = dir.join("voices.bin");
        let tokens = dir.join("tokens.txt");
        let data_dir = dir.join("espeak-ng-data");

        for (f, name) in [(&model, "model.onnx"), (&voices, "voices.bin"), (&tokens, "tokens.txt")] {
            if !f.exists() {
                return Err(format!(
                    "Missing required Kokoro file '{name}' in '{model_dir}'. \
                     Please download weights as documented in docs/MODELS.md."
                ));
            }
        }

        let config = KokoroTtsConfig {
            model: model.to_string_lossy().to_string(),
            voices: voices.to_string_lossy().to_string(),
            tokens: tokens.to_string_lossy().to_string(),
            data_dir: if data_dir.exists() {
                data_dir.to_string_lossy().to_string()
            } else {
                String::new()
            },
            length_scale: 1.0,
            onnx_config: OnnxConfig {
                provider: "cpu".to_string(),
                debug: false,
                num_threads,
            },
            ..Default::default()
        };

        let tts = KokoroTts::new(config);

        Ok(Self {
            tts: Some(tts),
            speed,
        })
    }

    /// Create a mock TTS instance for tests without model files.
    pub fn new_mock(speed: f32) -> Self {
        Self {
            tts: None,
            speed,
        }
    }
}

impl TextToSpeech for SherpaKokoroTts {
    fn synthesize(&mut self, sentence: &str, _voice: &str) -> Option<Vec<u8>> {
        if sentence.trim().is_empty() {
            return None;
        }

        if let Some(tts) = &mut self.tts {
            match tts.create(sentence.trim(), 0, self.speed) {
                Ok(audio) => {
                    // Encode f32 samples to 16-bit PCM WAV bytes
                    let mut cursor = Cursor::new(Vec::new());
                    let spec = hound::WavSpec {
                        channels: 1,
                        sample_rate: audio.sample_rate,
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    };
                    let write_res = (|| -> Result<(), hound::Error> {
                        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
                        for &sample in &audio.samples {
                            let clamped = sample.clamp(-1.0, 1.0);
                            let i = (clamped * 32767.0) as i16;
                            writer.write_sample(i)?;
                        }
                        writer.finalize()?;
                        Ok(())
                    })();

                    if write_res.is_ok() {
                        Some(cursor.into_inner())
                    } else {
                        None
                    }
                }
                Err(e) => {
                    eprintln!("Kokoro TTS synthesis error: {e}");
                    None
                }
            }
        } else {
            // Synthetic test output: minimal valid WAV containing sentence marker
            let mut cursor = Cursor::new(Vec::new());
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let write_res = (|| -> Result<(), hound::Error> {
                let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
                for _ in 0..160 {
                    writer.write_sample(0i16)?;
                }
                writer.finalize()?;
                Ok(())
            })();

            if write_res.is_ok() {
                Some(cursor.into_inner())
            } else {
                None
            }
        }
    }
}

/// Real hardware audio playback engine using CPAL with barge-in and echo suppression.
pub struct CpalTtsPlayer {
    stop_flag: Arc<AtomicBool>,
}

impl CpalTtsPlayer {
    pub fn new() -> Self {
        Self {
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Stop ongoing playback immediately (<500ms barge-in).
    pub fn stop_speaking(&self, pump: Option<&AudioPump>) {
        self.stop_flag.store(true, Ordering::SeqCst);
        // Suppress acoustic echo of the tail for 200 ms
        if let Some(p) = pump {
            p.gate(0.20);
        }
    }

    /// Play WAV audio bytes to the default sound output device.
    ///
    /// Respects `stop_flag`: cancels immediately when user barges in.
    pub fn play_wav(&self, wav_bytes: &[u8], pump: Option<&AudioPump>) -> Result<(), String> {
        self.stop_flag.store(false, Ordering::SeqCst);

        let cursor = Cursor::new(wav_bytes);
        let mut reader = hound::WavReader::new(cursor)
            .map_err(|e| format!("Failed to read WAV audio bytes: {e}"))?;

        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / 32768.0)
                .collect(),
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .filter_map(Result::ok)
                .collect(),
        };

        if samples.is_empty() {
            return Ok(());
        }

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "No default audio output device found".to_string())?;

        let config = device
            .default_output_config()
            .map_err(|e| format!("Failed to get default output config: {e}"))?;

        let stop_clone = Arc::clone(&self.stop_flag);
        let mut sample_idx = 0;
        let total_samples = samples.len();
        let channels = config.channels() as usize;

        let err_fn = |err| eprintln!("Audio output stream error: {err}");

        let stream = device
            .build_output_stream(
                &config.into(),
                move |data: &mut [f32], _: &_| {
                    for frame in data.chunks_mut(channels) {
                        if stop_clone.load(Ordering::Relaxed) || sample_idx >= total_samples {
                            for s in frame.iter_mut() {
                                *s = 0.0;
                            }
                        } else {
                            let sample = samples[sample_idx];
                            sample_idx += 1;
                            for s in frame.iter_mut() {
                                *s = sample;
                            }
                        }
                    }
                },
                err_fn,
                None,
            )
            .map_err(|e| format!("Failed to build audio output stream: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("Failed to play audio output stream: {e}"))?;

        // Wait until all samples are played or barge-in is triggered
        let sample_rate = spec.sample_rate as f64;
        let duration = Duration::from_secs_f64(total_samples as f64 / sample_rate);
        let start = std::time::Instant::now();

        while start.elapsed() < duration {
            if self.stop_flag.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }

        // Gate mic for 200 ms to suppress echo of the spoken TTS tail
        if let Some(p) = pump {
            p.gate(0.20);
        }

        Ok(())
    }
}

impl Default for CpalTtsPlayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_dir_reports_models_md() {
        let res = SherpaKokoroTts::new("nonexistent-kokoro-dir", 1.0, 2);
        assert!(res.is_err());
        let err = res.err().unwrap();
        assert!(err.contains("docs/MODELS.md"));
        assert!(err.contains("nonexistent-kokoro-dir"));
    }

    #[test]
    fn mock_tts_synthesizes_valid_wav() {
        let mut tts = SherpaKokoroTts::new_mock(1.0);
        let wav = tts.synthesize("Hello world", "af_bella");
        assert!(wav.is_some());
        let bytes = wav.unwrap();
        // Check RIFF and WAVE header
        assert!(bytes.starts_with(b"RIFF"));
        assert_eq!(&bytes[8..12], b"WAVE");
    }

    #[test]
    fn barge_in_stop_sets_flag_and_gates_pump() {
        let player = CpalTtsPlayer::new();
        let chunk = vec![0u8; 100];
        let src = Box::new(crate::ScriptedAudioSource::new(vec![chunk]));
        let pump = AudioPump::new(src);

        player.stop_speaking(Some(&pump));
        assert!(player.stop_flag.load(Ordering::SeqCst));
        assert!(pump.is_gated());
    }
}
