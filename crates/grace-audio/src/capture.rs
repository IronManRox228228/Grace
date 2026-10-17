use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::{AudioSource, PcmChunk};

/// A mic audio pump that reads chunks from an `AudioSource` on a background thread.
///
/// Supports:
/// - Echo suppression gating: `gate(duration_secs)` discards audio while the chime or
///   TTS is playing so Grace doesn't hear herself.
/// - Buffer draining: `drain()` drops any stale chunks from previous turns.
/// - Timeout reads: `get(timeout)` blocks for the next chunk or times out.
pub struct AudioPump {
    rx: Mutex<mpsc::Receiver<PcmChunk>>,
    gate_until: Arc<Mutex<Instant>>,
    running: Arc<AtomicBool>,
}

impl AudioPump {
    /// Creates a new `AudioPump` driving the given `AudioSource` on a worker thread.
    pub fn new(mut source: Box<dyn AudioSource>) -> Self {
        let (tx, rx) = mpsc::sync_channel::<PcmChunk>(64);
        let gate_until = Arc::new(Mutex::new(Instant::now()));
        let running = Arc::new(AtomicBool::new(true));

        let gate_clone = Arc::clone(&gate_until);
        let running_clone = Arc::clone(&running);

        std::thread::Builder::new()
            .name("grace-audio-pump".to_string())
            .spawn(move || {
                while running_clone.load(Ordering::Relaxed) {
                    if let Some(chunk) = source.next_chunk() {
                        let is_gated = {
                            let g = gate_clone.lock().unwrap();
                            Instant::now() < *g
                        };
                        if !is_gated {
                            let _ = tx.send(chunk);
                        }
                    } else {
                        // Stream ended or waiting
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
            })
            .expect("spawn audio pump thread");

        Self {
            rx: Mutex::new(rx),
            gate_until,
            running,
        }
    }

    /// Gate incoming audio for `duration_seconds` to suppress acoustic echo.
    pub fn gate(&self, duration_seconds: f64) {
        let until = Instant::now() + Duration::from_secs_f64(duration_seconds);
        let mut g = self.gate_until.lock().unwrap();
        if until > *g {
            *g = until;
        }
    }

    /// True if mic input is currently gated (suppressed).
    pub fn is_gated(&self) -> bool {
        let g = self.gate_until.lock().unwrap();
        Instant::now() < *g
    }

    /// Drain all queued chunks from the channel.
    pub fn drain(&self) {
        let rx = self.rx.lock().unwrap();
        while rx.try_recv().is_ok() {}
    }

    /// Read the next audio chunk, waiting up to `timeout`.
    pub fn get(&self, timeout: Duration) -> Option<PcmChunk> {
        let rx = self.rx.lock().unwrap();
        rx.recv_timeout(timeout).ok()
    }
}

impl Drop for AudioPump {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

/// Real hardware audio capture using CPAL (WASAPI on Windows).
///
/// Captures mic audio, resamples to 16,000 Hz 1-channel 16-bit PCM if needed.
pub struct CpalAudioCapture {
    _stream: Option<cpal::Stream>,
    rx: mpsc::Receiver<PcmChunk>,
}

impl CpalAudioCapture {
    /// Connects to the system default input device and starts recording.
    pub fn new_default() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| "No default audio input device found".to_string())?;

        Self::from_device(&device)
    }

    /// Connects to a specific CPAL audio input device.
    pub fn from_device(device: &cpal::Device) -> Result<Self, String> {
        let config = device
            .default_input_config()
            .map_err(|e| format!("Failed to get default input config: {e}"))?;

        let (tx, rx) = mpsc::sync_channel::<PcmChunk>(128);

        let err_fn = |err| eprintln!("Audio input stream error: {err}");

        let sample_rate = config.sample_rate();
        let channels = config.channels() as usize;

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => {
                let tx = tx.clone();
                device.build_input_stream(
                    &config.into(),
                    move |data: &[f32], _: &_| {
                        let pcm16 = resample_and_convert_f32_to_pcm16(data, sample_rate, channels);
                        let _ = tx.try_send(pcm16);
                    },
                    err_fn,
                    None,
                )
            }
            cpal::SampleFormat::I16 => {
                let tx = tx.clone();
                device.build_input_stream(
                    &config.into(),
                    move |data: &[i16], _: &_| {
                        let pcm16 = resample_and_convert_i16_to_pcm16(data, sample_rate, channels);
                        let _ = tx.try_send(pcm16);
                    },
                    err_fn,
                    None,
                )
            }
            sample_format => return Err(format!("Unsupported sample format: {sample_format:?}")),
        }
        .map_err(|e| format!("Failed to build input stream: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("Failed to start input stream: {e}"))?;

        Ok(Self {
            _stream: Some(stream),
            rx,
        })
    }
}

impl AudioSource for CpalAudioCapture {
    fn next_chunk(&mut self) -> Option<PcmChunk> {
        self.rx.recv().ok()
    }
}

/// Downmixes multi-channel audio to mono and resamples to 16,000 Hz 16-bit PCM.
pub fn resample_and_convert_f32_to_pcm16(
    data: &[f32],
    source_rate: u32,
    channels: usize,
) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    // 1. Downmix to mono
    let mono: Vec<f32> = if channels > 1 {
        data.chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        data.to_vec()
    };

    // 2. Resample to 16000 Hz if needed
    let resampled = if source_rate != 16000 && source_rate > 0 {
        linear_resample_f32(&mono, source_rate, 16000)
    } else {
        mono
    };

    // 3. Convert f32 [-1.0, 1.0] to 16-bit little-endian PCM bytes
    let mut out = Vec::with_capacity(resampled.len() * 2);
    for sample in resampled {
        let clamped = sample.clamp(-1.0, 1.0);
        let i = (clamped * 32767.0) as i16;
        out.extend_from_slice(&i.to_le_bytes());
    }
    out
}

/// Downmixes multi-channel i16 audio to mono and resamples to 16,000 Hz 16-bit PCM.
pub fn resample_and_convert_i16_to_pcm16(
    data: &[i16],
    source_rate: u32,
    channels: usize,
) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    // 1. Downmix to mono
    let mono: Vec<f32> = if channels > 1 {
        data.chunks(channels)
            .map(|chunk| {
                let sum: i32 = chunk.iter().map(|&s| s as i32).sum();
                (sum / channels as i32) as f32 / 32768.0
            })
            .collect()
    } else {
        data.iter().map(|&s| s as f32 / 32768.0).collect()
    };

    // 2. Resample to 16000 Hz if needed
    let resampled = if source_rate != 16000 && source_rate > 0 {
        linear_resample_f32(&mono, source_rate, 16000)
    } else {
        mono
    };

    let mut out = Vec::with_capacity(resampled.len() * 2);
    for sample in resampled {
        let clamped = sample.clamp(-1.0, 1.0);
        let i = (clamped * 32767.0) as i16;
        out.extend_from_slice(&i.to_le_bytes());
    }
    out
}

fn linear_resample_f32(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() || src_rate == dst_rate {
        return input.to_vec();
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let target_len = (input.len() as f64 * ratio).round() as usize;
    let mut output = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let src_idx = i as f64 / ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(input.len() - 1);
        let frac = (src_idx - idx0 as f64) as f32;
        let val = input[idx0] * (1.0 - frac) + input[idx1] * frac;
        output.push(val);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScriptedAudioSource;

    #[test]
    fn pump_echo_gating_discards_audio() {
        let chunk1 = vec![1, 2, 3, 4];
        let chunk2 = vec![5, 6, 7, 8];
        let src = Box::new(ScriptedAudioSource::new(vec![chunk1.clone(), chunk2.clone()]));
        let pump = AudioPump::new(src);

        // Gate for 500 ms immediately
        pump.gate(0.5);
        assert!(pump.is_gated());

        // Wait a tiny moment for worker thread to process while gated
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(pump.get(Duration::from_millis(50)), None);
    }

    #[test]
    fn pump_drain_empties_buffer() {
        let chunk1 = vec![10; 32];
        let chunk2 = vec![20; 32];
        let src = Box::new(ScriptedAudioSource::new(vec![chunk1, chunk2]));
        let pump = AudioPump::new(src);

        // Allow worker to push
        std::thread::sleep(Duration::from_millis(50));
        pump.drain();

        // Buffer should be empty
        assert_eq!(pump.get(Duration::from_millis(10)), None);
    }

    #[test]
    fn resampler_converts_rates_correctly() {
        let input = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
        let pcm = resample_and_convert_f32_to_pcm16(&input, 32000, 1);
        // Half as many samples at 16000 Hz, 2 bytes per sample
        assert_eq!(pcm.len(), 8); // 4 samples * 2 bytes
    }
}
