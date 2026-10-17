//! Manual real audio example for Grace (Task 17, Hard Rule 5).
//!
//! **DO NOT RUN IN AUTOMATED TESTS OR SCRIPTS.**
//! This example connects to real physical microphones and speakers via CPAL / WASAPI.
//! Run manually on a developer machine to test physical hardware audio I/O:
//!
//! ```sh
//! cargo run -p grace-audio --example real_audio
//! ```
//!
//! Prerequisites:
//! - Download model files as documented in `docs/MODELS.md`.
//! - Ensure a working microphone and speaker are connected.

use std::time::Duration;
use grace_audio::capture::{AudioPump, CpalAudioCapture};
use grace_audio::vad::VadDetector;

fn main() {
    println!("=== Grace Real Audio Manual Verification ===");
    println!("Testing physical microphone capture via CPAL/WASAPI...");

    let capture = match CpalAudioCapture::new_default() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to open real microphone: {e}");
            return;
        }
    };

    let pump = AudioPump::new(Box::new(capture));
    let mut vad = VadDetector::new_wall_clock(0.008, 1200);

    println!("Microphone opened successfully! Listening for 3 seconds of real audio...");
    println!("Please say something or stay silent...");

    let start = std::time::Instant::now();
    let mut chunks_received = 0;
    let mut speech_detected = false;

    while start.elapsed() < Duration::from_secs(3) {
        if let Some(chunk) = pump.get(Duration::from_millis(100)) {
            chunks_received += 1;
            if vad.process_chunk(&chunk) && vad.has_detected_speech() {
                speech_detected = true;
            }
        }
    }

    println!("Received {chunks_received} PCM audio chunks.");
    println!("Speech detected by VAD: {speech_detected}");

    println!("\nTesting echo suppression gating (gate 500ms)...");
    pump.gate(0.5);
    let gated_chunk = pump.get(Duration::from_millis(50));
    println!("Chunk during gate: {:?}", gated_chunk.is_none());

    println!("\nReal audio verification complete!");
}
