#!/usr/bin/env python3
"""Development launcher for Grace.

Runs the full Grace pipeline for testing:
1. Start microphone
2. Listen for wake word
3. Start Whisper streaming
4. Process command
5. Play response (TTS later)

Usage: python -m grace
"""

import sys
import os
import time
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.config import Config
from grace.audio.capture import AudioCapture
from grace.audio.wake_word import WakeWordDetector
from grace.vad.detector import VadDetector


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("grace")


def main() -> None:
    log.info("Starting Grace v0.0.1 (Step 1: Wake Word + Audio Capture)")

    config = Config()

    # Initialize audio capture
    capture = AudioCapture(
        sample_rate=config.mic_sample_rate,
        chunk_size=config.mic_chunk,
        channels=config.mic_channels,
        width=config.mic_width,
        device_index=config.mic_device_index,
    )

    # List available devices
    devices = capture.list_devices()
    log.info(f"Found {len(devices)} audio device(s):")
    for dev in devices:
        log.info(f"  [{dev['index']}] {dev['name']} (channels: {dev['max_input_channels']})")

    # Initialize wake word detector
    # Note: Vosk model needs to be downloaded first
    vosk_model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "vosk-model-small-en-us-0.15")
    if not os.path.exists(vosk_model_dir):
        log.warning(f"Vosk model not found at {vosk_model_dir}")
        log.warning("Run: python scripts/download_models.py")
        log.info("Continuing without wake word - press Ctrl+C to exit")

    # Start audio capture
    log.info("Starting microphone...")
    capture.start()
    log.info("Microphone active. Listening...")

    # Main loop
    try:
        while True:
            chunk = capture.get_chunk()
            rms = capture.get_rms(chunk)

            if rms > 0.1:
                print(f"\r  Signal: {rms:.3f}", end="", flush=True)
            else:
                print("\rListening...", end="", flush=True)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        capture.stop()
        capture.close()
        log.info("Grace stopped.")


if __name__ == "__main__":
    main()
