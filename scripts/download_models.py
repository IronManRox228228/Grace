"""Download required models for Grace.

Downloads:
- Vosk small English speech model (for wake word detection)
- Whisper small model (via faster-whisper / HuggingFace)
- Kokoro model (already cached in HuggingFace cache)
"""

import os
import sys
import subprocess


def download_vosk_model():
    """Download the Vosk small English speech model."""
    import vosk

    print("Downloading Vosk speech model...")
    vosk_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "vosk-model-small-en-us-0.15")

    if os.path.exists(os.path.join(vosk_model_path, "model", "conf", "mfcc.conf")):
        print(f"  Vosk model already exists at {vosk_model_path}")
        return

    # Download using the vosk package's built-in download
    try:
        import requests
        import zipfile
        import io

        url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        zip_path = os.path.join(os.path.dirname(__file__), "..", "models", "vosk-model-small-en-us-0.15.zip")

        os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        print(f"  Downloading from {url}...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

        print(f"\n  Extracting to {vosk_model_path}...")
        os.makedirs(vosk_model_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(vosk_model_path))

        # Rename if needed (zip may contain vosk-model-small-en-us-0.15/ directly)
        extracted_dir = os.path.join(os.path.dirname(vosk_model_path), "vosk-model-small-en-us-0.15")
        if os.path.exists(extracted_dir) and os.path.abspath(extracted_dir) != os.path.abspath(vosk_model_path):
            # Already extracted to the right place, nothing to do
            pass

        # Clean up
        if os.path.exists(zip_path):
            os.remove(zip_path)

        print(f"  Vosk model installed at {vosk_model_path}")
    except Exception as e:
        print(f"  Error downloading Vosk model: {e}")
        print("  Please download manually from: https://alphacephei.com/vosk/models")
        print("  Extract to: models/vosk-model-small-en-us-0.15/")


def download_whisper_model():
    """Download Whisper small model via faster-whisper."""
    from faster_whisper.utils import download_model

    print("Downloading Whisper small model...")
    try:
        model_path = download_model("small", cache_dir=os.path.join(os.path.dirname(__file__), "..", "models"))
        print(f"  Whisper model installed at {model_path}")
    except Exception as e:
        print(f"  Error: {e}")


KOKORO_REPO = "hexgrad/Kokoro-82M"
# Weights, config and the one voice pack Grace uses. Skips the sample audio and
# the other ~50 voice packs, which would otherwise triple the download.
KOKORO_PATTERNS = ["config.json", "kokoro-v1_0.pth", "voices/af_bella.pt"]


def download_kokoro_model():
    """Download the Kokoro-82M weights and the af_bella voice pack.

    Grace forces HF_HUB_OFFLINE=1 at runtime, so the model has to be fetched
    ahead of time. This used to only *check* for the model, and the check was
    broken: os.path.expanduser() on a path with no leading '~' returns it
    unchanged, so it tested a relative path that could never exist.
    """
    print(f"Downloading Kokoro model ({KOKORO_REPO})...")

    # snapshot_download honours these, and Grace's own modules set them at
    # import time - clear them so this script can actually reach the network.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(var, None)

    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(KOKORO_REPO, allow_patterns=KOKORO_PATTERNS)
        print(f"  Kokoro model installed at {path}")

        voices = os.path.join(path, "voices", "af_bella.pt")
        if not os.path.exists(voices):
            print(f"  WARNING: voice pack missing at {voices}")
            return False

        print("  Set these in your .env:")
        print(f"    KOKORO_MODEL_PATH={os.path.join(path, 'kokoro-v1_0.pth')}")
        print(f"    KOKORO_VOICES_PATH={voices}")
        return True
    except Exception as e:
        print(f"  Error downloading Kokoro model: {e}")
        print(f"  Download manually with: huggingface-cli download {KOKORO_REPO}")
        return False


def main():
    print("=" * 60)
    print("Grace - Model Download")
    print("=" * 60)

    print("\n1. Downloading Kokoro model...")
    download_kokoro_model()

    print("\n2. Downloading Vosk model...")
    download_vosk_model()

    print("\n3. Downloading Whisper model...")
    download_whisper_model()

    print("\nDone!")


if __name__ == "__main__":
    main()
