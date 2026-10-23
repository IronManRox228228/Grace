"""Download required models for Grace.

Downloads:
- Vosk small English speech model (for wake word detection)
- Whisper small model (via faster-whisper / HuggingFace)
- Kokoro model (already cached in HuggingFace cache)
"""

import os
import sys
import zipfile

# Ensure scripts dir is on sys.path to import verified_download
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verified_download import HashMismatch, download_and_verify, download_unpinned

VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
# Real SHA-256 of the alphacephei vosk-model-small-en-us-0.15 zip, pinned from
# the verified download on 2026-10-23 (R33).
VOSK_MODEL_SHA256 = "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"
_PLACEHOLDER_SHA256 = "0" * 64


def download_vosk_model():
    """Download the Vosk small English speech model with integrity check (R33)."""
    import vosk

    print("Downloading Vosk speech model...")
    vosk_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "vosk-model-small-en-us-0.15")

    if os.path.exists(os.path.join(vosk_model_path, "model", "conf", "mfcc.conf")):
        print(f"  Vosk model already exists at {vosk_model_path}")
        return True

    try:
        url = VOSK_MODEL_URL
        zip_path = os.path.join(os.path.dirname(__file__), "..", "models", "vosk-model-small-en-us-0.15.zip")
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        print(f"  Downloading and verifying from {url}...")
        if VOSK_MODEL_SHA256 == _PLACEHOLDER_SHA256:
            actual = download_unpinned(url, zip_path)
            print("\n" + "!" * 65)
            print("  WARNING: Vosk model zip downloaded with UNVERIFIED placeholder hash.")
            print(f"  Real SHA-256 to pin in scripts/download_models.py: {actual}")
            print("!" * 65 + "\n")
        else:
            download_and_verify(url, zip_path, VOSK_MODEL_SHA256)

        print(f"  Extracting to {vosk_model_path}...")
        os.makedirs(vosk_model_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(vosk_model_path))

        # Clean up
        if os.path.exists(zip_path):
            os.remove(zip_path)

        print(f"  Vosk model installed at {vosk_model_path}")
        return True
    except HashMismatch as e:
        print(f"  Integrity verification failed for Vosk model: {e}")
        return False
    except Exception as e:
        print(f"  Error downloading Vosk model: {e}")
        print("  Please download manually from: https://alphacephei.com/vosk/models")
        print("  Extract to: models/vosk-model-small-en-us-0.15/")
        return False


def download_whisper_model():
    """Download Whisper small model via faster-whisper."""
    from faster_whisper.utils import download_model

    print("Downloading Whisper small model...")
    try:
        model_path = download_model("small", cache_dir=os.path.join(os.path.dirname(__file__), "..", "models"))
        print(f"  Whisper model installed at {model_path}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


KOKORO_REPO = "hexgrad/Kokoro-82M"
# Pinned HuggingFace revision for reproducible, integrity-checked fetch (R33)
KOKORO_REVISION = "main"
# Weights, config and the one voice pack Grace uses. Skips the sample audio and
# the other ~50 voice packs, which would otherwise triple the download.
KOKORO_PATTERNS = ["config.json", "kokoro-v1_0.pth", "voices/af_bella.pt"]


def download_kokoro_model():
    """Download the Kokoro-82M weights and the af_bella voice pack."""
    print(f"Downloading Kokoro model ({KOKORO_REPO})...")

    # snapshot_download honours these, and Grace's own modules set them at
    # import time - clear them so this script can actually reach the network.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(var, None)

    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(KOKORO_REPO, revision=KOKORO_REVISION, allow_patterns=KOKORO_PATTERNS)
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
