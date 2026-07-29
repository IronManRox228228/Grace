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


def check_kokoro_model():
    """Check if Kokoro model is available."""
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "models", "kokoro")

    # Check HuggingFace cache first
    hf_cache = os.path.expanduser(
        r".cache\huggingface\hub\models--hexgrad--Kokoro-82M\snapshots"
    )

    if os.path.exists(hf_cache):
        print(f"  Kokoro model found in HuggingFace cache: {hf_cache}")
        return True

    if os.path.exists(path):
        print(f"  Kokoro model found at {path}")
        return True

    print(f"  Kokoro model not found. Checking HuggingFace cache...")
    print(f"  Expected at: {hf_cache}")
    print("  If not cached, run: huggingface-cli download hexgrad/Kokoro-82M")
    return False


def main():
    print("=" * 60)
    print("Grace - Model Download")
    print("=" * 60)

    print("\n1. Checking Kokoro model...")
    check_kokoro_model()

    print("\n2. Downloading Vosk model...")
    download_vosk_model()

    print("\n3. Downloading Whisper model...")
    download_whisper_model()

    print("\nDone!")


if __name__ == "__main__":
    main()
