"""Setup script to download and install custom llama-cpp-turboquant binaries for Grace.

Source Release: https://github.com/TheTom/llama-cpp-turboquant/releases/tag/tqp-v0.3.0
Asset: turboquant-plus-tqp-v0.3.0-windows-x64-cuda12.4.zip

This binary is used twice: as the local planner LLM, and - regardless of the
cloud/local setting - as the vision model that decides where Grace clicks on
screen. GitHub lets a release asset be silently swapped under the same tag,
so a hash check here is not optional (R4): the download is verified against
`LLAMA_CPP_SHA256` before extraction, and extraction never runs on a zip that
fails it.
"""

import os
import sys
import zipfile
import shutil

from verified_download import HashMismatch, download_and_verify

LLAMA_CPP_URL = (
    "https://github.com/TheTom/llama-cpp-turboquant/releases/download/"
    "tqp-v0.3.0/turboquant-plus-tqp-v0.3.0-windows-x64-cuda12.4.zip"
)

# SHA-256 of turboquant-plus-tqp-v0.3.0-windows-x64-cuda12.4.zip.
#
# UNVERIFIED - this is a placeholder, not a real hash. No copy of this exact
# zip exists anywhere in this repo or dev environment to compute one from,
# and this script must never download anything just to derive a hash (that
# would defeat the point of pinning one). Fill it in yourself, once, from a
# copy of the zip you already trust:
#
#   certutil -hashfile turboquant-plus-tqp-v0.3.0-windows-x64-cuda12.4.zip SHA256
#
# (or `sha256sum` / Python's `hashlib.sha256` - see verified_download.sha256_of)
#
# Until the real 64-character hex digest replaces the line below, this script
# fails closed: it downloads the zip, then refuses to extract it, rather than
# silently trusting whatever GitHub currently serves under that tag.
LLAMA_CPP_SHA256 = "0000000000000000000000000000000000000000000000000000000000000000"  # noqa: E501  <-- REPLACE with the real hash

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_CPP_DIR = os.path.join(PROJECT_ROOT, "llama cpp")
EXE_PATH = os.path.join(LLAMA_CPP_DIR, "llama-server.exe")


def download_and_setup():
    print("=" * 65)
    print("  Project Grace - Custom Llama CPP (TurboQuant v0.3.0) Setup")
    print("=" * 65)

    if os.path.isfile(EXE_PATH):
        print(f"[OK] llama-server.exe already exists at: {LLAMA_CPP_DIR}")
        return True

    os.makedirs(LLAMA_CPP_DIR, exist_ok=True)
    zip_path = os.path.join(PROJECT_ROOT, "llama_cpp_turboquant_temp.zip")

    print(f"[-->] Downloading TurboQuant llama.cpp build from:\n      {LLAMA_CPP_URL}")

    try:
        download_and_verify(LLAMA_CPP_URL, zip_path, LLAMA_CPP_SHA256)
    except HashMismatch as e:
        print(f"[X] Refusing to install: {e}")
        print(
            "    If this is a legitimate new release, update LLAMA_CPP_SHA256 "
            "in scripts/setup_llama_cpp.py after verifying the new zip yourself."
        )
        return False
    except Exception as e:
        print(f"[X] Failed to download llama.cpp: {e}")
        return False

    try:
        print(f"[-->] Extracting binaries to: {LLAMA_CPP_DIR}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(LLAMA_CPP_DIR)

        # Check if contents were extracted inside a subfolder
        extracted_items = os.listdir(LLAMA_CPP_DIR)
        if len(extracted_items) == 1:
            single_item = os.path.join(LLAMA_CPP_DIR, extracted_items[0])
            if os.path.isdir(single_item) and os.path.isfile(
                os.path.join(single_item, "llama-server.exe")
            ):
                # Move all contents from subfolder to LLAMA_CPP_DIR
                for filename in os.listdir(single_item):
                    shutil.move(
                        os.path.join(single_item, filename),
                        os.path.join(LLAMA_CPP_DIR, filename),
                    )
                os.rmdir(single_item)

        if os.path.exists(zip_path):
            os.remove(zip_path)

        if os.path.isfile(EXE_PATH):
            print(f"[SUCCESS] Llama CPP TurboQuant successfully installed to {LLAMA_CPP_DIR}")
            return True
        else:
            print("[X] Extraction finished but llama-server.exe was not found in destination.")
            return False

    except Exception as e:
        print(f"[X] Failed to setup llama.cpp: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False


if __name__ == "__main__":
    success = download_and_setup()
    if not success:
        sys.exit(1)
