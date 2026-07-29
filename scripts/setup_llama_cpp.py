"""Setup script to download and install custom llama-cpp-turboquant binaries for Grace.

Source Release: https://github.com/TheTom/llama-cpp-turboquant/releases/tag/tqp-v0.3.0
Asset: turboquant-plus-tqp-v0.3.0-windows-x64-cuda12.4.zip
"""

import os
import sys
import zipfile
import shutil
import urllib.request

LLAMA_CPP_URL = (
    "https://github.com/TheTom/llama-cpp-turboquant/releases/download/"
    "tqp-v0.3.0/turboquant-plus-tqp-v0.3.0-windows-x64-cuda12.4.zip"
)

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
        req = urllib.request.Request(
            LLAMA_CPP_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(
                        f"\r  Progress: {pct:.1f}% ({mb_down:.1f} MB / {mb_total:.1f} MB)",
                        end="",
                        flush=True,
                    )
            print()

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
