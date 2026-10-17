"""Data preparation pipeline for Grace fleet fine-tunes (PLAN.md §3, §8, Task 12).

Pre-converts:
1. Raw audio recordings into 16 kHz 16-bit mono WAV cached in the SSD working set (.cache/working_set/audio_16k/).
2. Raw screenshots into standard training resolutions (e.g. 512x512, 768x768) cached in (.cache/working_set/vision/).
3. Handles multi-part OS-Atlas Windows split archive verification and preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Add src to sys.path
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logger = logging.getLogger(__name__)

DEFAULT_WORKING_SET_DIR = _ROOT / ".cache" / "working_set"


def convert_wav_to_16k_mono(src_wav: Path, dest_wav: Path) -> Path:
    """Convert an arbitrary audio file to 16 kHz 16-bit mono PCM WAV."""
    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    if dest_wav.is_file() and dest_wav.stat().st_size > 0:
        return dest_wav

    try:
        from scipy.io import wavfile
        import scipy.signal
        import numpy as np

        sr, data = wavfile.read(str(src_wav))
        if data.ndim > 1:
            data = data.mean(axis=1)  # Convert stereo to mono

        if sr != 16000:
            num_samples = int(len(data) * 16000 / float(sr))
            data = scipy.signal.resample(data, num_samples)

        data = np.clip(data, -32768, 32767).astype(np.int16)
        wavfile.write(str(dest_wav), 16000, data)
        return dest_wav
    except Exception as e:
        logger.warning("Failed to resample %s: %s; falling back to copy", src_wav, e)
        shutil.copy2(src_wav, dest_wav)
        return dest_wav


def resize_screenshot_for_training(
    src_image: Path,
    dest_image: Path,
    target_size: Tuple[int, int] = (512, 512),
) -> Path:
    """Pre-convert a screenshot to standard training resolution and PNG format."""
    dest_image.parent.mkdir(parents=True, exist_ok=True)
    if dest_image.is_file() and dest_image.stat().st_size > 0:
        return dest_image

    try:
        from PIL import Image
        with Image.open(src_image) as img:
            resized = img.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
            resized.save(dest_image, format="PNG", optimize=True)
        return dest_image
    except Exception as e:
        logger.warning("Failed to resize %s: %s; falling back to copy", src_image, e)
        shutil.copy2(src_image, dest_image)
        return dest_image


def verify_and_merge_os_atlas_parts(parts_dir: Path, output_archive: Path) -> bool:
    """Check and merge multi-part split downloads for OS-Atlas Windows dataset."""
    if output_archive.is_file() and output_archive.stat().st_size > 0:
        logger.info("OS-Atlas merged archive already exists: %s", output_archive)
        return True

    if not parts_dir.is_dir():
        logger.warning("OS-Atlas parts directory does not exist: %s", parts_dir)
        return False

    # Find parts like os_atlas_windows.tar.gz.part* or .001, .002
    part_files = sorted([
        f for f in parts_dir.iterdir()
        if f.is_file() and ("part" in f.name.lower() or f.suffix in [f".{i:03d}" for i in range(1, 20)])
    ])

    if not part_files:
        logger.warning("No OS-Atlas archive split parts found in %s", parts_dir)
        return False

    logger.info("Found %d split parts for OS-Atlas in %s. Merging to %s...", len(part_files), parts_dir, output_archive)
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    with open(output_archive, "wb") as out_f:
        for part in part_files:
            logger.info("Appending part: %s", part.name)
            with open(part, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f)

    logger.info("Merged %d parts successfully into %s (%d bytes).", len(part_files), output_archive, output_archive.stat().st_size)
    return True


def prep_working_set(
    corpus_dir: Path,
    working_set_dir: Path = DEFAULT_WORKING_SET_DIR,
    target_img_size: Tuple[int, int] = (512, 512),
) -> Dict[str, int]:
    """Scan corpus directory and prepare pre-converted working set assets on the SSD."""
    audio_out = working_set_dir / "audio_16k"
    vision_out = working_set_dir / "vision"

    audio_out.mkdir(parents=True, exist_ok=True)
    vision_out.mkdir(parents=True, exist_ok=True)

    audio_count = 0
    vision_count = 0

    if corpus_dir.is_dir():
        for tape_dir in corpus_dir.iterdir():
            if not tape_dir.is_dir():
                continue

            # Check for audio files
            pcm_dir = tape_dir / "pcm"
            if pcm_dir.is_dir():
                for wav in pcm_dir.glob("*.wav"):
                    out_wav = audio_out / f"{tape_dir.name}_{wav.name}"
                    convert_wav_to_16k_mono(wav, out_wav)
                    audio_count += 1

            # Check for screenshots or snapshot images
            for img in tape_dir.glob("*.png"):
                out_img = vision_out / f"{tape_dir.name}_{img.name}"
                resize_screenshot_for_training(img, out_img, target_size=target_img_size)
                vision_count += 1

    return {
        "audio_files_converted": audio_count,
        "vision_images_converted": vision_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Grace fleet fine-tuning data prep")
    parser.add_argument("--corpus-dir", type=Path, default=_ROOT / "corpus", help="Corpus directory path")
    parser.add_argument("--working-set-dir", type=Path, default=DEFAULT_WORKING_SET_DIR, help="Working set SSD cache directory")
    parser.add_argument("--merge-os-atlas", type=Path, default=None, help="Directory containing OS-Atlas split parts to merge")
    parser.add_argument("--os-atlas-out", type=Path, default=None, help="Output file path for merged OS-Atlas archive")
    args = parser.parse_args()

    if args.merge_os_atlas and args.os_atlas_out:
        verify_and_merge_os_atlas_parts(args.merge_os_atlas, args.os_atlas_out)

    counts = prep_working_set(args.corpus_dir, args.working_set_dir)
    print(f"Data preparation complete: {counts}")


if __name__ == "__main__":
    main()
