"""TORGO speech dataset loader and preprocessor (PLAN.md §8, §9, Task 09).

Parses TORGO dataset directory structure, distinguishes dysarthric vs control
speakers and severity levels, filters out image-description and non-speech prompts,
and provides on-demand resampling to 16 kHz mono cached audio.

NOTE ON DATASET ACCESS:
TORGO is released by the University of Toronto for non-commercial research purposes.
The user must download it directly from the official University of Toronto page:
http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
This module NEVER downloads data automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import wave

import numpy as np
import scipy.signal

from bench.normalizer import normalize_text

logger = logging.getLogger(__name__)

# Official TORGO severity classifications based on clinical assessment
# (Rudzicz et al., 2012; Kim et al., 2018)
TORGO_SPEAKER_METADATA: Dict[str, Dict[str, Any]] = {
    # Dysarthric speakers (Cerebral Palsy or ALS)
    "F01": {"gender": "F", "is_dysarthric": True, "severity": "Severe"},
    "F03": {"gender": "F", "is_dysarthric": True, "severity": "Moderate"},
    "F04": {"gender": "F", "is_dysarthric": True, "severity": "Moderate"},
    "M01": {"gender": "M", "is_dysarthric": True, "severity": "Mild"},
    "M02": {"gender": "M", "is_dysarthric": True, "severity": "Severe"},
    "M03": {"gender": "M", "is_dysarthric": True, "severity": "Severe"},
    "M04": {"gender": "M", "is_dysarthric": True, "severity": "Moderate"},
    "M05": {"gender": "M", "is_dysarthric": True, "severity": "Mild"},
    # Age-matched control speakers
    "FC01": {"gender": "F", "is_dysarthric": False, "severity": "Control"},
    "FC02": {"gender": "F", "is_dysarthric": False, "severity": "Control"},
    "FC03": {"gender": "F", "is_dysarthric": False, "severity": "Control"},
    "MC01": {"gender": "M", "is_dysarthric": False, "severity": "Control"},
    "MC02": {"gender": "M", "is_dysarthric": False, "severity": "Control"},
    "MC03": {"gender": "M", "is_dysarthric": False, "severity": "Control"},
    "MC04": {"gender": "M", "is_dysarthric": False, "severity": "Control"},
}

# Patterns indicating image description tasks or non-speech prompts to exclude
_IMAGE_DESCRIPTION_PATTERN = re.compile(
    r"\.jpg|\.png|\[image|describe the picture|picture|image:|file:",
    re.IGNORECASE,
)
_NON_SPEECH_TOKENS = {
    "[cough]",
    "[laughter]",
    "[sigh]",
    "[applause]",
    "[snort]",
    "[throat-clearing]",
    "[gasp]",
    "[pant]",
    "[whisper]",
    "xxx",
    "*",
    "n/a",
}


def is_image_description_prompt(prompt_text: str) -> bool:
    """Check if prompt is an image description task (where prompt != transcript)."""
    return bool(_IMAGE_DESCRIPTION_PATTERN.search(prompt_text))


def is_non_speech_prompt(prompt_text: str) -> bool:
    """Check if prompt is non-speech, acoustic artifact, or calibration code."""
    cleaned = prompt_text.strip().lower()
    if not cleaned:
        return True
    if cleaned in _NON_SPEECH_TOKENS:
        return True
    # If the prompt consists entirely of bracketed tags: [cough], [laughter], etc.
    bracket_content = re.sub(r"\[.*?\]|\<.*?\>", "", cleaned).strip()
    return len(bracket_content) == 0


def resample_wav_to_16k_mono(src_path: Union[str, Path], dst_path: Union[str, Path]) -> Path:
    """Convert a WAV file to 16 kHz 16-bit mono PCM, saving to dst_path."""
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(src), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    # Decode bytes to numpy int16 array
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    elif sampwidth == 1:
        dtype = np.uint8
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    audio_data = np.frombuffer(raw_bytes, dtype=dtype)
    if sampwidth == 1:
        audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
    else:
        audio_data = audio_data.astype(np.float32) / float(np.iinfo(dtype).max)

    # Convert to mono if multi-channel
    if n_channels > 1:
        audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)

    # Resample to 16,000 Hz if needed
    target_sr = 16000
    if framerate != target_sr:
        num_target_samples = int(round(len(audio_data) * target_sr / float(framerate)))
        audio_data = scipy.signal.resample(audio_data, num_target_samples)

    # Clip and convert to int16 PCM
    audio_data = np.clip(audio_data, -1.0, 1.0)
    int16_pcm = (audio_data * 32767.0).astype(np.int16)

    # Write destination WAV
    with wave.open(str(dst), "wb") as out_wf:
        out_wf.setnchannels(1)
        out_wf.setsampwidth(2)
        out_wf.setframerate(target_sr)
        out_wf.writeframes(int16_pcm.tobytes())

    return dst


@dataclass
class TorgoUtterance:
    """Single evaluated speech utterance in TORGO."""
    utt_id: str
    speaker_id: str
    session_id: str
    prompt_id: str
    is_dysarthric: bool
    severity: str
    prompt_raw: str
    prompt_normalized: str
    wav_path: Path
    mic_type: str

    def get_wav_16k(self, cache_dir: Optional[Union[str, Path]] = None) -> Path:
        """Return path to 16 kHz mono WAV, converting into cache_dir if necessary."""
        if cache_dir is None:
            cache_dir = self.wav_path.parent / ".cache_16k"
        else:
            cache_dir = Path(cache_dir)

        dst = cache_dir / self.speaker_id / self.session_id / f"{self.prompt_id}_16k.wav"

        # Check if cached converted file is up-to-date
        if dst.is_file() and dst.stat().st_size > 0:
            if dst.stat().st_mtime >= self.wav_path.stat().st_mtime:
                return dst

        return resample_wav_to_16k_mono(self.wav_path, dst)


@dataclass
class TorgoDataset:
    """Parsed TORGO dataset collection."""
    root_path: Path
    utterances: List[TorgoUtterance] = field(default_factory=list)
    exclusions: Dict[str, int] = field(default_factory=lambda: {
        "non_speech": 0,
        "image_description": 0,
        "empty_prompt": 0,
        "missing_wav": 0,
    })

    def get_dysarthric(self) -> List[TorgoUtterance]:
        return [u for u in self.utterances if u.is_dysarthric]

    def get_controls(self) -> List[TorgoUtterance]:
        return [u for u in self.utterances if not u.is_dysarthric]

    def get_by_speaker(self, speaker_id: str) -> List[TorgoUtterance]:
        s_id = speaker_id.upper()
        return [u for u in self.utterances if u.speaker_id.upper() == s_id]

    def get_by_severity(self, severity: str) -> List[TorgoUtterance]:
        sev = severity.capitalize()
        return [u for u in self.utterances if u.severity == sev]

    def summary(self) -> Dict[str, Any]:
        speakers = sorted(list({u.speaker_id for u in self.utterances}))
        dys_speakers = [s for s in speakers if s in TORGO_SPEAKER_METADATA and TORGO_SPEAKER_METADATA[s]["is_dysarthric"]]
        ctrl_speakers = [s for s in speakers if s not in dys_speakers]

        return {
            "root_path": str(self.root_path),
            "total_utterances": len(self.utterances),
            "dysarthric_utterances": len(self.get_dysarthric()),
            "control_utterances": len(self.get_controls()),
            "total_speakers": len(speakers),
            "dysarthric_speakers": dys_speakers,
            "control_speakers": ctrl_speakers,
            "exclusions": dict(self.exclusions),
        }


def load_torgo_dataset(
    root_dir: Union[str, Path],
    mic_preference: str = "wav_headMic",
    skip_exclusions: bool = True,
) -> TorgoDataset:
    """Load and index the TORGO speech dataset.

    Args:
        root_dir: Path to extracted TORGO root directory.
        mic_preference: Primary mic directory to use ('wav_headMic' or 'wav_arrayMic').
        skip_exclusions: If True, skips image description and non-speech prompts.

    Returns:
        TorgoDataset object containing indexed utterances and exclusion stats.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"TORGO root directory not found: {root}")

    dataset = TorgoDataset(root_path=root)

    # Locate speaker directories (e.g. F01, M01, FC01, etc.)
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name.upper()

        meta = TORGO_SPEAKER_METADATA.get(speaker_id)
        if meta is None:
            # Fallback heuristic: FC/MC are controls, F/M followed by digits are dysarthric
            is_dys = not (speaker_id.startswith("FC") or speaker_id.startswith("MC"))
            meta = {
                "gender": speaker_id[0] if not speaker_id.startswith(("FC", "MC")) else speaker_id[1],
                "is_dysarthric": is_dys,
                "severity": "Unknown" if is_dys else "Control",
            }

        # Locate Session directories (e.g. Session1, Session2)
        session_dirs = [d for d in speaker_dir.iterdir() if d.is_dir() and "session" in d.name.lower()]
        if not session_dirs:
            # Some archives place prompts directly in speaker dir
            session_dirs = [speaker_dir]

        for sess_dir in session_dirs:
            session_id = sess_dir.name

            # Look for prompts directory
            prompts_dir = sess_dir / "prompts"
            if not prompts_dir.is_dir():
                prompts_dir = sess_dir / "Prompts"
            if not prompts_dir.is_dir():
                continue

            # Look for WAV directory (headMic preferred, fallback to arrayMic)
            wav_dir = sess_dir / mic_preference
            if not wav_dir.is_dir():
                alt_mic = "wav_arrayMic" if mic_preference == "wav_headMic" else "wav_headMic"
                wav_dir = sess_dir / alt_mic
            if not wav_dir.is_dir():
                # Check case variations
                candidates = [d for d in sess_dir.iterdir() if d.is_dir() and "wav" in d.name.lower()]
                if candidates:
                    wav_dir = candidates[0]
                else:
                    continue

            mic_type = "headMic" if "head" in wav_dir.name.lower() else "arrayMic"

            # Iterate over prompt txt files
            for prompt_file in sorted(prompts_dir.glob("*.txt")):
                prompt_id = prompt_file.stem
                wav_file = wav_dir / f"{prompt_id}.wav"
                if not wav_file.is_file():
                    dataset.exclusions["missing_wav"] += 1
                    continue

                try:
                    raw_text = prompt_file.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    dataset.exclusions["empty_prompt"] += 1
                    continue

                if not raw_text:
                    dataset.exclusions["empty_prompt"] += 1
                    continue

                if skip_exclusions:
                    if is_image_description_prompt(raw_text):
                        dataset.exclusions["image_description"] += 1
                        continue
                    if is_non_speech_prompt(raw_text):
                        dataset.exclusions["non_speech"] += 1
                        continue

                norm_text = normalize_text(raw_text)
                if not norm_text:
                    dataset.exclusions["empty_prompt"] += 1
                    continue

                utt_id = f"{speaker_id}_{session_id}_{prompt_id}"
                utterance = TorgoUtterance(
                    utt_id=utt_id,
                    speaker_id=speaker_id,
                    session_id=session_id,
                    prompt_id=prompt_id,
                    is_dysarthric=meta["is_dysarthric"],
                    severity=meta["severity"],
                    prompt_raw=raw_text,
                    prompt_normalized=norm_text,
                    wav_path=wav_file,
                    mic_type=mic_type,
                )
                dataset.utterances.append(utterance)

    logger.info(
        "Loaded TORGO dataset from %s: %d valid utterances (%d dysarthric, %d control). Exclusions: %s",
        root, len(dataset.utterances), len(dataset.get_dysarthric()), len(dataset.get_controls()),
        dataset.exclusions,
    )
    return dataset
