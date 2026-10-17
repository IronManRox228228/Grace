"""Encrypted local at-rest storage for speech enrollment recordings (PLAN §6, Task 14).

Privacy guarantees:
1. All audio recordings and metadata stay strictly local under the app data directory.
2. Protected at rest with Windows Data Protection API (DPAPI, user scope) where available.
3. User retains full control: can list, play, or delete any recording or entire session by voice or UI.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("grace.enrollment.storage")

# Windows DPAPI check
try:
    import win32crypt
    _HAS_WIN32CRYPT = True
except ImportError:
    _HAS_WIN32CRYPT = False


def _make_wav_header(pcm_bytes_len: int, sample_rate: int = 16000, channels: int = 1, bits: int = 16) -> bytes:
    """Generate a standard 44-byte RIFF WAV header for raw PCM bytes."""
    byte_rate = sample_rate * channels * (bits // 8)
    block_align = channels * (bits // 8)
    total_data_len = pcm_bytes_len
    total_file_len = total_data_len + 36

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        total_file_len,
        b"WAVE",
        b"fmt ",
        16,  # PCM subchunk size
        1,   # AudioFormat: 1 = PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        total_data_len,
    )
    return header


class EnrollmentStorage:
    """Manages encrypted at-rest enrollment recordings and session metadata."""

    def __init__(self, base_dir: Optional[str | Path] = None, use_dpapi: Optional[bool] = None):
        if base_dir is None:
            local_appdata = os.environ.get("LOCALAPPDATA")
            if local_appdata:
                self.base_dir = Path(local_appdata) / "Grace" / "enrollment"
            else:
                self.base_dir = Path.home() / ".grace" / "enrollment"
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)

        if use_dpapi is not None:
            self._use_dpapi = use_dpapi and _HAS_WIN32CRYPT and sys.platform.startswith("win")
        else:
            self._use_dpapi = _HAS_WIN32CRYPT and sys.platform.startswith("win")

        if not self._use_dpapi and _HAS_WIN32CRYPT and sys.platform.startswith("win"):
            logger.info("DPAPI explicitly disabled for EnrollmentStorage")
        elif not self._use_dpapi:
            logger.debug("Operating EnrollmentStorage without DPAPI at-rest encryption")

    def _encrypt(self, data: bytes) -> bytes:
        if self._use_dpapi and _HAS_WIN32CRYPT:
            try:
                return win32crypt.CryptProtectData(data, "GraceEnrollmentAudio", None, None, None, 1)
            except Exception as e:
                logger.warning(f"DPAPI encryption failed, storing raw ({e})")
                return data
        return data

    def _decrypt(self, data: bytes) -> bytes:
        if self._use_dpapi and _HAS_WIN32CRYPT:
            try:
                _, decrypted = win32crypt.CryptUnprotectData(data, None, None, None, 1)
                return decrypted
            except Exception:
                # May have been stored unencrypted or on different machine/user
                return data
        return data

    def get_session_dir(self, session_id: str) -> Path:
        sdir = self.base_dir / session_id
        sdir.mkdir(parents=True, exist_ok=True)
        return sdir

    def save_utterance(
        self,
        session_id: str,
        utterance_id: str,
        phrase_id: str,
        phrase_text: str,
        category: str,
        pcm_bytes: bytes,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Atomically save utterance audio with WAV header and encrypted metadata."""
        sdir = self.get_session_dir(session_id)
        audio_filename = f"{utterance_id}.wav.enc" if self._use_dpapi else f"{utterance_id}.wav"
        meta_filename = f"{utterance_id}.json.enc" if self._use_dpapi else f"{utterance_id}.json"

        # Combine WAV header + PCM bytes
        wav_header = _make_wav_header(len(pcm_bytes))
        full_wav = wav_header + pcm_bytes

        # Encrypt and save audio
        encrypted_audio = self._encrypt(full_wav)
        audio_path = sdir / audio_filename
        tmp_audio = sdir / f"{audio_filename}.tmp"
        with open(tmp_audio, "wb") as f:
            f.write(encrypted_audio)
        tmp_audio.replace(audio_path)

        # Meta
        meta = {
            "utterance_id": utterance_id,
            "session_id": session_id,
            "phrase_id": phrase_id,
            "phrase_text": phrase_text,
            "category": category,
            "pcm_bytes_len": len(pcm_bytes),
            "audio_file": audio_filename,
            "encrypted": self._use_dpapi,
        }
        if extra_meta:
            meta.update(extra_meta)

        encrypted_meta = self._encrypt(json.dumps(meta, indent=2).encode("utf-8"))
        meta_path = sdir / meta_filename
        tmp_meta = sdir / f"{meta_filename}.tmp"
        with open(tmp_meta, "wb") as f:
            f.write(encrypted_meta)
        tmp_meta.replace(meta_path)

        return audio_path

    def load_utterance_pcm(self, session_id: str, utterance_id: str) -> Optional[bytes]:
        """Load and decrypt utterance audio, stripping the 44-byte WAV header to return raw PCM."""
        sdir = self.get_session_dir(session_id)
        candidate_paths = [
            sdir / f"{utterance_id}.wav.enc",
            sdir / f"{utterance_id}.wav",
        ]
        target = next((p for p in candidate_paths if p.exists()), None)
        if not target:
            return None

        with open(target, "rb") as f:
            raw = f.read()
        decrypted = self._decrypt(raw)

        # If has WAV header (starts with RIFF), strip 44 bytes
        if len(decrypted) >= 44 and decrypted[:4] == b"RIFF":
            return decrypted[44:]
        return decrypted

    def load_utterance_meta(self, session_id: str, utterance_id: str) -> Optional[Dict[str, Any]]:
        """Load and decrypt metadata for an utterance."""
        sdir = self.get_session_dir(session_id)
        candidate_paths = [
            sdir / f"{utterance_id}.json.enc",
            sdir / f"{utterance_id}.json",
        ]
        target = next((p for p in candidate_paths if p.exists()), None)
        if not target:
            return None

        with open(target, "rb") as f:
            raw = f.read()
        decrypted = self._decrypt(raw)
        try:
            return json.loads(decrypted.decode("utf-8"))
        except Exception:
            return None

    def delete_utterance(self, session_id: str, utterance_id: str) -> bool:
        """Securely delete all files associated with a single utterance."""
        sdir = self.get_session_dir(session_id)
        deleted = False
        for ext in (".wav", ".wav.enc", ".json", ".json.enc", ".tmp"):
            path = sdir / f"{utterance_id}{ext}"
            if path.exists():
                try:
                    path.unlink()
                    deleted = True
                except Exception as e:
                    logger.warning(f"Failed to delete {path}: {e}")
        return deleted

    def delete_session(self, session_id: str) -> bool:
        """Delete an entire enrollment session directory and all its contents."""
        sdir = self.base_dir / session_id
        if sdir.exists() and sdir.is_dir():
            shutil.rmtree(sdir, ignore_errors=True)
            return True
        return False

    def list_recordings(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List metadata for all saved enrollment recordings for inspection or voice queries."""
        recordings: List[Dict[str, Any]] = []
        if session_id:
            session_dirs = [self.base_dir / session_id] if (self.base_dir / session_id).exists() else []
        else:
            session_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]

        for sdir in session_dirs:
            for mfile in sdir.glob("*.json*"):
                if mfile.name.endswith(".tmp"):
                    continue
                try:
                    with open(mfile, "rb") as f:
                        raw = f.read()
                    data = json.loads(self._decrypt(raw).decode("utf-8"))
                    recordings.append(data)
                except Exception:
                    continue
        return recordings
