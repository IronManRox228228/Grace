"""Personal speech correction list for Grace ASR (PLAN.md §6, Task 10).

Stores learned `heard -> meant` word/phrase substitutions locally and immediately.
Applies corrections to raw ASR transcripts before capability routing.
Provides voice-driven learning ("no, Chrome") and voice-driven reversal ("forget that correction").

Rule: Never learn a correction from model output alone — only from an explicit user
correction or a confirmed disambiguation pick.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger("grace.stt.corrections")

# Regex to detect "not X, Y" or "not X but Y"
_NOT_X_BUT_Y_RE = re.compile(
    r"^(?:no\s*,?\s*)?not\s+([a-zA-Z0-9_\-\.\'\s]+?)\s*(?:,|\s+but\s+)\s*([a-zA-Z0-9_\-\.\'\s]+)$",
    re.IGNORECASE,
)

# Regex to detect "no, I meant X" or "I meant X"
_I_MEANT_RE = re.compile(
    r"^(?:no\s*,?\s*)?(?:i\s+meant|actually|correction\s*:?)\s+([a-zA-Z0-9_\-\.\'\s]+)$",
    re.IGNORECASE,
)

# Regex to detect leading "no, X"
_NO_COMMA_RE = re.compile(
    r"^no\s*,\s*([a-zA-Z0-9_\-\.\'\s]+)$",
    re.IGNORECASE,
)

# Regex to detect voice reversal requests
_FORGET_CORRECTION_RE = re.compile(
    r"^(?:please\s+)?(?:forget|remove|undo)\s+(?:that\s+correction|the\s+correction|correction(?:\s+for\s+([a-zA-Z0-9_\-\.\'\s]+))?)$",
    re.IGNORECASE,
)


class PersonalCorrectionList:
    """Manages personal voice corrections with immediate persistence and voice reversal."""

    def __init__(self, filepath: Optional[Union[str, Path]] = None) -> None:
        self.filepath = Path(filepath) if filepath else None
        self._corrections: Dict[str, str] = {}
        self._last_added_heard: Optional[str] = None
        self.load()

    def load(self) -> None:
        """Load stored corrections from disk."""
        if not self.filepath or not self.filepath.is_file():
            return
        try:
            data = json.loads(self.filepath.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._corrections = {str(k).strip().lower(): str(v).strip() for k, v in data.items() if k and v}
                logger.info("Loaded %d personal voice corrections from %s", len(self._corrections), self.filepath)
        except Exception as e:
            logger.warning("Failed to load corrections from %s: %s", self.filepath, e)

    def save(self) -> None:
        """Persist corrections atomically to disk."""
        if not self.filepath:
            return
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write via temp file
            tmp = self.filepath.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._corrections, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.filepath)
        except Exception as e:
            logger.error("Failed to save corrections to %s: %s", self.filepath, e)

    def add_correction(self, heard: str, meant: str) -> None:
        """Store a new heard -> meant mapping immediately."""
        h_norm = heard.strip().lower()
        m_norm = meant.strip()
        if not h_norm or not m_norm or h_norm == m_norm.lower():
            return

        self._corrections[h_norm] = m_norm
        self._last_added_heard = h_norm
        self.save()
        logger.info("Stored personal voice correction: '%s' -> '%s'", h_norm, m_norm)

    def remove_correction(self, query: str) -> bool:
        """Remove a correction matching heard key or meant value."""
        q_norm = query.strip().lower()
        if not q_norm:
            return False

        # 1. Match exact heard key
        if q_norm in self._corrections:
            del self._corrections[q_norm]
            if self._last_added_heard == q_norm:
                self._last_added_heard = None
            self.save()
            logger.info("Removed correction for '%s'", q_norm)
            return True

        # 2. Match meant value
        for h, m in list(self._corrections.items()):
            if m.lower() == q_norm:
                del self._corrections[h]
                if self._last_added_heard == h:
                    self._last_added_heard = None
                self.save()
                logger.info("Removed correction for meant target '%s' (was '%s')", m, h)
                return True

        return False

    def remove_last_correction(self) -> Optional[str]:
        """Remove the most recently added correction."""
        if self._last_added_heard and self._last_added_heard in self._corrections:
            h = self._last_added_heard
            del self._corrections[h]
            self._last_added_heard = None
            self.save()
            return h
        elif self._corrections:
            # Pop the last entry
            last_key = list(self._corrections.keys())[-1]
            del self._corrections[last_key]
            self._last_added_heard = None
            self.save()
            return last_key
        return None

    def get_all(self) -> Dict[str, str]:
        """Return a copy of all active corrections."""
        return dict(self._corrections)

    def clear(self) -> None:
        """Clear all stored corrections."""
        self._corrections.clear()
        self._last_added_heard = None
        self.save()

    def apply(self, transcript: str) -> str:
        """Apply known personal corrections to an ASR transcript before routing."""
        if not transcript or not self._corrections:
            return transcript

        res = transcript
        # Sort keys by length descending to match longer phrases first
        for heard, meant in sorted(self._corrections.items(), key=lambda item: len(item[0]), reverse=True):
            # Whole-word regex substitution
            pattern = re.compile(rf"\b{re.escape(heard)}\b", re.IGNORECASE)
            res = pattern.sub(meant, res)

        return res

    def detect_explicit_correction(
        self, transcript: str, previous_transcript: Optional[str] = None
    ) -> Optional[Tuple[str, str]]:
        """Detect if the user utterance is an explicit correction.

        Returns (heard, meant) tuple if detected, otherwise None.
        """
        text = transcript.strip()

        # 1. Pattern: "not X, Y" or "not X but Y" (e.g. "not crow, Chrome")
        m1 = _NOT_X_BUT_Y_RE.match(text)
        if m1:
            heard = m1.group(1).strip()
            meant = m1.group(2).strip()
            return heard, meant

        # 2. Pattern: "I meant X" or "no, I meant X" or "correction: X"
        m2 = _I_MEANT_RE.match(text)
        if m2 and previous_transcript:
            meant = m2.group(1).strip()
            heard = self._infer_misheard(previous_transcript, meant)
            return heard, meant

        # 3. Pattern: "no, X" (e.g. "no, Chrome" following "open crow")
        m3 = _NO_COMMA_RE.match(text)
        if m3 and previous_transcript:
            meant = m3.group(1).strip()
            heard = self._infer_misheard(previous_transcript, meant)
            return heard, meant

        return None

    def _infer_misheard(self, previous_transcript: str, meant_text: str) -> str:
        """Infer the misheard token from previous transcript given meant replacement."""
        prev_words = previous_transcript.strip().split()
        meant_words = meant_text.strip().split()

        # If previous is a single token, that was the mishearing
        if len(prev_words) == 1:
            return prev_words[0]

        # If both have the same length and prefix matches (e.g. "open crow" vs "open Chrome")
        if len(prev_words) == len(meant_words) and len(prev_words) > 1:
            for p, m in zip(prev_words, meant_words):
                if p.lower() != m.lower():
                    return p

        # If user corrected just the last word (e.g. "open crow" -> meant "Chrome")
        if len(meant_words) == 1 and len(prev_words) > 1:
            return prev_words[-1]

        # Fallback: whole previous transcript
        return previous_transcript.strip()

    def detect_forget_request(self, transcript: str) -> Optional[str]:
        """Detect if user asks to forget a correction ("forget that correction").

        Returns "*" if user wants to forget the last correction, or specific word if named.
        """
        m = _FORGET_CORRECTION_RE.match(transcript.strip())
        if not m:
            return None

        specific_word = m.group(1)
        if specific_word:
            return specific_word.strip()
        return "*"
