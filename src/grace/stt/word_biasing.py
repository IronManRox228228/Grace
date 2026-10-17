"""Live word biasing and hotword boosting for Grace ASR (PLAN.md §6, Task 10).

Builds a dynamic bias vocabulary from:
1. Grace core command vocabulary and tool action names.
2. Installed applications from the app indexer.
3. Open window titles from the desktop snapshot.
4. Contact names from confirmed memory facts only.

Feeds biased words to the active ASR engine:
- Whisper: `initial_prompt` keyword priming.
- Parakeet / sherpa-onnx: hotword boosting list.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, List, Optional, Set

logger = logging.getLogger("grace.stt.word_biasing")

# Core command vocabulary for Grace voice interaction
CORE_COMMAND_VOCABULARY: List[str] = [
    "Grace",
    "open",
    "close",
    "click",
    "type",
    "press",
    "scroll",
    "drag",
    "undo",
    "stop",
    "cancel",
    "yes",
    "no",
    "repeat",
    "help",
    "describe screen",
    "set speech rate",
    "search files",
    "lock computer",
    "volume up",
    "volume down",
    "mute",
    "unmute",
    "maximize",
    "minimize",
    "restore",
    "switch window",
    "remember",
    "recall",
    "forget",
    "button",
    "submit",
    "confirm",
]

_CLEAN_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-\.\'\s]+$")


class WordBiaser:
    """Manages dynamic word biasing across desktop context and user facts."""

    def __init__(self, max_words: int = 100) -> None:
        self.max_words = max(10, max_words)
        self._cached_words: List[str] = list(CORE_COMMAND_VOCABULARY)
        self._last_signature: Optional[str] = None

    @property
    def current_words(self) -> List[str]:
        return list(self._cached_words)

    def extract_window_title_tokens(self, titles: Iterable[str]) -> List[str]:
        """Extract meaningful application/document names from open window titles."""
        extracted: List[str] = []
        for title in titles:
            if not title or not title.strip():
                continue
            # Typical Windows title format: "Document - AppName" or "AppName"
            parts = [p.strip() for p in re.split(r"[-–—|:]", title) if p.strip()]
            for part in parts:
                # Discard common non-identifying window title tokens
                if len(part) >= 2 and len(part) <= 40 and _CLEAN_TOKEN_RE.match(part):
                    extracted.append(part)
        return extracted

    def extract_confirmed_contacts(self, facts: Optional[Iterable[Any]]) -> List[str]:
        """Extract contact names strictly from confirmed memory facts."""
        contacts: List[str] = []
        if not facts:
            return contacts

        for fact in facts:
            # Fact can be a dict, Fact object, or tuple
            topic = getattr(fact, "topic", "") or (fact.get("topic") if isinstance(fact, dict) else "")
            value = getattr(fact, "value", "") or (fact.get("value") if isinstance(fact, dict) else "")
            is_confirmed = getattr(fact, "is_confirmed", False) or (fact.get("is_confirmed", False) if isinstance(fact, dict) else False)
            provenance = getattr(fact, "provenance", "") or (fact.get("provenance") if isinstance(fact, dict) else "")

            # Strict security gate: only confirmed facts
            if not is_confirmed and provenance != "user_confirmed":
                continue

            topic_str = str(topic).lower()
            if "contact" in topic_str or "person" in topic_str or "friend" in topic_str or "doctor" in topic_str:
                val_clean = str(value).strip()
                if val_clean and len(val_clean) <= 40 and _CLEAN_TOKEN_RE.match(val_clean):
                    contacts.append(val_clean)

        return contacts

    def build_bias_list(
        self,
        app_indexer: Optional[Any] = None,
        window_titles: Optional[Iterable[str]] = None,
        confirmed_contacts: Optional[Iterable[Any]] = None,
    ) -> List[str]:
        """Build a prioritized, deduplicated, size-capped word bias list."""
        seen: Set[str] = set()
        result: List[str] = []

        def add_token(token: str) -> None:
            norm = token.strip()
            if not norm:
                return
            key = norm.lower()
            if key not in seen and len(result) < self.max_words:
                seen.add(key)
                result.append(norm)

        # 1. Core command vocabulary (highest priority for voice control)
        for cmd in CORE_COMMAND_VOCABULARY:
            add_token(cmd)

        # 2. Open window titles (immediate desktop context)
        if window_titles:
            for w_tok in self.extract_window_title_tokens(window_titles):
                add_token(w_tok)

        # 3. Confirmed contacts (personal voice vocabulary)
        if confirmed_contacts:
            for contact in self.extract_confirmed_contacts(confirmed_contacts):
                add_token(contact)

        # 4. Installed applications from app indexer
        if app_indexer is not None:
            try:
                # AppIndexer can provide list of installed apps
                apps = getattr(app_indexer, "list_apps", None)
                if callable(apps):
                    for app_name in apps():
                        add_token(str(app_name))
                elif hasattr(app_indexer, "_cache") and isinstance(app_indexer._cache, dict):
                    for app_name in app_indexer._cache.keys():
                        add_token(str(app_name))
            except Exception as e:
                logger.debug(f"Failed to query app indexer for word biasing: {e}")

        self._cached_words = result
        return result

    def to_whisper_initial_prompt(self, words: Optional[List[str]] = None) -> str:
        """Format bias list as an initial prompt string for faster-whisper."""
        vocab = words if words is not None else self._cached_words
        if not vocab:
            return ""
        # Whisper initial prompt is natural comma-separated text
        return "Grace voice commands and desktop applications: " + ", ".join(vocab) + "."

    def to_sherpa_hotwords(self, words: Optional[List[str]] = None) -> str:
        """Format bias list as hotwords string for Sherpa-ONNX / NeMo Parakeet."""
        vocab = words if words is not None else self._cached_words
        if not vocab:
            return ""
        # Sherpa-ONNX hotwords syntax: phrases separated by /
        # e.g., "Grace / open / Chrome / Notepad / Calculator"
        return " / ".join(vocab)
