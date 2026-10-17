"""Speech enrollment session manager with listen-and-repeat and switch input support (PLAN §6, Task 14).

Features:
1. Short (10-15 min, configurable) sessions designed to prevent vocal and cognitive fatigue.
2. Resumable mid-session with atomic state persistence.
3. Next session suggestions without nagging alerts.
4. Listen-and-repeat flow: Grace prompts; user repeats; big visual text.
5. Switch input integration (single press: continue/record, double press: skip, long press: redo).
6. Adaptive stopping with concrete metrics ("commands 97%, yes/no 99.5%").
"""

from __future__ import annotations

import datetime
from enum import Enum
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from grace.enrollment.audio_validator import AudioQualityValidator, ValidationResult
from grace.enrollment.evaluator import AdaptiveStoppingEvaluator
from grace.enrollment.phrase_generator import EnrollmentPhrase, PhraseGenerator
from grace.enrollment.storage import EnrollmentStorage
from grace.input.switch_input import SwitchAction

logger = logging.getLogger("grace.enrollment.session")


class SessionState(str, Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    AWAITING_INPUT = "awaiting_input"
    PAUSED = "paused"
    ROUND_COMPLETE = "round_complete"
    COMPLETED = "completed"


class EnrollmentSession:
    """Manages an individual speech enrollment session."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        duration_minutes: float = 12.0,
        storage: Optional[EnrollmentStorage] = None,
        phrase_generator: Optional[PhraseGenerator] = None,
        evaluator: Optional[AdaptiveStoppingEvaluator] = None,
        validator: Optional[AudioQualityValidator] = None,
        on_speak_prompt: Optional[Callable[[str], None]] = None,
        on_phrase_changed: Optional[Callable[[EnrollmentPhrase, int, int], None]] = None,
    ):
        self.session_id = session_id or f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.duration_seconds = duration_minutes * 60.0
        self.storage = storage or EnrollmentStorage()
        self.phrase_generator = phrase_generator or PhraseGenerator()
        self.evaluator = evaluator or AdaptiveStoppingEvaluator()
        self.validator = validator or AudioQualityValidator()
        self.on_speak_prompt = on_speak_prompt
        self.on_phrase_changed = on_phrase_changed

        self.state: SessionState = SessionState.INITIALIZING
        self.phrases: List[EnrollmentPhrase] = []
        self.current_index: int = 0
        self.current_round: int = 1
        self.elapsed_active_seconds: float = 0.0
        self._last_active_timestamp: Optional[float] = None
        self.completed_utterance_ids: List[str] = []
        self.suggested_next_session: Optional[str] = None
        self.headline_accuracy_report: str = ""

        self.state_file = self.storage.get_session_dir(self.session_id) / "session_state.json"

    def initialize_new_session(self) -> None:
        """Populate full phrase curriculum for a brand new enrollment session."""
        self.phrases = self.phrase_generator.generate_full_curriculum()
        self.current_index = 0
        self.current_round = 1
        self.state = SessionState.ACTIVE
        self._last_active_timestamp = time.monotonic()
        self._save_state()
        self._notify_phrase()

    def start_or_resume(self) -> bool:
        """Start new session or resume previous session from state file."""
        if self.state_file.exists():
            return self.load_state()
        self.initialize_new_session()
        return True

    def pause(self) -> None:
        """Pause active session and update elapsed time."""
        if self._last_active_timestamp:
            self.elapsed_active_seconds += (time.monotonic() - self._last_active_timestamp)
            self._last_active_timestamp = None
        self.state = SessionState.PAUSED
        self._save_state()

    def resume(self) -> None:
        """Resume paused session."""
        self.state = SessionState.ACTIVE
        self._last_active_timestamp = time.monotonic()
        self._save_state()
        self._notify_phrase()

    def is_time_expired(self) -> bool:
        """Check if session duration limit has been reached."""
        current_elapsed = self.elapsed_active_seconds
        if self._last_active_timestamp:
            current_elapsed += (time.monotonic() - self._last_active_timestamp)
        return current_elapsed >= self.duration_seconds

    def compute_suggested_next_session(self) -> str:
        """Suggest next session time (e.g. tomorrow at the same hour) without nagging."""
        tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
        # Round to convenient time (e.g. 10:00 AM or 2:00 PM based on current time)
        hour = tomorrow.hour
        if hour < 12:
            time_str = "10:00 AM"
        elif hour < 17:
            time_str = "2:30 PM"
        else:
            time_str = "6:00 PM"
        return f"Tomorrow at {time_str}"

    def get_current_phrase(self) -> Optional[EnrollmentPhrase]:
        """Return the current phrase to be repeated by the user."""
        if 0 <= self.current_index < len(self.phrases):
            return self.phrases[self.current_index]
        return None

    def submit_recording(
        self,
        pcm_bytes: bytes,
        transcribed_text: str = "",
    ) -> Tuple[bool, str, Optional[ValidationResult]]:
        """Process a user's recording for the current phrase.

        Returns:
            (success, message, validation_result)
        """
        phrase = self.get_current_phrase()
        if not phrase:
            return (False, "No phrase is active in this session.", None)

        # 1. Audio quality check (Hard Rule: reject clipped/silent only)
        val = self.validator.validate_pcm(pcm_bytes)
        if not val.is_valid:
            # Do not advance phrase; user repeats with constructive guidance
            return (False, val.message, val)

        # 2. Save encrypted audio to local storage
        utterance_id = f"utt_{self.current_round:02d}_{self.current_index:04d}_{phrase.phrase_id}"
        self.storage.save_utterance(
            session_id=self.session_id,
            utterance_id=utterance_id,
            phrase_id=phrase.phrase_id,
            phrase_text=phrase.text,
            category=phrase.category,
            pcm_bytes=pcm_bytes,
            extra_meta={
                "round": self.current_round,
                "is_held_out": phrase.is_held_out,
                "transcribed_text": transcribed_text,
            },
        )
        self.completed_utterance_ids.append(utterance_id)

        # 3. Record evaluation metrics
        if transcribed_text:
            self.evaluator.record_sample(
                category=phrase.category,
                target_text=phrase.text,
                transcribed_text=transcribed_text,
                is_held_out=phrase.is_held_out,
            )

        # 4. Advance to next phrase
        self.current_index += 1
        if self.current_index >= len(self.phrases):
            self._finish_round()
            if self.state == SessionState.COMPLETED:
                return (True, f"Enrollment complete! {self.headline_accuracy_report}", val)

        # 5. Check session time expiration
        if self.is_time_expired():
            self.pause()
            self.suggested_next_session = self.compute_suggested_next_session()
            self.headline_accuracy_report = self.evaluator.format_headline_report()
            msg = (
                f"Session completed for today ({int(self.duration_seconds/60)} minutes). "
                f"Accuracy: {self.headline_accuracy_report}. "
                f"Suggested next session: {self.suggested_next_session}."
            )
            self._save_state()
            return (True, msg, val)

        self._save_state()
        self._notify_phrase()
        return (True, "Phrase accepted.", val)

    def skip_phrase(self) -> None:
        """Skip current phrase without recording (supported via double-press switch)."""
        if self.current_index < len(self.phrases):
            self.current_index += 1
            if self.current_index >= len(self.phrases):
                self._finish_round()
            self._save_state()
            self._notify_phrase()

    def redo_last_phrase(self) -> bool:
        """Redo previous phrase recording (supported via long-press switch)."""
        if self.current_index > 0:
            self.current_index -= 1
            if self.completed_utterance_ids:
                last_id = self.completed_utterance_ids.pop()
                self.storage.delete_utterance(self.session_id, last_id)
            self._save_state()
            self._notify_phrase()
            return True
        return False

    def handle_switch_input(self, action: SwitchAction) -> None:
        """Map accessibility switch actions directly into enrollment operations.

        - Single press (YES): Repeat/speak phrase prompt or proceed.
        - Double press (NO): Skip current phrase.
        - Long press (STOP): Redo previous phrase.
        """
        if action == SwitchAction.YES:
            # Single press: repeat phrase prompt or proceed
            phrase = self.get_current_phrase()
            if phrase and self.on_speak_prompt:
                self.on_speak_prompt(phrase.text)
        elif action == SwitchAction.NO:
            # Double press: skip phrase
            logger.info("Switch double-press (NO): skipping phrase")
            self.skip_phrase()
        elif action == SwitchAction.STOP:
            # Long press: redo previous phrase
            logger.info("Switch long-press (STOP): redoing previous phrase")
            self.redo_last_phrase()

    def _finish_round(self) -> None:
        """Conclude current curriculum round and evaluate adaptive stopping."""
        self.evaluator.conclude_round()
        should_stop, rationale, weak_cats = self.evaluator.evaluate_stopping_decision()
        self.headline_accuracy_report = self.evaluator.format_headline_report()

        if should_stop:
            self.state = SessionState.COMPLETED
            logger.info(f"Adaptive stopping triggered: {rationale}")
        else:
            self.current_round += 1
            # Adapt curriculum: focus on weak categories
            if weak_cats:
                filtered_curriculum = [
                    p for p in self.phrase_generator.generate_full_curriculum()
                    if p.category in weak_cats
                ]
                self.phrases = filtered_curriculum if filtered_curriculum else self.phrase_generator.generate_full_curriculum()
            else:
                self.phrases = self.phrase_generator.generate_full_curriculum()
            self.current_index = 0
            self.state = SessionState.ACTIVE

    def _notify_phrase(self) -> None:
        phrase = self.get_current_phrase()
        if phrase and self.on_phrase_changed:
            self.on_phrase_changed(phrase, self.current_index + 1, len(self.phrases))

    def _save_state(self) -> None:
        """Atomically persist session state to disk."""
        data = {
            "session_id": self.session_id,
            "state": self.state.value,
            "current_index": self.current_index,
            "current_round": self.current_round,
            "elapsed_active_seconds": self.elapsed_active_seconds,
            "suggested_next_session": self.suggested_next_session,
            "headline_accuracy_report": self.headline_accuracy_report,
            "completed_utterances_count": len(self.completed_utterance_ids),
            "phrases": [
                {
                    "phrase_id": p.phrase_id,
                    "text": p.text,
                    "category": p.category,
                    "sub_category": p.sub_category,
                    "is_held_out": p.is_held_out,
                }
                for p in self.phrases
            ],
        }
        tmp = self.state_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.state_file)

    def load_state(self) -> bool:
        """Load and resume persisted session state."""
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.state = SessionState(data.get("state", SessionState.PAUSED.value))
            self.current_index = data.get("current_index", 0)
            self.current_round = data.get("current_round", 1)
            self.elapsed_active_seconds = data.get("elapsed_active_seconds", 0.0)
            self.suggested_next_session = data.get("suggested_next_session")
            self.headline_accuracy_report = data.get("headline_accuracy_report", "")
            phrases_data = data.get("phrases", [])
            self.phrases = [
                EnrollmentPhrase(
                    phrase_id=p["phrase_id"],
                    text=p["text"],
                    category=p["category"],
                    sub_category=p.get("sub_category", ""),
                    is_held_out=p.get("is_held_out", False),
                )
                for p in phrases_data
            ]
            self._last_active_timestamp = time.monotonic()
            logger.info(f"Resumed session {self.session_id} at phrase {self.current_index + 1}/{len(self.phrases)}")
            self._notify_phrase()
            return True
        except Exception as e:
            logger.warning(f"Failed to load session state: {e}")
            return False
