"""Ongoing accuracy tracking and consent-gated everyday sample intake (PLAN §6, Task 14).

Features:
1. Tracks ASR accuracy over days/sessions.
2. When rolling accuracy drops (e.g. progressive dysarthria or fatigue changes), offers a short top-up session without nagging.
3. Consent-governed everyday utterance intake: only confirmed, successful everyday turns become local training samples, and only with explicit user consent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from grace.enrollment.storage import EnrollmentStorage

logger = logging.getLogger("grace.enrollment.ongoing")


@dataclass
class DailyAccuracyRecord:
    date_str: str
    total_turns: int = 0
    verified_turns: int = 0

    @property
    def accuracy(self) -> float:
        return (self.verified_turns / self.total_turns) if self.total_turns > 0 else 1.0


class OngoingTracker:
    """Monitors ongoing recognition accuracy and manages everyday speech intake."""

    def __init__(
        self,
        storage: Optional[EnrollmentStorage] = None,
        state_file: Optional[Path] = None,
        drop_threshold: float = 0.08,
        min_turns_to_evaluate: int = 15,
        user_consent_everyday_samples: bool = False,
    ):
        self.storage = storage or EnrollmentStorage()
        self.state_file = state_file or (self.storage.base_dir / "ongoing_tracker.json")
        self.drop_threshold = drop_threshold
        self.min_turns_to_evaluate = min_turns_to_evaluate
        self.user_consent = user_consent_everyday_samples

        self._baseline_accuracy: float = 0.95
        self._daily_records: Dict[str, DailyAccuracyRecord] = {}
        self._last_topup_offer_date: Optional[str] = None
        self._load_state()

    def set_user_consent(self, consent: bool) -> None:
        """Update explicit user consent for collecting everyday training samples."""
        self.user_consent = consent
        self._save_state()
        logger.info(f"User consent for everyday speech samples set to: {self.user_consent}")

    def record_turn(self, verified_correct: bool, date_str: Optional[str] = None) -> None:
        """Record the outcome of an everyday speech turn."""
        today = date_str or datetime.date.today().isoformat()
        if today not in self._daily_records:
            self._daily_records[today] = DailyAccuracyRecord(date_str=today)

        rec = self._daily_records[today]
        rec.total_turns += 1
        if verified_correct:
            rec.verified_turns += 1

        self._save_state()

    def get_recent_rolling_accuracy(self, days: int = 7) -> Optional[float]:
        """Compute rolling accuracy over the last N days."""
        sorted_dates = sorted(self._daily_records.keys(), reverse=True)[:days]
        if not sorted_dates:
            return None

        total = sum(self._daily_records[d].total_turns for d in sorted_dates)
        correct = sum(self._daily_records[d].verified_turns for d in sorted_dates)
        if total < self.min_turns_to_evaluate:
            return None

        return correct / total

    def check_for_topup_suggestion(self) -> Optional[str]:
        """Check if ongoing accuracy has dropped enough to offer a short top-up session.

        Does not nag: offers at most once every 7 days.
        """
        today = datetime.date.today().isoformat()
        if self._last_topup_offer_date:
            try:
                last_dt = datetime.date.fromisoformat(self._last_topup_offer_date)
                if (datetime.date.today() - last_dt).days < 7:
                    return None
            except Exception:
                pass

        rolling_acc = self.get_recent_rolling_accuracy(days=7)
        if rolling_acc is None:
            return None

        drop = self._baseline_accuracy - rolling_acc
        if drop >= self.drop_threshold:
            self._last_topup_offer_date = today
            self._save_state()
            return (
                f"Grace noticed your speech recognition accuracy is at {rolling_acc*100:.0f}%. "
                "Would you like to do a quick 5-minute top-up enrollment session?"
            )
        return None

    def add_everyday_sample(
        self,
        phrase_text: str,
        category: str,
        pcm_bytes: bytes,
        verified_success: bool,
    ) -> bool:
        """Store an everyday confirmed utterance as a local training sample if consent is granted."""
        # Hard Privacy Rule: explicit consent required
        if not self.user_consent:
            return False

        # Only store confirmed successful interactions
        if not verified_success:
            return False

        today = datetime.date.today().isoformat()
        session_id = f"everyday_{today}"
        h = hex(abs(hash(phrase_text + str(len(pcm_bytes)))))[2:10]
        utterance_id = f"turn_{h}"

        try:
            self.storage.save_utterance(
                session_id=session_id,
                utterance_id=utterance_id,
                phrase_id=f"everyday_{h}",
                phrase_text=phrase_text,
                category=category,
                pcm_bytes=pcm_bytes,
                extra_meta={"source": "confirmed_everyday_turn", "date": today},
            )
            return True
        except Exception as e:
            logger.warning(f"Could not save everyday training sample: {e}")
            return False

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._baseline_accuracy = data.get("baseline_accuracy", 0.95)
            self._last_topup_offer_date = data.get("last_topup_offer_date")
            self.user_consent = data.get("user_consent", False)
            records = data.get("daily_records", {})
            for d, rec in records.items():
                self._daily_records[d] = DailyAccuracyRecord(
                    date_str=d,
                    total_turns=rec.get("total_turns", 0),
                    verified_turns=rec.get("verified_turns", 0),
                )
        except Exception as e:
            logger.debug(f"Failed to load ongoing tracker state: {e}")

    def _save_state(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "baseline_accuracy": self._baseline_accuracy,
                "last_topup_offer_date": self._last_topup_offer_date,
                "user_consent": self.user_consent,
                "daily_records": {
                    d: {"total_turns": r.total_turns, "verified_turns": r.verified_turns}
                    for d, r in self._daily_records.items()
                },
            }
            tmp = self.state_file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(self.state_file)
        except Exception as e:
            logger.debug(f"Failed to save ongoing tracker state: {e}")
