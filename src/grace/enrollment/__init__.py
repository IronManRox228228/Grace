"""Speech Enrollment & Personalization Package for Grace (PLAN §6, Task 14).

Provides:
- Phrase generation for wake words, safety, tool commands, app index, Harvard sentences, and personal facts.
- Audio quality validation (clipping and silence checks; never rejecting unclear speech).
- DPAPI-protected local at-rest recording storage.
- Adaptive stopping with held-out splits, per-category accuracy tracking, and concrete reporting.
- Short resumable sessions with switch input (skip/redo) and fatigue-aware next session scheduling.
- Ongoing drift tracking and consent-gated everyday sample intake.
"""

from grace.enrollment.phrase_generator import EnrollmentPhrase, PhraseGenerator
from grace.enrollment.audio_validator import AudioQualityValidator, ValidationResult
from grace.enrollment.storage import EnrollmentStorage
from grace.enrollment.evaluator import AdaptiveStoppingEvaluator, CategoryMetrics
from grace.enrollment.session import EnrollmentSession, SessionState
from grace.enrollment.ongoing import OngoingTracker

__all__ = [
    "EnrollmentPhrase",
    "PhraseGenerator",
    "AudioQualityValidator",
    "ValidationResult",
    "EnrollmentStorage",
    "AdaptiveStoppingEvaluator",
    "CategoryMetrics",
    "EnrollmentSession",
    "SessionState",
    "OngoingTracker",
]
