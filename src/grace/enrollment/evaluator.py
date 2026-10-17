"""Adaptive stopping evaluator for speech enrollment (PLAN §6, Task 14).

Measures per-category held-out accuracy with current ASR.
Reports concrete numbers ("commands 97%, yes/no 99.5%").
Stops when accuracy stops improving; asks for more only in weak categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("grace.enrollment.evaluator")


def _normalize_text(text: str) -> str:
    """Normalize text for speech evaluation (lowercase, punctuation-stripped, whitespace-collapsed)."""
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


@dataclass
class CategoryMetrics:
    """Per-category speech recognition performance metrics."""
    category: str
    total_evaluated: int = 0
    correct: int = 0
    held_out_total: int = 0
    held_out_correct: int = 0

    @property
    def overall_accuracy(self) -> float:
        return (self.correct / self.total_evaluated) if self.total_evaluated > 0 else 0.0

    @property
    def held_out_accuracy(self) -> float:
        return (self.held_out_correct / self.held_out_total) if self.held_out_total > 0 else 0.0


class AdaptiveStoppingEvaluator:
    """Tracks per-category held-out accuracy and determines adaptive stopping."""

    def __init__(
        self,
        target_accuracy: float = 0.95,
        min_improvement_threshold: float = 0.015,
        weak_category_threshold: float = 0.88,
        min_held_out_samples_per_category: int = 3,
    ):
        self.target_accuracy = target_accuracy
        self.min_improvement_threshold = min_improvement_threshold
        self.weak_category_threshold = weak_category_threshold
        self.min_held_out_samples_per_category = min_held_out_samples_per_category

        self._categories: Dict[str, CategoryMetrics] = {}
        self._round_history: List[Dict[str, float]] = []

    def record_sample(
        self,
        category: str,
        target_text: str,
        transcribed_text: str,
        is_held_out: bool,
    ) -> bool:
        """Record one utterance's evaluation result against target text. Returns True if accurate."""
        if category not in self._categories:
            self._categories[category] = CategoryMetrics(category=category)

        metrics = self._categories[category]
        metrics.total_evaluated += 1
        if is_held_out:
            metrics.held_out_total += 1

        norm_target = _normalize_text(target_text)
        norm_trans = _normalize_text(transcribed_text)

        # Word-level overlap or exact match
        is_correct = (norm_target == norm_trans)
        if not is_correct and norm_target and norm_trans:
            target_words = norm_target.split()
            trans_words = norm_trans.split()
            # If at least 80% of words match in order
            matched = sum(1 for w in target_words if w in trans_words)
            if matched / len(target_words) >= 0.80 and len(trans_words) >= len(target_words) * 0.80:
                is_correct = True

        if is_correct:
            metrics.correct += 1
            if is_held_out:
                metrics.held_out_correct += 1

        return is_correct

    def compute_category_accuracies(self, use_held_out: bool = True) -> Dict[str, float]:
        """Return dict of category name to accuracy (held-out by default)."""
        accuracies: Dict[str, float] = {}
        for cat, metric in self._categories.items():
            if use_held_out and metric.held_out_total >= self.min_held_out_samples_per_category:
                accuracies[cat] = metric.held_out_accuracy
            else:
                accuracies[cat] = metric.overall_accuracy
        return accuracies

    def conclude_round(self) -> None:
        """Snapshot current round accuracies for improvement comparison."""
        current = self.compute_category_accuracies(use_held_out=True)
        self._round_history.append(current)

    def format_headline_report(self) -> str:
        """Speakable concrete report: 'commands 97%, yes/no 99.5%, wake word 98%'."""
        accs = self.compute_category_accuracies(use_held_out=True)
        if not accs:
            return "No evaluated categories yet."

        label_map = {
            "commands": "commands",
            "safety_confirmation": "yes/no",
            "wake_word": "wake word",
            "apps_folders": "apps and folders",
            "alphabet_digits": "alphabet and digits",
            "phonetic_sentences": "sentences",
            "personal_words": "personal words",
        }

        parts: List[str] = []
        for cat, acc in sorted(accs.items()):
            label = label_map.get(cat, cat.replace("_", " "))
            percentage = acc * 100.0
            if percentage == 100.0 or percentage == int(percentage):
                parts.append(f"{label} {int(percentage)}%")
            else:
                parts.append(f"{label} {percentage:.1f}%")

        return ", ".join(parts)

    def evaluate_stopping_decision(self) -> Tuple[bool, str, List[str]]:
        """Evaluate if enrollment should stop, or continue on weak categories.

        Returns:
            (should_stop, rationale, weak_categories)
        """
        accs = self.compute_category_accuracies(use_held_out=True)
        if not accs:
            return (False, "Initial round in progress", [])

        # Check weak categories (below threshold)
        weak_categories = [cat for cat, acc in accs.items() if acc < self.weak_category_threshold]

        # 1. Target accuracy reached across all evaluated categories
        if len(accs) >= 1 and not weak_categories and all(acc >= self.target_accuracy for acc in accs.values()):
            return (
                True,
                f"Target accuracy reached: {self.format_headline_report()}",
                [],
            )

        # 2. Check for plateau across rounds
        if len(self._round_history) >= 2:
            prev = self._round_history[-2]
            curr = self._round_history[-1]
            shared_cats = set(prev.keys()) & set(curr.keys())
            if shared_cats:
                avg_prev = sum(prev[c] for c in shared_cats) / len(shared_cats)
                avg_curr = sum(curr[c] for c in shared_cats) / len(shared_cats)
                delta = avg_curr - avg_prev

                # If improvement has slowed to below threshold and overall accuracy is solid
                if delta < self.min_improvement_threshold and avg_curr >= 0.85:
                    return (
                        True,
                        f"Accuracy has stabilized with no further gains (+{delta*100:.1f}%): {self.format_headline_report()}",
                        [],
                    )

        # 3. Continue, focusing on weak categories if any exist
        if weak_categories:
            cat_list = ", ".join(weak_categories)
            return (
                False,
                f"Accuracy improving; requesting additional samples in weak categories: {cat_list}",
                weak_categories,
            )

        return (False, "Continuing enrollment round to collect held-out benchmarks", [])
