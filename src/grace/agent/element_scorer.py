"""Decision Model for Element Selection in Grace (PLAN §3, §5, Task 13).

Scores candidate element summaries given a target description in words from the tool caller.
Enforces confidence threshold and margin checks before returning an element:
  - top_score >= threshold
  - (top_score - runner_up_score) >= margin
Otherwise flags decision as ambiguous, triggering escalation down the decision ladder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from grace.fleet import SlotConfig, get_fleet_config

logger = logging.getLogger("grace.agent.element_scorer")


@dataclass
class ElementCandidate:
    """Compact summary of a UI element candidate."""
    id: int
    name: str
    control_type: str
    parent_path: str = ""
    window_title: str = ""
    neighbouring_labels: str = ""
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)

    @classmethod
    def from_node_or_dict(cls, node_or_dict: Any) -> "ElementCandidate":
        if isinstance(node_or_dict, dict):
            return cls(
                id=int(node_or_dict.get("id", -1)),
                name=str(node_or_dict.get("name", "")),
                control_type=str(node_or_dict.get("control_type", node_or_dict.get("role", ""))),
                parent_path=str(node_or_dict.get("parent_path", "")),
                window_title=str(node_or_dict.get("window_title", "")),
                neighbouring_labels=str(node_or_dict.get("neighbouring_labels", "")),
                bounds=tuple(node_or_dict.get("bounds", (0, 0, 0, 0))),
            )
        # ElementNode or UIElement object
        elem_id = getattr(node_or_dict, "id", getattr(node_or_dict, "index", -1))
        name = getattr(node_or_dict, "name", "")
        role = getattr(node_or_dict, "role", getattr(node_or_dict, "control_type", ""))
        bounds = getattr(node_or_dict, "bounds", getattr(node_or_dict, "rect", (0, 0, 0, 0)))
        return cls(
            id=int(elem_id if elem_id is not None else -1),
            name=str(name or ""),
            control_type=str(role or ""),
            bounds=tuple(bounds) if isinstance(bounds, (list, tuple)) and len(bounds) == 4 else (0, 0, 0, 0),
        )

    def summary_text(self) -> str:
        """Formatted compact summary (never the raw tree)."""
        parts = [f"[{self.id}] {self.control_type}: \"{self.name}\""]
        if self.parent_path:
            parts.append(f"in {self.parent_path}")
        if self.neighbouring_labels:
            parts.append(f"near {self.neighbouring_labels}")
        return " | ".join(parts)


@dataclass
class ScoredSelection:
    """Result of scoring candidates with safety threshold verification."""
    selected_id: Optional[int]
    top_score: float
    runner_up_score: float
    margin: float
    is_confident: bool
    candidate_scores: List[Tuple[ElementCandidate, float]]
    reason: str


class ElementScorer:
    """Specialist decision model scoring elements against target descriptions."""

    def __init__(self, slot_config: Optional[SlotConfig] = None):
        if slot_config is None:
            fleet = get_fleet_config()
            slot_config = fleet.get_slot("element_selection")
        self.config = slot_config
        self.threshold = slot_config.threshold
        self.margin = slot_config.margin

    def score_candidates(
        self,
        target_description: str,
        candidates: Sequence[Any],
        window_title: str = "",
    ) -> ScoredSelection:
        """Score candidates against the target description and check safety margins."""
        clean_target = (target_description or "").lower().strip()
        if not clean_target or not candidates:
            return ScoredSelection(
                selected_id=None,
                top_score=0.0,
                runner_up_score=0.0,
                margin=0.0,
                is_confident=False,
                candidate_scores=[],
                reason="No target description or empty candidate list",
            )

        parsed_candidates = [
            ElementCandidate.from_node_or_dict(c) if not isinstance(c, ElementCandidate) else c
            for c in candidates
        ]

        # Calculate scores: exact name match > token overlap > control type match
        scores: List[Tuple[ElementCandidate, float]] = []
        target_tokens = set(clean_target.split())

        for cand in parsed_candidates:
            cand_name = cand.name.lower()
            score = 0.0

            if cand_name == clean_target:
                score = 0.98
            elif clean_target in cand_name:
                score = 0.88
            elif cand_name in clean_target and len(cand_name) > 2:
                score = 0.82
            else:
                cand_tokens = set(cand_name.split())
                overlap = len(target_tokens & cand_tokens)
                if overlap > 0:
                    score = 0.50 + (0.35 * (overlap / max(len(target_tokens), 1)))

            # Small bonus if control type is named in target
            if cand.control_type.lower() in clean_target:
                score = min(1.0, score + 0.08)

            scores.append((cand, round(score, 3)))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)

        if not scores:
            return ScoredSelection(
                selected_id=None,
                top_score=0.0,
                runner_up_score=0.0,
                margin=0.0,
                is_confident=False,
                candidate_scores=[],
                reason="No candidate elements scored",
            )

        top_cand, top_score = scores[0]
        runner_up_score = scores[1][1] if len(scores) > 1 else 0.0
        score_margin = round(top_score - runner_up_score, 3)

        # Check safety rule (PLAN §3, §5):
        # 1. Top score must clear the threshold
        # 2. Must beat runner up by at least the margin
        is_confident = (top_score >= self.threshold) and (score_margin >= self.margin)

        if is_confident:
            reason = f"Cleared threshold ({top_score} >= {self.threshold}) with margin {score_margin} >= {self.margin}"
            selected_id = top_cand.id
        else:
            if top_score < self.threshold:
                reason = f"Top score ({top_score}) below threshold ({self.threshold})"
            else:
                reason = f"Score margin ({score_margin}) below required margin ({self.margin}) against runner-up"
            selected_id = None

        return ScoredSelection(
            selected_id=selected_id,
            top_score=top_score,
            runner_up_score=runner_up_score,
            margin=score_margin,
            is_confident=is_confident,
            candidate_scores=scores,
            reason=reason,
        )
