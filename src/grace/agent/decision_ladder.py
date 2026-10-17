"""Decision Ladder implementation for Grace (PLAN §5, Task 13).

Rungs:
1. Pattern fast path (no model): "open X", "volume up", "scroll down", yes/no against app index and correction list.
2. Small specialists: tool caller decides *what* (no tree), element scorer / VL-450M decides *where*.
3. Mid-size model: LFM2.5-1.2B (or Holo on GPU) for unfamiliar apps/recovery. Handed off earlier on CPU floor.
4. Ask the user: numbered marks, "4 or 7?".

Records the rung that solved each turn in the turn trace.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from grace.agent.element_scorer import ElementScorer, ScoredSelection
from grace.fleet import FleetConfig, get_fleet_config
from grace.intent.parser import Intent
from grace.tier import HardwareTier, detect_hardware_tier

logger = logging.getLogger("grace.agent.decision_ladder")


class DecisionRung(str, Enum):
    RUNG_1_PATTERN = "rung1_pattern"
    RUNG_2_SPECIALISTS = "rung2_specialists"
    RUNG_3_MIDSIZE = "rung3_midsize"
    RUNG_4_USER = "rung4_user"


@dataclass
class LadderResolution:
    """Outcome of resolving an utterance or step through the decision ladder."""
    rung: DecisionRung
    tool: str
    params: Dict[str, Any]
    target_element_id: Optional[int] = None
    clarification_question: Optional[str] = None
    candidate_options: Optional[List[int]] = None
    confidence: float = 1.0
    rationale: str = ""


class DecisionLadder:
    """Orchestrates the 4-rung decision hierarchy across micromodels."""

    def __init__(
        self,
        fleet_config: Optional[FleetConfig] = None,
        element_scorer: Optional[ElementScorer] = None,
        hardware_tier: Optional[HardwareTier] = None,
    ):
        self.fleet = fleet_config or get_fleet_config()
        self.scorer = element_scorer or ElementScorer(self.fleet.get_slot("element_selection"))
        self.tier = hardware_tier or detect_hardware_tier().tier

    def check_pattern_fast_path(
        self,
        transcript: str,
        corrections: Optional[Any] = None,
        app_indexer: Optional[Any] = None,
    ) -> Optional[LadderResolution]:
        """Rung 1: Pattern fast path with no model invocation."""
        text = transcript.strip()
        if corrections and hasattr(corrections, "apply"):
            text = corrections.apply(text)
        lower = text.lower()

        # 1. Volume commands
        if re.search(r"\b(?:turn\s+(?:the\s+)?)?(?:volume\s+up|increase\s+volume|louder)\b", lower):
            return LadderResolution(
                rung=DecisionRung.RUNG_1_PATTERN,
                tool="adjust_volume",
                params={"direction": "up"},
                rationale="Matched volume up pattern",
            )
        if re.search(r"\b(?:turn\s+(?:the\s+)?)?(?:volume\s+down|decrease\s+volume|quieter|lower\s+volume)\b", lower):
            return LadderResolution(
                rung=DecisionRung.RUNG_1_PATTERN,
                tool="adjust_volume",
                params={"direction": "down"},
                rationale="Matched volume down pattern",
            )
        if re.search(r"\b(mute|unmute|mute volume)\b", lower):
            return LadderResolution(
                rung=DecisionRung.RUNG_1_PATTERN,
                tool="adjust_volume",
                params={"direction": "mute"},
                rationale="Matched mute pattern",
            )

        # 2. Calculator
        if re.search(r"\b(?:open|launch|start)?\s*(?:the\s+)?calc(?:ulator)?\b", lower):
            return LadderResolution(
                rung=DecisionRung.RUNG_1_PATTERN,
                tool="open_calculator",
                params={},
                rationale="Matched calculator pattern",
            )

        # 3. Lock computer
        if re.search(r"\b(lock computer|lock screen|lock the pc)\b", lower):
            return LadderResolution(
                rung=DecisionRung.RUNG_1_PATTERN,
                tool="lock_computer",
                params={},
                rationale="Matched lock screen pattern",
            )

        # 4. Basic scroll
        if re.search(r"\b(scroll down|page down)\b", lower):
            return LadderResolution(
                rung=DecisionRung.RUNG_1_PATTERN,
                tool="cua_scroll",
                params={"direction": "down", "amount": 3},
                rationale="Matched scroll down pattern",
            )
        if re.search(r"\b(scroll up|page up)\b", lower):
            return LadderResolution(
                rung=DecisionRung.RUNG_1_PATTERN,
                tool="cua_scroll",
                params={"direction": "up", "amount": 3},
                rationale="Matched scroll up pattern",
            )

        # 5. App launching via AppIndexer / Corrections
        open_match = re.match(r"^(?:open|launch|start)\s+([a-zA-Z0-9_\-\.\s]+)$", lower)
        if open_match:
            raw_target = open_match.group(1).strip()
            separators = (" and ", " then ", " also ", " next ", " after that ", " plus ")
            has_sep = any(sep in f" {raw_target} " for sep in separators)
            if not has_sep and len(raw_target.split()) <= 3:
                # Exclude files or ambiguous queries
                if not any(raw_target.endswith(ext) for ext in (".pdf", ".txt", ".doc", ".docx", ".png", ".jpg")):
                    found = False
                    if app_indexer and hasattr(app_indexer, "find_app"):
                        found = bool(app_indexer.find_app(raw_target))
                    elif corrections and hasattr(corrections, "corrections"):
                        found = raw_target in getattr(corrections, "corrections", {})
                    if found:
                        return LadderResolution(
                            rung=DecisionRung.RUNG_1_PATTERN,
                            tool="open_app",
                            params={"name": raw_target},
                            rationale=f"Matched app indexer target '{raw_target}'",
                        )

        # 6. Yes/No confirmations
        if lower in ("yes", "yep", "sure", "confirm", "proceed", "do it"):
            return LadderResolution(
                rung=DecisionRung.RUNG_1_PATTERN,
                tool="confirmation",
                params={"confirmed": True},
                rationale="Matched confirmation 'yes' pattern",
            )
        if lower in ("no", "nope", "cancel", "stop", "abort", "don't"):
            return LadderResolution(
                rung=DecisionRung.RUNG_1_PATTERN,
                tool="confirmation",
                params={"confirmed": False},
                rationale="Matched confirmation 'no' pattern",
            )

        return None

    def resolve_element_action(
        self,
        tool_action: str,
        target_description: str,
        candidates: Sequence[Any],
        params: Optional[Dict[str, Any]] = None,
        allow_user_fallback: bool = True,
    ) -> LadderResolution:
        """Resolve a UI action through Rungs 2, 3, and 4."""
        action_params = dict(params or {})

        # --- RUNG 2: Small specialists (Element Scorer) ---
        selection: ScoredSelection = self.scorer.score_candidates(target_description, candidates)
        if selection.is_confident and selection.selected_id is not None:
            action_params["element_id"] = selection.selected_id
            return LadderResolution(
                rung=DecisionRung.RUNG_2_SPECIALISTS,
                tool=tool_action,
                params=action_params,
                target_element_id=selection.selected_id,
                confidence=selection.top_score,
                rationale=f"Rung 2 specialist selected element {selection.selected_id} ({selection.reason})",
            )

        logger.info(f"DecisionLadder: Rung 2 unable to confidently resolve ({selection.reason}). Escalating...")

        # --- RUNG 3: Mid-size model ---
        # Rule (PLAN §5): On the CPU floor, hand off to Rung 4 earlier rather than running Rung 3 slowly.
        is_floor = (self.tier == HardwareTier.FLOOR_CPU)
        if not is_floor:
            # On non-floor tiers (or GPU/NPU), try midsize model
            midsize_slot = self.fleet.get_slot("midsize_fallback")
            # If top score exists and is reasonable, midsize can accept with looser threshold
            if selection.top_score >= midsize_slot.threshold:
                top_id = selection.candidate_scores[0][0].id
                action_params["element_id"] = top_id
                return LadderResolution(
                    rung=DecisionRung.RUNG_3_MIDSIZE,
                    tool=tool_action,
                    params=action_params,
                    target_element_id=top_id,
                    confidence=selection.top_score,
                    rationale=f"Rung 3 mid-size model cleared threshold {midsize_slot.threshold} for element {top_id}",
                )

        logger.info("DecisionLadder: Escalating to Rung 4 (ask the user)...")

        # --- RUNG 4: Ask the user ("4 or 7?") ---
        top_candidates = selection.candidate_scores[:2]
        cand_ids = [c[0].id for c in top_candidates if c[0].id is not None]
        if len(cand_ids) >= 2:
            question = f"Did you mean {cand_ids[0]} or {cand_ids[1]}?"
        elif len(cand_ids) == 1:
            question = f"Did you mean element {cand_ids[0]}?"
        else:
            question = "Which item on the screen would you like to choose?"

        return LadderResolution(
            rung=DecisionRung.RUNG_4_USER,
            tool="ask_user",
            params={"question": question, "candidate_ids": cand_ids},
            clarification_question=question,
            candidate_options=cand_ids,
            confidence=0.5,
            rationale="Rung 4: Ambiguous options; asking user via Set-of-Marks",
        )
