"""Tests for Fleet Configuration, Decision Ladder, and Specialist Element Scorer (PLAN §3, §5, Task 13)."""

from unittest.mock import MagicMock
import pytest

from grace.fleet import (
    SlotConfig,
    FleetConfig,
    build_gpu_single_preset,
    build_cpu_fleet_preset,
    get_fleet_config,
)
from grace.agent.element_scorer import ElementScorer, ElementCandidate
from grace.agent.decision_ladder import DecisionLadder, DecisionRung
from grace.agent.planner import Planner
from grace.tier import HardwareTier


def test_fleet_config_presets():
    gpu_preset = build_gpu_single_preset()
    assert gpu_preset.preset == "gpu-single"
    assert gpu_preset.is_single is True
    assert gpu_preset.is_fleet is False
    assert gpu_preset.get_slot("tool_calling").model_name == "Holo-3.1-4B"
    assert gpu_preset.get_slot("element_selection").model_name == "Holo-3.1-4B"

    cpu_preset = build_cpu_fleet_preset()
    assert cpu_preset.preset == "cpu-fleet"
    assert cpu_preset.is_fleet is True
    assert cpu_preset.is_single is False
    assert "LFM" in cpu_preset.get_slot("tool_calling").model_name
    assert "GLiNER" in cpu_preset.get_slot("element_selection").model_name
    assert cpu_preset.get_slot("element_selection").threshold == 0.70
    assert cpu_preset.get_slot("element_selection").margin == 0.15

    # Auto tier pick
    fleet_default = get_fleet_config()
    assert isinstance(fleet_default, FleetConfig)


def test_element_scorer_threshold_and_margin():
    slot_cfg = SlotConfig(
        slot_name="element_selection",
        model_name="test-scorer",
        model_path="dummy",
        runtime="onnx",
        threshold=0.70,
        margin=0.15,
    )
    scorer = ElementScorer(slot_cfg)

    candidates = [
        ElementCandidate(id=1, name="Send message", control_type="Button"),
        ElementCandidate(id=2, name="Attach file", control_type="Button"),
        ElementCandidate(id=3, name="Settings", control_type="Button"),
    ]

    # Clear winner
    res = scorer.score_candidates("Send message", candidates)
    assert res.is_confident is True
    assert res.selected_id == 1
    assert res.top_score >= 0.70
    assert res.margin >= 0.15

    # Close runner-up (ambiguity -> must not guess)
    ambiguous_cands = [
        ElementCandidate(id=10, name="Save file", control_type="Button"),
        ElementCandidate(id=11, name="Save as", control_type="Button"),
    ]
    res_ambig = scorer.score_candidates("Save", ambiguous_cands)
    # Both score identically or close -> margin check fails
    assert res_ambig.is_confident is False
    assert res_ambig.selected_id is None

    # Below threshold
    res_none = scorer.score_candidates("Completely Unrelated Widget", candidates)
    assert res_none.is_confident is False
    assert res_none.selected_id is None


def test_decision_ladder_rung1_pattern():
    ladder = DecisionLadder()

    r1 = ladder.check_pattern_fast_path("open calculator")
    assert r1 is not None
    assert r1.rung == DecisionRung.RUNG_1_PATTERN
    assert r1.tool == "open_calculator"

    r2 = ladder.check_pattern_fast_path("volume up")
    assert r2 is not None
    assert r2.rung == DecisionRung.RUNG_1_PATTERN
    assert r2.tool == "adjust_volume"

    r3 = ladder.check_pattern_fast_path("scroll down")
    assert r3 is not None
    assert r3.rung == DecisionRung.RUNG_1_PATTERN
    assert r3.tool == "cua_scroll"

    r4 = ladder.check_pattern_fast_path("lock screen")
    assert r4 is not None
    assert r4.rung == DecisionRung.RUNG_1_PATTERN
    assert r4.tool == "lock_computer"

    # Multi-step or non-pattern command should return None (escalates)
    r_multi = ladder.check_pattern_fast_path("open whatsapp and send a message to john")
    assert r_multi is None


def test_decision_ladder_specialist_and_user_escalation():
    # Test on CPU floor tier (hermetic: fleet forced to cpu-fleet so the result
    # does not depend on the physical GPU present on the machine running the suite).
    ladder_cpu = DecisionLadder(
        hardware_tier=HardwareTier.FLOOR_CPU,
        fleet_config=build_cpu_fleet_preset(),
    )

    candidates = [
        ElementCandidate(id=4, name="Submit", control_type="Button"),
        ElementCandidate(id=7, name="Cancel", control_type="Button"),
    ]

    # Clear match -> Rung 2
    res_r2 = ladder_cpu.resolve_element_action(
        tool_action="cua_click",
        target_description="Submit button",
        candidates=candidates,
    )
    assert res_r2.rung == DecisionRung.RUNG_2_SPECIALISTS
    assert res_r2.target_element_id == 4

    # Ambiguous match on CPU floor -> escalates directly to Rung 4 (ask user "4 or 7?")
    ambiguous_cands = [
        ElementCandidate(id=4, name="Download", control_type="Button"),
        ElementCandidate(id=7, name="Download mirror", control_type="Button"),
    ]
    res_r4 = ladder_cpu.resolve_element_action(
        tool_action="cua_click",
        target_description="Download",
        candidates=ambiguous_cands,
    )
    assert res_r4.rung == DecisionRung.RUNG_4_USER
    assert res_r4.candidate_options == [4, 7]
    assert "Did you mean 4 or 7?" in res_r4.clarification_question


def test_tool_caller_responsibilities_split():
    mock_llm = MagicMock()
    # Cockpit 8: In fleet mode (is_single=False), the tool caller never sees the tree
    # when hide_tree=True, and sees it when hide_tree=False.
    planner = Planner(llm=mock_llm, is_single=False)

    tree_markdown = '[1] Button: "Send"\n[2] Button: "Cancel"'
    prompt = planner.build_prompt(
        goal="Send email",
        elements_prompt=tree_markdown,
        hide_tree=True,
    )
    assert tree_markdown not in prompt
    assert "A specialist decision model" in prompt
    assert "target_name" in prompt

    prompt_fleet_tree = planner.build_prompt(
        goal="Send email",
        elements_prompt=tree_markdown,
        hide_tree=False,
    )
    assert tree_markdown in prompt_fleet_tree

    # Cockpit 8: In single mode (is_single=True), the element list is always dropped
    planner_single = Planner(llm=mock_llm, is_single=True)
    prompt_single = planner_single.build_prompt(
        goal="Send email",
        elements_prompt=tree_markdown,
        hide_tree=False,
    )
    assert tree_markdown not in prompt_single
