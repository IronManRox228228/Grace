"""Tests for scripts/demo_gate.py verdict and status logic."""

from __future__ import annotations

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.demo_gate import (
    FROZEN_SCREEN_NOTE,
    compute_overall_status,
    evaluate_tape_verdict,
    extract_raw_planner_outputs,
)


def test_evaluate_tape_verdict_pass_on_status_ok():
    res = {
        "goal_status": "ok",
        "ok": True,
        "steps": [{"action": "click"}, {"action": "done"}],
        "final_response": "Goal reached successfully.",
    }
    assert evaluate_tape_verdict(res) == "PASS"


def test_evaluate_tape_verdict_not_proven_on_no_progress():
    # no_progress must be NOT PROVEN, never PASS
    res = {
        "goal_status": "no_progress",
        "ok": False,
        "steps": [{"action": "click"}, {"action": "click"}],
        "final_response": "I've tried the same thing several times without getting anywhere.",
    }
    assert evaluate_tape_verdict(res) == "NOT PROVEN"


def test_evaluate_tape_verdict_not_proven_on_honest_failures():
    # failed with explanation
    res1 = {
        "goal_status": "failed",
        "ok": False,
        "steps": [{"action": "click"}],
        "final_response": "Could not locate element.",
    }
    assert evaluate_tape_verdict(res1) == "NOT PROVEN"

    # budget_exceeded with explanation
    res2 = {
        "goal_status": "budget_exceeded",
        "ok": False,
        "steps": [{"action": "click"}] * 5,
        "final_response": "Exceeded budget.",
    }
    assert evaluate_tape_verdict(res2) == "NOT PROVEN"


def test_evaluate_tape_verdict_fail_on_stuck_loop():
    # More than 12 steps is a stuck loop / runaway
    res_ok_stuck = {
        "goal_status": "ok",
        "ok": True,
        "steps": [{"action": "click"}] * 13,
        "final_response": "Done",
    }
    assert evaluate_tape_verdict(res_ok_stuck) == "FAIL"

    res_np_stuck = {
        "goal_status": "no_progress",
        "ok": False,
        "steps": [{"action": "click"}] * 13,
        "final_response": "Stopped",
    }
    assert evaluate_tape_verdict(res_np_stuck) == "FAIL"


def test_evaluate_tape_verdict_fail_on_error_or_invalid():
    assert evaluate_tape_verdict({"goal_status": "error"}) == "FAIL"
    assert evaluate_tape_verdict({"goal_status": None}) == "FAIL"
    assert evaluate_tape_verdict({}) == "FAIL"
    assert evaluate_tape_verdict(None) == "FAIL"


def test_compute_overall_status():
    # All PASS -> PASS, 0
    all_pass = {
        "Smoke Grounding": "PASS",
        "Smoke Planner Parse": "PASS",
        "Agent Intent Routing": "PASS",
        "Tape Drive (agentic_two_step)": "PASS",
    }
    assert compute_overall_status(all_pass) == ("PASS", 0)

    # Any NOT PROVEN without FAIL -> NOT PROVEN, 2
    tape_not_proven = {
        "Smoke Grounding": "PASS",
        "Smoke Planner Parse": "PASS",
        "Agent Intent Routing": "PASS",
        "Tape Drive (agentic_two_step)": "NOT PROVEN",
        "Tape Drive (agentic_read_pdf)": "NOT PROVEN",
    }
    assert compute_overall_status(tape_not_proven) == ("NOT PROVEN", 2)

    # FAIL takes precedence over NOT PROVEN -> FAIL, 1
    has_fail = {
        "Smoke Grounding": "FAIL",
        "Smoke Planner Parse": "PASS",
        "Tape Drive (agentic_two_step)": "NOT PROVEN",
    }
    assert compute_overall_status(has_fail) == ("FAIL", 1)


def test_extract_raw_planner_outputs():
    res = {
        "llm_records": [
            {"call": "generate_intent", "raw": '{"intent": "agent"}'},
            {"call": "generate_text", "raw": '{"thought": "step 1", "action": "open"}'},
            {"call": "generate_text", "raw": '{"thought": "step 2", "action": "click"}'},
            {"call": "generate_text", "raw": '{"thought": "step 3", "action": "done"}'},
        ]
    }
    outputs = extract_raw_planner_outputs(res)
    assert len(outputs) == 3
    assert "step 1" in outputs[0]
    assert "step 2" in outputs[1]
    assert "step 3" in outputs[2]
    # Check last 2
    last_2 = outputs[-2:]
    assert len(last_2) == 2
    assert "step 2" in last_2[0]
    assert "step 3" in last_2[1]


def test_frozen_screen_note_content():
    expected_clause = (
        "the tape harness serves a frozen screen, so tapes can only prove "
        "routing and loop safety; task completion is proven only by a real-desktop run."
    )
    assert expected_clause in FROZEN_SCREEN_NOTE.lower()


def test_main_exit_code_not_proven(monkeypatch):
    monkeypatch.setattr("scripts.demo_gate.start_server", lambda *a, **k: None)
    monkeypatch.setattr("scripts.demo_gate.wait_for_server", lambda *a, **k: True)
    monkeypatch.setattr("scripts.demo_gate._llama_pids", lambda: [])
    monkeypatch.setattr("scripts.demo_gate.run_smoke_grounding", lambda *a, **k: True)
    monkeypatch.setattr("scripts.demo_gate.run_smoke_planner", lambda *a, **k: True)
    monkeypatch.setattr("scripts.demo_gate.run_intents_gate", lambda *a, **k: (True, 15, 15))
    monkeypatch.setattr("scripts.demo_gate.run_tapes_gate", lambda *a, **k: {
        "agentic_two_step": "NOT PROVEN",
        "agentic_read_pdf": "NOT PROVEN",
    })
    from scripts.demo_gate import main
    code = main(["--server-url", "http://127.0.0.1:8080"])
    assert code == 2


def test_main_exit_code_pass(monkeypatch):
    monkeypatch.setattr("scripts.demo_gate.start_server", lambda *a, **k: None)
    monkeypatch.setattr("scripts.demo_gate.wait_for_server", lambda *a, **k: True)
    monkeypatch.setattr("scripts.demo_gate._llama_pids", lambda: [])
    monkeypatch.setattr("scripts.demo_gate.run_smoke_grounding", lambda *a, **k: True)
    monkeypatch.setattr("scripts.demo_gate.run_smoke_planner", lambda *a, **k: True)
    monkeypatch.setattr("scripts.demo_gate.run_intents_gate", lambda *a, **k: (True, 15, 15))
    monkeypatch.setattr("scripts.demo_gate.run_tapes_gate", lambda *a, **k: {
        "agentic_two_step": "PASS",
        "agentic_read_pdf": "PASS",
    })
    from scripts.demo_gate import main
    code = main(["--server-url", "http://127.0.0.1:8080"])
    assert code == 0


def test_main_exit_code_fail(monkeypatch):
    monkeypatch.setattr("scripts.demo_gate.start_server", lambda *a, **k: None)
    monkeypatch.setattr("scripts.demo_gate.wait_for_server", lambda *a, **k: True)
    monkeypatch.setattr("scripts.demo_gate._llama_pids", lambda: [])
    monkeypatch.setattr("scripts.demo_gate.run_smoke_grounding", lambda *a, **k: False)
    monkeypatch.setattr("scripts.demo_gate.run_smoke_planner", lambda *a, **k: True)
    monkeypatch.setattr("scripts.demo_gate.run_intents_gate", lambda *a, **k: (True, 15, 15))
    monkeypatch.setattr("scripts.demo_gate.run_tapes_gate", lambda *a, **k: {
        "agentic_two_step": "NOT PROVEN",
        "agentic_read_pdf": "NOT PROVEN",
    })
    from scripts.demo_gate import main
    code = main(["--server-url", "http://127.0.0.1:8080"])
    assert code == 1

