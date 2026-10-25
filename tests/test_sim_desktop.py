"""Tests for the stateful simulated desktop (scripts/holo_tape_drive.py fixtures)."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from grace.agent.perception import ScreenSnapshot
from grace.harness.sim_desktop import (
    ReadPdfSim,
    SimDispatcher,
    SimPerception,
    TwoStepSim,
    create_sim_for_tape,
    render_element_tree_image,
)
from grace.intent.parser import Intent
from grace.perception.elements import ElementNode
from scripts.demo_gate import evaluate_tape_verdict


def test_render_element_tree_image_generates_valid_png():
    elements = [
        ElementNode(id=0, role="edit", name="Search", rect=(100, 100, 400, 150),
                    center=(250, 125), focused=True, value="test query"),
        ElementNode(id=1, role="button", name="Submit", rect=(420, 100, 500, 150),
                    center=(460, 125), focused=False),
    ]
    png_bytes = render_element_tree_image("Test Window", elements, width=1280, height=720)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 100
    # Valid PNG signature
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


class TestTwoStepSim:
    def test_initial_state_not_goal(self):
        sim = TwoStepSim()
        assert sim.current_state_name == "initial"
        assert not sim.is_goal_state()
        snap = sim.get_snapshot()
        assert isinstance(snap, ScreenSnapshot)
        assert snap.active_window.title == "Grace test page - Edge"
        assert snap.png_bytes is not None

        # Element 2 is unfocused
        el2 = next(e for e in snap.graph.elements if e.id == 2)
        assert not el2.focused
        assert el2.value == ""

    def test_wrong_action_leaves_state_unchanged(self):
        sim = TwoStepSim()
        # Typing before clicking search box: leaves state in 'initial'
        res = sim.dispatch(Intent(tool="cua_type_text", params={"text": "hello"}))
        assert not res.get("result", {}).get("ok", True)
        assert sim.current_state_name == "initial"
        assert not sim.is_goal_state()

        # Clicking unrelated element (Refresh)
        sim.dispatch(Intent(tool="cua_click", params={"element_id": 1}))
        assert sim.current_state_name == "initial"

    def test_correct_click_focuses_search_box(self):
        sim = TwoStepSim()
        res = sim.dispatch(Intent(tool="cua_click", params={"element_id": 2}))
        assert res.get("status") == "ok"
        assert sim.current_state_name == "focused"
        assert not sim.is_goal_state()

        snap = sim.get_snapshot()
        el2 = next(e for e in snap.graph.elements if e.id == 2)
        assert el2.focused
        assert snap.graph.focused().id == 2

    def test_type_hello_reaches_goal_state(self):
        sim = TwoStepSim()
        # 1. Click search box
        sim.dispatch(Intent(tool="cua_click", params={"element_id": 2}))
        # 2. Type "hello"
        res = sim.dispatch(Intent(tool="cua_type_text", params={"text": "hello"}))
        assert res.get("status") == "ok"
        assert sim.current_state_name == "typed"
        assert sim.is_goal_state()

        snap = sim.get_snapshot()
        el2 = next(e for e in snap.graph.elements if e.id == 2)
        assert el2.value == "hello"

    def test_enter_submits_search_and_stays_in_goal_state(self):
        sim = TwoStepSim()
        sim.dispatch(Intent(tool="cua_click", params={"element_id": 2}))
        sim.dispatch(Intent(tool="cua_type_text", params={"text": "hello"}))
        res = sim.dispatch(Intent(tool="cua_press_key", params={"key": "enter"}))
        assert res.get("status") == "ok"
        assert sim.current_state_name == "submitted"
        assert sim.is_goal_state()

        snap = sim.get_snapshot()
        assert any(e.name == "Search Results for 'hello'" for e in snap.graph.elements)


class TestReadPdfSim:
    def test_initial_state_not_goal(self):
        sim = ReadPdfSim()
        assert sim.current_state_name == "initial"
        assert not sim.is_goal_state()
        snap = sim.get_snapshot()
        assert snap.active_window.title == "Untitled - Notepad"

    def test_search_files_then_read_pdf_reaches_goal(self):
        sim = ReadPdfSim()
        # 1. search_files
        res1 = sim.dispatch(Intent(tool="search_files", params={"query": "report.pdf"}))
        assert res1.get("status") == "ok"
        assert "report.pdf" in res1.get("text", "")
        assert sim.current_state_name == "file_found"
        assert not sim.is_goal_state()

        # 2. read_pdf
        pdf_path = res1["files"][0]
        res2 = sim.dispatch(Intent(tool="read_pdf", params={"path": pdf_path}))
        assert res2.get("status") == "ok"
        assert "projections" in res2.get("text", "")
        assert sim.current_state_name == "pdf_read"
        assert sim.is_goal_state()

    def test_direct_read_pdf_reaches_goal(self):
        sim = ReadPdfSim()
        res = sim.dispatch(Intent(tool="read_pdf", params={"path": sim._pdf_path}))
        assert res.get("status") == "ok"
        assert sim.is_goal_state()

    def test_open_file_updates_window_and_reaches_goal(self):
        sim = ReadPdfSim()
        res = sim.dispatch(Intent(tool="open_file", params={"name": "report.pdf"}))
        assert res.get("status") == "ok"
        assert sim.current_state_name == "pdf_opened"
        assert sim.is_goal_state()
        snap = sim.get_snapshot()
        assert "report.pdf" in snap.active_window.title


    def test_documents_root_from_argument(self, tmp_path):
        custom_root = str(tmp_path / "custom_docs")
        sim = ReadPdfSim(documents_root=custom_root)
        assert sim._documents_root == custom_root
        assert sim._pdf_path == os.path.join(custom_root, "report.pdf")
        res = sim.dispatch(Intent(tool="search_files", params={"query": "report.pdf"}))
        assert os.path.join(custom_root, "report.pdf") in res["files"]

    def test_documents_root_from_environment(self, tmp_path, monkeypatch):
        env_root = str(tmp_path / "env_docs")
        monkeypatch.setenv("GRACE_DOCUMENTS_DIR", env_root)
        sim = ReadPdfSim()
        assert sim._documents_root == env_root
        assert sim._pdf_path == os.path.join(env_root, "report.pdf")
        res = sim.dispatch(Intent(tool="search_files", params={"query": "report.pdf"}))
        assert os.path.join(env_root, "report.pdf") in res["files"]


class TestSimDispatcherAndPerception:
    def test_sim_perception_and_dispatcher_integration(self):
        sim = TwoStepSim()
        perception = SimPerception(sim)
        inner_dispatcher = MagicMock()
        inner_dispatcher._emit_tool_started = AsyncMock()
        inner_dispatcher._emit_tool_finished = AsyncMock()
        inner_dispatcher._emit_cua_started = AsyncMock()
        dispatcher = SimDispatcher(sim, inner_dispatcher)

        # Snapshot before click
        snap0 = perception.capture_snapshot()
        assert not next(e for e in snap0.graph.elements if e.id == 2).focused

        # Dispatch click
        res = asyncio.run(dispatcher.execute(Intent(tool="cua_click", params={"element_id": 2})))
        assert res["status"] == "ok"
        assert inner_dispatcher._emit_cua_started.await_count == 1
        assert inner_dispatcher._emit_tool_finished.await_count == 1

        # Snapshot after click
        snap1 = perception.capture_snapshot()
        assert next(e for e in snap1.graph.elements if e.id == 2).focused
        assert not sim.is_goal_state()


def test_early_done_not_in_goal_state_gives_not_proven():
    sim = TwoStepSim()
    # Planner does nothing, declares done early
    assert not sim.is_goal_state()
    res = {
        "goal_status": "ok",
        "ok": sim.is_goal_state(),
        "sim_goal_reached": sim.is_goal_state(),
        "steps": [{"action": "converse", "is_completed": True}],
        "final_response": "I am done.",
    }
    verdict = evaluate_tape_verdict(res)
    assert verdict == "NOT PROVEN"
