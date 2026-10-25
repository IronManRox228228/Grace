"""Tests for the stateful simulated desktop (scripts/holo_tape_drive.py fixtures)."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from grace.agent.perception import ScreenSnapshot
from grace.harness.sim_desktop import (
    PdfToNotesSim,
    ReadPdfSim,
    SimDispatcher,
    SimPerception,
    TwoStepSim,
    YouTubeMusicSim,
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


def test_create_sim_for_tape_all_scenarios():
    tape_two_step = MagicMock()
    tape_two_step.meta = {"scenario": "agentic_two_step"}
    assert isinstance(create_sim_for_tape(tape_two_step), TwoStepSim)

    tape_read_pdf = MagicMock()
    tape_read_pdf.meta = {"scenario": "agentic_read_pdf"}
    assert isinstance(create_sim_for_tape(tape_read_pdf), ReadPdfSim)

    tape_yt = MagicMock()
    tape_yt.meta = {"scenario": "agentic_youtube_music"}
    assert isinstance(create_sim_for_tape(tape_yt), YouTubeMusicSim)

    tape_pdf_notes = MagicMock()
    tape_pdf_notes.meta = {"scenario": "agentic_pdf_to_notes"}
    assert isinstance(create_sim_for_tape(tape_pdf_notes), PdfToNotesSim)


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


class TestYouTubeMusicSim:
    def test_initial_state_not_goal(self):
        sim = YouTubeMusicSim()
        assert not sim.is_goal_state()
        snap = sim.get_snapshot()
        assert "YouTube Music" in snap.active_window.title

    def test_youtube_music_flow_reaches_goal(self):
        sim = YouTubeMusicSim()
        # 1. Click search box
        res1 = sim.dispatch(Intent(tool="cua_click", params={"element_id": 2}))
        assert res1.get("status") == "ok"
        assert sim.current_state_name == "search_focused"
        assert not sim.is_goal_state()

        # 2. Type query
        res2 = sim.dispatch(Intent(tool="cua_type_text", params={"text": "signal by home"}))
        assert res2.get("status") == "ok"
        assert sim.current_state_name == "query_typed"
        assert not sim.is_goal_state()

        # 3. Press Enter to submit search
        res3 = sim.dispatch(Intent(tool="cua_press_key", params={"key": "enter"}))
        assert res3.get("status") == "ok"
        assert sim.current_state_name == "results_shown"
        assert not sim.is_goal_state()

        # 4. Click Play / Signal
        res4 = sim.dispatch(Intent(tool="cua_click", params={"element_id": 10}))
        assert res4.get("status") == "ok"
        assert sim.current_state_name == "song_playing"
        assert sim.is_goal_state()

        snap = sim.get_snapshot()
        assert any(e.container == "Player bar" and "signal · home" in (e.name or e.value or "").lower()
                   for e in snap.graph.elements)

    def test_youtube_music_click_search_button_submits_query(self):
        sim = YouTubeMusicSim()
        sim.dispatch(Intent(tool="cua_click", params={"element_id": 2}))
        sim.dispatch(Intent(tool="cua_type_text", params={"text": "signal by home"}))
        assert sim.current_state_name == "query_typed"

        # Clicking element 1 (Search button) submits search
        res = sim.dispatch(Intent(tool="cua_click", params={"element_id": 1}))
        assert res.get("status") == "ok"
        assert sim.current_state_name == "results_shown"

        # Clicking Play reaches goal
        res2 = sim.dispatch(Intent(tool="cua_click", params={"element_id": 11}))
        assert res2.get("status") == "ok"
        assert sim.current_state_name == "song_playing"
        assert sim.is_goal_state()

    def test_wrong_action_changes_nothing(self):
        sim = YouTubeMusicSim()
        # Click an unrelated element (e.g. element 5 "Library")
        res = sim.dispatch(Intent(tool="cua_click", params={"element_id": 5}))
        assert res.get("status") == "ok"
        assert sim.current_state_name == "initial"
        assert not sim.is_goal_state()


class TestPdfToNotesSim:
    def test_initial_state_not_goal(self):
        sim = PdfToNotesSim()
        assert not sim.is_goal_state()
        snap = sim.get_snapshot()
        assert snap.active_window.title == "Desktop"

    def test_pdf_to_notes_flow_reaches_goal(self):
        sim = PdfToNotesSim()
        # 1. search_files
        res1 = sim.dispatch(Intent(tool="search_files", params={"query": "report.pdf"}))
        assert res1.get("status") == "ok"
        assert not sim.is_goal_state()

        # 2. read_pdf
        res2 = sim.dispatch(Intent(tool="read_pdf", params={"path": res1["files"][0]}))
        assert res2.get("status") == "ok"
        assert "Final submission deadline: 14 November" in res2.get("text", "")
        assert not sim.is_goal_state()

        # 3. open_app notepad
        res3 = sim.dispatch(Intent(tool="open_app", params={"name": "notepad"}))
        assert res3.get("status") == "ok"
        assert sim.current_state_name == "notepad_open"
        assert not sim.is_goal_state()

        # 4. type deadline into Notepad
        res4 = sim.dispatch(Intent(tool="cua_type_text", params={"text": "14 November"}))
        assert res4.get("status") == "ok"
        assert sim.current_state_name == "note_typed"
        assert sim.is_goal_state()

        snap = sim.get_snapshot()
        editor = next(e for e in snap.graph.elements if e.role == "document")
        assert editor.value == "14 November"

    def test_wrong_text_typed_does_not_reach_goal(self):
        sim = PdfToNotesSim()
        sim.dispatch(Intent(tool="open_app", params={"name": "notepad"}))
        res = sim.dispatch(Intent(tool="cua_type_text", params={"text": "arbitrary text"}))
        assert res.get("status") == "ok"
        assert not sim.is_goal_state()

    def test_documents_root_from_argument(self, tmp_path):
        custom_root = str(tmp_path / "pdf_notes_docs")
        sim = PdfToNotesSim(documents_root=custom_root)
        assert sim._documents_root == custom_root
        assert sim._pdf_path == os.path.join(custom_root, "report.pdf")
        res = sim.dispatch(Intent(tool="search_files", params={"query": "report.pdf"}))
        assert os.path.join(custom_root, "report.pdf") in res["files"]

    def test_documents_root_from_environment(self, tmp_path, monkeypatch):
        env_root = str(tmp_path / "env_pdf_notes")
        monkeypatch.setenv("GRACE_DOCUMENTS_DIR", env_root)
        sim = PdfToNotesSim()
        assert sim._documents_root == env_root
        assert sim._pdf_path == os.path.join(env_root, "report.pdf")


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
