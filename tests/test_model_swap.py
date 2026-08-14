"""Tests for the local model hot-swap manager.

This exists for the fully-local setup where the planner and UI-TARS can't
both fit on one GPU at once: exactly one of them is ever resident, and
swapping means stopping the current llama-server process and starting the
other model in its place.
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.llm.model_swap import ModelSwapManager, LocalModelSpec


def make_manager():
    specs = {
        "planner": LocalModelSpec(name="planner", model_path="planner.gguf", ngl=999),
        "grounder": LocalModelSpec(
            name="grounder", model_path="ui-tars.gguf", mmproj_path="mmproj.gguf", ngl=999
        ),
    }
    return ModelSwapManager(
        host="127.0.0.1", port=8080, specs=specs, llama_server_exe="llama-server.exe"
    )


def fake_process():
    proc = MagicMock()
    proc.poll.return_value = None
    return proc


class TestSwapTo:
    def test_starts_the_requested_model(self):
        manager = make_manager()
        with patch("grace.llm.model_swap.subprocess.Popen", return_value=fake_process()) as popen, \
             patch.object(ModelSwapManager, "_wait_for_health", return_value=True):
            ok = asyncio.run(manager.swap_to("planner"))

        assert ok is True
        assert manager.active == "planner"
        args = popen.call_args.args[0]
        assert "planner.gguf" in args

    def test_is_a_noop_when_already_active(self):
        manager = make_manager()
        with patch("grace.llm.model_swap.subprocess.Popen", return_value=fake_process()) as popen, \
             patch.object(ModelSwapManager, "_wait_for_health", return_value=True):
            asyncio.run(manager.swap_to("planner"))
            asyncio.run(manager.swap_to("planner"))

        assert popen.call_count == 1

    def test_stops_the_previous_process_before_starting_the_next(self):
        manager = make_manager()
        first_proc, second_proc = fake_process(), fake_process()
        with patch("grace.llm.model_swap.subprocess.Popen", side_effect=[first_proc, second_proc]), \
             patch.object(ModelSwapManager, "_wait_for_health", return_value=True):
            asyncio.run(manager.swap_to("planner"))
            asyncio.run(manager.swap_to("grounder"))

        assert first_proc.terminate.called
        assert manager.active == "grounder"

    def test_mmproj_is_only_passed_for_the_model_that_has_one(self):
        manager = make_manager()
        with patch("grace.llm.model_swap.subprocess.Popen", return_value=fake_process()) as popen, \
             patch.object(ModelSwapManager, "_wait_for_health", return_value=True), \
             patch("grace.llm.model_swap.os.path.exists", return_value=True):
            asyncio.run(manager.swap_to("grounder"))
            grounder_args = popen.call_args.args[0]

            asyncio.run(manager.swap_to("planner"))
            planner_args = popen.call_args.args[0]

        assert "--mmproj" in grounder_args
        assert "--mmproj" not in planner_args

    def test_unknown_model_name_raises(self):
        manager = make_manager()
        with pytest.raises(KeyError):
            asyncio.run(manager.swap_to("nonexistent"))

    def test_failed_start_leaves_active_unset(self):
        manager = make_manager()
        with patch("grace.llm.model_swap.subprocess.Popen", return_value=fake_process()), \
             patch.object(ModelSwapManager, "_wait_for_health", return_value=False):
            ok = asyncio.run(manager.swap_to("planner"))

        assert ok is False
        assert manager.active is None


class TestShutdown:
    def test_shutdown_terminates_the_running_process(self):
        manager = make_manager()
        proc = fake_process()
        with patch("grace.llm.model_swap.subprocess.Popen", return_value=proc), \
             patch.object(ModelSwapManager, "_wait_for_health", return_value=True):
            asyncio.run(manager.swap_to("planner"))
            asyncio.run(manager.shutdown())

        assert proc.terminate.called
        assert manager.active is None
