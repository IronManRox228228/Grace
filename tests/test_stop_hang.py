"""Tests for the stop-and-hang fixes.

An in-flight ``planner.plan()`` call is now run as an ``asyncio`` task, raced
against the cancel flag (polled every 100 ms) and a per-step time limit
(``PLAN_STEP_TIMEOUT_S``, default 45 s; ``0`` disables).

No real desktop, mic, speakers or network: the planner is a mock that sleeps
and never returns, standing in for an in-flight llama-server request.
"""

import asyncio
import logging
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.agent.loop import AgentLoop
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.agent.planner import PlannedStep
from grace.config import Config


def snapshot(title="Untitled - Notepad", graph=None):
    return ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title=title, class_name="X", rect=(0, 0, 800, 600)),
        ocr_lines=[], width=1920, height=1080, graph=graph,
    )


def make_loop(plan_fn, config=None, dispatch_result=None):
    gemma = AsyncMock()
    dispatcher = AsyncMock()
    dispatcher.execute.return_value = dispatch_result or {"status": "ok"}
    perception = MagicMock()
    perception.capture_snapshot.return_value = snapshot()
    perception.capture_snapshot_async.return_value = snapshot()
    kwargs = dict(gemma=gemma, dispatcher=dispatcher, perception=perception)
    if config is not None:
        kwargs["config"] = config
    loop = AgentLoop(**kwargs)
    loop._planner.plan = plan_fn
    return loop, gemma, dispatcher


async def never_returns(*args, **kwargs):
    """A planner that never answers: stands in for an in-flight llama-server request."""
    await asyncio.sleep(60)


class TestCancelInterruptsInFlightPlan:
    """A spoken "stop" must interrupt a planner call that is still running."""

    def test_cancel_set_while_plan_pending_returns_cancel_result_quickly(self):
        cancelled = []
        plan_started = asyncio.Event()

        async def plan(*a, **k):
            plan_started.set()
            try:
                await never_returns(*a, **k)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        loop, _, _ = make_loop(plan)

        async def drive():
            task = asyncio.create_task(loop.run(user_goal="open notepad"))
            await plan_started.wait()  # the plan task is in flight
            loop.request_cancel()
            t0 = time.monotonic()
            res = await asyncio.wait_for(task, timeout=2.0)
            return res, time.monotonic() - t0

        res, elapsed = asyncio.run(drive())

        assert res["status"] == "cancelled"
        assert cancelled, "the plan task must have been cancelled"
        assert elapsed < 0.5, f"cancel took {elapsed:.3f}s, expected < 0.5 s"


class TestPlannerStepTimeout:
    """PLAN_STEP_TIMEOUT_S bounds how long one planning step may run."""

    def test_timeout_counts_as_plan_failure_and_logs_warning(self, caplog):
        cfg = Config(planner_step_timeout_s=1, agent_max_consecutive_plan_failures=1)
        loop, _, _ = make_loop(never_returns, config=cfg)

        async def drive():
            return await asyncio.wait_for(
                loop.run(user_goal="open notepad"), timeout=10.0
            )

        with caplog.at_level(logging.WARNING, logger="grace.agent.loop"):
            res = asyncio.run(drive())

        assert res["status"] == "planner_failed"
        messages = [rec.getMessage() for rec in caplog.records]
        assert any(
            "Planner step" in m and "timed out after" in m for m in messages
        ), messages

    def test_zero_disables_the_timeout(self):
        cfg = Config(planner_step_timeout_s=0)
        cancelled = []
        plan_started = asyncio.Event()

        async def plan(*a, **k):
            plan_started.set()
            try:
                await never_returns(*a, **k)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        loop, _, _ = make_loop(plan, config=cfg)

        async def drive():
            task = asyncio.create_task(loop.run(user_goal="open notepad"))
            await plan_started.wait()  # the plan task is in flight
            loop.request_cancel()
            t0 = time.monotonic()
            res = await asyncio.wait_for(task, timeout=2.0)
            return res, time.monotonic() - t0

        res, elapsed = asyncio.run(drive())

        # Timeout disabled: the never-returning plan is stopped by cancel, not
        # abandoned as a planner failure.
        assert res["status"] == "cancelled"
        assert cancelled, "the plan task must have been cancelled"
        assert elapsed < 0.5, f"cancel took {elapsed:.3f}s, expected < 0.5 s"

    def test_normal_fast_plan_call_is_unaffected(self):
        async def fast_plan(*a, **k):
            return PlannedStep(
                action="converse", params={}, is_completed=True, final_response="All done."
            )

        loop, _, dispatcher = make_loop(fast_plan)
        loop._planner.calls_made = 0

        res = asyncio.run(loop.run(user_goal="read the question on screen"))

        assert res["status"] == "ok"
        assert res["final_response"] == "All done."
        assert dispatcher.execute.await_count == 0


class TestPlanSlowerThanPollStillCompletes:
    """A planner slower than the 100 ms poll must not be killed by it.

    With the old `asyncio.wait_for(task, 0.1)` poll, a planner call that
    needed more than 100 ms was cancelled on the first poll and could never
    return. This test would have caught that: the mock sleeps 0.5 s and the
    helper must still return its step.
    """

    def test_plan_that_sleeps_0_5_s_returns_its_step(self):
        plan_started = asyncio.Event()

        async def slow_plan(*a, **k):
            plan_started.set()
            await asyncio.sleep(0.5)
            return PlannedStep(
                action="converse", params={}, is_completed=True, final_response="All done."
            )

        loop, _, dispatcher = make_loop(slow_plan)
        loop._planner.calls_made = 0

        res = asyncio.run(loop.run(user_goal="read the question on screen"))

        assert res["status"] == "ok"
        assert res["final_response"] == "All done."
        assert dispatcher.execute.await_count == 0
