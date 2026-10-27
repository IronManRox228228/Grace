"""Tests verifying honest escalation offline vs online (real-desktop.md #4)."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock
import pytest

from grace.agent.loop import AgentLoop
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.agent.planner import PlannedStep


def make_loop(responses, config=None):
    from grace.fleet import build_cpu_fleet_preset
    gemma = AsyncMock()
    gemma.generate_text.side_effect = responses
    dispatcher = AsyncMock()
    dispatcher.execute.return_value = {"status": "ok"}
    perception = MagicMock()
    perception.capture_snapshot_async.return_value = ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title="Test", class_name="X", rect=(0, 0, 800, 600)),
        ocr_lines=[],
        width=1920,
        height=1080,
    )
    # Cockpit 8: Escalation is a fleet-only mechanism (removed from single mode)
    loop = AgentLoop(
        gemma=gemma,
        dispatcher=dispatcher,
        perception=perception,
        config=config,
        fleet_config=build_cpu_fleet_preset(),
    )
    return loop, gemma


def test_offline_escalation_never_claims_cloud_model(caplog):
    """When running offline, escalation must state it uses the same model with a strategy-change note."""
    plan_json = '{"thought": "click it", "action": "cua_click", "params": {"element_id": 1}}'
    loop, _ = make_loop([plan_json] * 6, config=None)

    with caplog.at_level(logging.WARNING):
        asyncio.run(loop.run(user_goal="test goal"))

    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    escalation_lines = [m for m in warning_msgs if "escalating" in m]

    assert len(escalation_lines) > 0
    for line in escalation_lines:
        assert "gemini" not in line.lower()
        assert "the same model (strategy-change note)" in line


def test_online_escalation_names_configured_model(caplog):
    """When cloud is enabled, escalation names the stronger model."""
    class CloudConfig:
        use_cloud_llm = True
        stronger_planner_model = "gemini-3.1-flash"

    plan_json = '{"thought": "click it", "action": "cua_click", "params": {"element_id": 1}}'
    loop, _ = make_loop([plan_json] * 6, config=CloudConfig())

    with caplog.at_level(logging.WARNING):
        asyncio.run(loop.run(user_goal="test goal"))

    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    escalation_lines = [m for m in warning_msgs if "escalating" in m]

    assert len(escalation_lines) > 0
    assert any("gemini-3.1-flash" in m for m in escalation_lines)


def test_offline_escalation_with_real_config_never_claims_or_passes_cloud_model(caplog):
    """Real Config with use_cloud_llm=False: label must be 'the same model...' and planner call gets model=None."""
    import dataclasses
    from grace.config import Config

    cfg = dataclasses.replace(Config(), use_cloud_llm=False)
    assert cfg.stronger_planner_model == "gemini-3.1-flash"

    plan_json = '{"thought": "click it", "action": "cua_click", "params": {"element_id": 1}}'
    loop, gemma = make_loop([plan_json] * 6, config=cfg)

    assert loop._escalation_label == "the same model (strategy-change note)"
    assert loop._stronger_model is None

    with caplog.at_level(logging.WARNING):
        asyncio.run(loop.run(user_goal="test goal"))

    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    escalation_lines = [m for m in warning_msgs if "escalating" in m]
    assert len(escalation_lines) > 0
    for line in escalation_lines:
        assert "gemini" not in line.lower()
        assert "the same model (strategy-change note)" in line

    planner_calls = gemma.generate_text.call_args_list
    assert len(planner_calls) >= 2
    models = [c.kwargs.get("model") for c in planner_calls]
    assert all(m is None for m in models), f"expected all model calls to be None offline, got: {models}"


def test_online_escalation_with_real_config_names_and_passes_cloud_model(caplog):
    """Real Config with use_cloud_llm=True: label names stronger model and planner call gets model=stronger."""
    import dataclasses
    from grace.config import Config

    cfg = dataclasses.replace(Config(), use_cloud_llm=True, stronger_planner_model="gemini-3.1-flash")

    plan_json = '{"thought": "click it", "action": "cua_click", "params": {"element_id": 1}}'
    loop, gemma = make_loop([plan_json] * 6, config=cfg)

    assert loop._escalation_label == "gemini-3.1-flash"
    assert loop._stronger_model == "gemini-3.1-flash"

    with caplog.at_level(logging.WARNING):
        asyncio.run(loop.run(user_goal="test goal"))

    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    escalation_lines = [m for m in warning_msgs if "escalating" in m]
    assert len(escalation_lines) > 0
    assert any("gemini-3.1-flash" in m for m in escalation_lines)

    planner_calls = gemma.generate_text.call_args_list
    models = [c.kwargs.get("model") for c in planner_calls]
    assert "gemini-3.1-flash" in models, f"expected stronger model in planner calls, got: {models}"

