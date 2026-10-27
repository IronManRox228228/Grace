"""Tests for loop breaker catching repeating actions with target_name variations or coordinate wobble (real-desktop.md #5)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from grace.agent.loop import AgentLoop
from grace.agent.perception import ScreenSnapshot, WindowInfo


def make_loop_with_frozen_screen(plans):
    gemma = AsyncMock()
    gemma.generate_text.side_effect = plans
    dispatcher = AsyncMock()
    dispatcher.execute.return_value = {"status": "ok"}
    perception = MagicMock()
    # Screen never changes (frozen screen)
    fixed_snapshot = ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title="YouTube Music", class_name="Chrome", rect=(0, 0, 800, 600)),
        ocr_lines=[],
        width=1920,
        height=1080,
        ui_elements=[],
    )
    perception.capture_snapshot.return_value = fixed_snapshot
    perception.capture_snapshot_async = AsyncMock(return_value=fixed_snapshot)
    from grace.fleet import build_cpu_fleet_preset
    # Cockpit 8: loop breaker / escalation ladder stops belong to fleet mode
    loop = AgentLoop(
        gemma=gemma,
        dispatcher=dispatcher,
        perception=perception,
        fleet_config=build_cpu_fleet_preset(),
    )
    return loop


def test_loop_breaker_stops_on_different_target_names_for_same_element():
    """Holo clicked element 1 with slightly different target_names and thoughts; loop breaker must stop."""
    plans = [
        '{"thought": "click address bar", "action": "cua_click", "params": {"element_id": 1, "target_name": "the address bar"}}',
        '{"thought": "try search bar", "action": "cua_click", "params": {"element_id": 1, "target_name": "address and search bar"}}',
        '{"thought": "click navigation input", "action": "cua_click", "params": {"element_id": 1, "target_name": "omnibox"}}',
        '{"thought": "click again", "action": "cua_click", "params": {"element_id": 1, "target_name": "URL field"}}',
        '{"thought": "click again 5", "action": "cua_click", "params": {"element_id": 1, "target_name": "URL field 5"}}',
        '{"thought": "click again 6", "action": "cua_click", "params": {"element_id": 1, "target_name": "URL field 6"}}',
    ]
    loop = make_loop_with_frozen_screen(plans)
    res = asyncio.run(loop.run(user_goal="play music"))
    assert res["status"] == "no_progress"


def test_loop_breaker_stops_on_coordinate_wobble_within_12px():
    """Clicks within 12px of previous click on unchanged screen count as identical; stops after 3 attempts."""
    plans = [
        '{"thought": "click point", "action": "cua_click", "params": {"x": 500, "y": 60}}',
        '{"thought": "click point slightly shifted", "action": "cua_click", "params": {"x": 505, "y": 63}}',
        '{"thought": "click point shifted other way", "action": "cua_click", "params": {"x": 497, "y": 58}}',
        '{"thought": "click point again", "action": "cua_click", "params": {"x": 502, "y": 61}}',
        '{"thought": "click point 5", "action": "cua_click", "params": {"x": 501, "y": 60}}',
        '{"thought": "click point 6", "action": "cua_click", "params": {"x": 500, "y": 59}}',
    ]
    loop = make_loop_with_frozen_screen(plans)
    res = asyncio.run(loop.run(user_goal="play music"))
    assert res["status"] == "no_progress"
