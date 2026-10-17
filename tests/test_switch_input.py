"""Unit tests for switch accessibility input, race resolution, and mark scanning (Task 10, PLAN §7)."""

import time
from unittest.mock import MagicMock
import pytest

from grace.input.switch_input import (
    MarkScanner,
    SwitchAction,
    SwitchInputHandler,
    VoiceSwitchRaceResolver,
)


class TestSwitchInputHandler:
    def test_single_press_fires_yes_action(self):
        actions = []
        handler = SwitchInputHandler(
            double_press_window_ms=50.0,
            long_press_threshold_ms=200.0,
            on_action=lambda a: actions.append(a),
        )

        t0 = 1000.0
        handler.press_down(timestamp=t0)
        res = handler.press_up(timestamp=t0 + 0.02)  # 20ms tap
        assert res is None  # Waiting for possible second tap

        time.sleep(0.08)  # Wait for single press timer to expire
        assert actions == [SwitchAction.YES]

    def test_double_press_fires_no_action(self):
        actions = []
        handler = SwitchInputHandler(
            double_press_window_ms=100.0,
            long_press_threshold_ms=300.0,
            on_action=lambda a: actions.append(a),
        )

        t0 = 1000.0
        handler.press_down(timestamp=t0)
        handler.press_up(timestamp=t0 + 0.02)

        # Second press within 50ms
        handler.press_down(timestamp=t0 + 0.05)
        res = handler.press_up(timestamp=t0 + 0.07)

        assert res == SwitchAction.NO
        assert actions == [SwitchAction.NO]

    def test_long_press_fires_stop_action_synchronously(self):
        actions = []
        handler = SwitchInputHandler(
            double_press_window_ms=50.0,
            long_press_threshold_ms=100.0,
            on_action=lambda a: actions.append(a),
        )

        t0 = 1000.0
        handler.press_down(timestamp=t0)
        res = handler.press_up(timestamp=t0 + 0.15)  # 150ms press >= 100ms threshold

        assert res == SwitchAction.STOP
        assert SwitchAction.STOP in actions

    def test_key_handling_with_configured_switch_key(self):
        actions = []
        handler = SwitchInputHandler(
            switch_key="f12",
            double_press_window_ms=50.0,
            long_press_threshold_ms=100.0,
            on_action=lambda a: actions.append(a),
        )

        # Unrelated key is ignored
        assert handler.handle_key("a", is_down=True) is None

        # Configured switch key
        t0 = 1000.0
        handler.handle_key("F12", is_down=True, timestamp=t0)
        res = handler.handle_key("F12", is_down=False, timestamp=t0 + 0.15)
        assert res == SwitchAction.STOP


class TestVoiceSwitchRaceResolver:
    def test_voice_wins_first(self):
        resolver = VoiceSwitchRaceResolver()
        assert not resolver.is_resolved

        # Voice arrives first
        won = resolver.resolve_voice("yes")
        assert won is True
        assert resolver.is_resolved
        assert resolver.winner == "voice"
        assert resolver.result == "yes"

        # Subsequent switch press is rejected
        switch_won = resolver.resolve_switch(SwitchAction.NO)
        assert switch_won is False
        assert resolver.winner == "voice"

    def test_switch_wins_first(self):
        resolver = VoiceSwitchRaceResolver()

        # Switch arrives first
        won = resolver.resolve_switch(SwitchAction.YES)
        assert won is True
        assert resolver.is_resolved
        assert resolver.winner == "switch"
        assert resolver.result == "yes"

        # Subsequent voice answer is rejected
        voice_won = resolver.resolve_voice("no")
        assert voice_won is False
        assert resolver.winner == "switch"


class TestMarkScanner:
    def test_mark_scanning_and_selection(self):
        highlights = []
        scanner = MarkScanner(
            options=["button 4", "button 7"],
            scan_rate_ms=20.0,
            on_highlight=lambda idx, opt: highlights.append((idx, opt)),
        )

        scanner.start()
        time.sleep(0.06)

        assert len(highlights) > 0
        assert scanner.current_option in ["button 4", "button 7"]

        # User presses switch to select
        selected = scanner.select_current()
        assert selected in ["button 4", "button 7"]
        assert not scanner._running
