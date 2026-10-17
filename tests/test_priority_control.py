"""Tests for process priority and EcoQoS control (PLAN.md §4, Task 08)."""

import pytest
from unittest.mock import patch

from grace.priority import (
    set_turn_priority,
    set_idle_priority,
    set_background_task_priority,
    turn_priority,
    set_priority_class,
    set_eco_qos,
)


def test_priority_and_eco_qos_calls_succeed_or_fallback():
    # Verify calls do not raise exceptions on Windows
    assert isinstance(set_priority_class(0x00000020), bool)
    assert isinstance(set_eco_qos(True), bool)
    assert isinstance(set_eco_qos(False), bool)


def test_turn_priority_and_idle_priority():
    set_turn_priority()
    set_idle_priority()
    set_background_task_priority()
    set_idle_priority()


def test_turn_priority_context_manager():
    with patch("grace.priority.set_turn_priority") as mock_turn:
        with patch("grace.priority.set_idle_priority") as mock_idle:
            with turn_priority():
                mock_turn.assert_called_once()
                mock_idle.assert_not_called()
            mock_idle.assert_called_once()
