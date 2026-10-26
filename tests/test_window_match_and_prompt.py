import pytest
from unittest.mock import MagicMock
from grace.automation.computer_use import pick_best_window_match, ComputerUse
from grace.agent.planner import get_planner_system_prompt


def test_window_match_prefers_all_tokens_and_closest_length():
    wanted = "Home and 1 more page - Personal - Microsoft\u200b Edge Beta"
    candidates = [
        (1, "Home"),
        (2, "Home and 1 more page - Personal - Microsoft\u200b Edge Beta"),
    ]

    best = pick_best_window_match(wanted, candidates)
    assert best is not None
    assert best[0] == 2
    assert "Microsoft" in best[1]

    # When wanted is short, it should prefer the exact short match rather than the longer one
    wanted_short = "Home"
    best_short = pick_best_window_match(wanted_short, candidates)
    assert best_short is not None
    assert best_short[0] == 1
    assert best_short[1] == "Home"


def test_activate_window_picks_correct_best_match():
    cu = ComputerUse()
    cu._list_windows = MagicMock(return_value={
        "windows": [
            {"id": 1, "title": "Home"},
            {"id": 2, "title": "Home and 1 more page - Personal - Microsoft\u200b Edge Beta"},
        ]
    })
    cu._activate_hwnd = MagicMock(return_value={"ok": True, "status": "ok", "action": "activate"})

    res = cu.perform("activate", {"window_title": "Home and 1 more page - Personal - Microsoft\u200b Edge Beta"})
    assert cu._activate_hwnd.call_count == 1
    call_args = cu._activate_hwnd.call_args[0]
    # Should pick id 2, not id 1
    assert call_args[0] == 2


def test_single_mode_prompt_says_whole_screen():
    prompt = get_planner_system_prompt(is_single=True)
    assert "current screenshot of the whole screen" in prompt
    assert "current screenshot of the active window" not in prompt


def test_single_mode_prompt_says_xy_beats_element_id():
    prompt = get_planner_system_prompt(is_single=True)
    assert "`x`/`y` takes precedence" in prompt
    assert "the element id takes precedence" not in prompt
    assert "takes precedence" not in get_planner_system_prompt(is_single=False)
