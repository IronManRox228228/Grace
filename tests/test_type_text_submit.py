"""Tests for cua_type_text submit option (cockpit-2 Commit 5).

- Tool schema has submit parameter (bool, default false).
- _type_text executes type, then Return/Enter, in the same window.
- Reports 'Typed N characters into [...] and pressed Enter'.
"""

from unittest.mock import MagicMock, patch
import pytest

from grace.automation.computer_use import ComputerUse
from grace.intent.tools import ALL_TOOLS


def test_cua_type_text_has_submit_param():
    tool = next(t for t in ALL_TOOLS if t.name == "cua_type_text")
    submit_param = next((p for p in tool.params if p.name == "submit"), None)
    assert submit_param is not None
    assert submit_param.param_type == "bool"
    assert submit_param.required is False
    assert submit_param.default == "false"
    assert "Enter" in submit_param.description


def test_type_text_with_submit_types_and_presses_enter(monkeypatch):
    cu = ComputerUse()
    press_mock = MagicMock()
    write_mock = MagicMock()

    import pyautogui
    monkeypatch.setattr(pyautogui, "press", press_mock)
    monkeypatch.setattr(pyautogui, "write", write_mock)
    monkeypatch.setattr(cu, "_target_from_params", lambda p: (None, None))
    monkeypatch.setattr(cu, "_probe_typed", lambda text, replace: (True, "typed"))
    # Mock window checks
    monkeypatch.setattr("grace.automation.computer_use._foreground_title", lambda: "Notepad")
    monkeypatch.setattr("grace.automation.computer_use._wrong_window", lambda params, action, initial_title=None: None)
    monkeypatch.setattr("grace.automation.computer_use._invalidate_graph", lambda: None)

    # 1. submit = True
    params_submit = {"text": "hello world", "submit": True}
    res = cu._type_text(params_submit)
    assert res["ok"] is True
    assert press_mock.called
    assert "enter" in [call.args[0] for call in press_mock.call_args_list]
    assert "and pressed Enter" in res["message"]
    assert "Typed 11 characters" in res["message"]

    # 2. submit = False (default)
    press_mock.reset_mock()
    params_no_submit = {"text": "hello world", "submit": False}
    res_no = cu._type_text(params_no_submit)
    assert res_no["ok"] is True
    assert not any(call.args[0] == "enter" for call in press_mock.call_args_list)
    assert "and pressed Enter" not in res_no["message"]
    assert "Typed 11 characters" in res_no["message"]
