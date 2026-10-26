"""Cockpit 5 layer 2: send-time terminal cover in the CUA executor.

Right before `cua_type_text`, `cua_set_value` or `cua_press_key` sends input,
the real foreground window is re-checked. If it is a terminal and the call was
not confirmed, nothing is sent and an error fact is returned. Fakes only: no
real desktop, keyboard, mic or screen.
"""

from unittest.mock import MagicMock

import pyautogui
import pytest

from grace.automation.computer_use import ComputerUse

FACT = "The window in front is a terminal (windowsterminal.exe); typing there needs the user's yes."


def _cu(monkeypatch, fg_proc):
    """ComputerUse with the input driver and focus helpers faked out."""
    cu = ComputerUse()
    write = MagicMock()
    press = MagicMock()
    hotkey = MagicMock()
    monkeypatch.setattr(pyautogui, "write", write)
    monkeypatch.setattr(pyautogui, "press", press)
    monkeypatch.setattr(pyautogui, "hotkey", hotkey)
    monkeypatch.setattr("grace.automation.computer_use._terminal_foreground", lambda: fg_proc)
    monkeypatch.setattr("grace.automation.computer_use._foreground_title", lambda: "Untitled - Notepad")
    monkeypatch.setattr(
        "grace.automation.computer_use._wrong_window",
        lambda params, action, initial_title=None: None,
    )
    monkeypatch.setattr("grace.automation.computer_use._invalidate_graph", lambda: None)
    monkeypatch.setattr(cu, "_target_from_params", lambda p: (None, None))
    # Keep the focus/graph probes off the real desktop.
    monkeypatch.setattr("grace.agent.perception.PerceptionEngine.get_graph_builder", lambda *a, **k: None)
    monkeypatch.setattr(cu, "_probe_typed", lambda text, replace: (True, "typed"))
    return cu, write, press, hotkey


def test_type_text_blocked_when_foreground_is_terminal_and_unconfirmed(monkeypatch):
    cu, write, press, hotkey = _cu(monkeypatch, "windowsterminal.exe")
    res = cu._type_text({"text": "rm -r ...", "submit": True}, confirmed=False)
    assert res["sent"] is False
    assert not write.called and not press.called and not hotkey.called
    assert res["error"] == FACT
    assert res["text"] == FACT


def test_type_text_sends_when_confirmed_true(monkeypatch):
    cu, write, press, hotkey = _cu(monkeypatch, "windowsterminal.exe")
    res = cu._type_text({"text": "rm -r ...", "submit": True}, confirmed=True)
    assert res["ok"] is True and res["sent"] is True
    assert write.called
    assert press.called  # submit presses Enter


def test_set_value_blocked_when_foreground_is_terminal_and_unconfirmed(monkeypatch):
    cu, write, press, hotkey = _cu(monkeypatch, "powershell.exe")
    target = MagicMock(center=(100, 100), id=5, name="field")
    monkeypatch.setattr(cu, "_resolve_from_graph", lambda p, e, t: target)
    monkeypatch.setattr(cu, "_click_at", lambda *a, **k: None)
    res = cu._set_value({"element_id": 5, "value": "secret"}, confirmed=False)
    assert res["sent"] is False
    assert not write.called and not press.called and not hotkey.called
    assert res["error"] == "The window in front is a terminal (powershell.exe); typing there needs the user's yes."


def test_set_value_sends_when_confirmed_true(monkeypatch):
    cu, write, press, hotkey = _cu(monkeypatch, "powershell.exe")
    target = MagicMock(center=(100, 100), id=5, name="field")
    monkeypatch.setattr(cu, "_resolve_from_graph", lambda p, e, t: target)
    monkeypatch.setattr(cu, "_click_at", lambda *a, **k: None)
    res = cu._set_value({"element_id": 5, "value": "secret"}, confirmed=True)
    assert res["ok"] is True and res["sent"] is True
    assert write.called or hotkey.called


def test_press_key_enter_blocked_but_down_passes(monkeypatch):
    cu = ComputerUse()
    hotkey = MagicMock()
    press = MagicMock()
    monkeypatch.setattr(pyautogui, "hotkey", hotkey)
    monkeypatch.setattr(pyautogui, "press", press)
    monkeypatch.setattr("grace.automation.computer_use._terminal_foreground", lambda: "cmd.exe")
    monkeypatch.setattr("grace.automation.computer_use._foreground_title", lambda: "Notepad")
    monkeypatch.setattr(
        "grace.automation.computer_use._wrong_window",
        lambda params, action, initial_title=None: None,
    )
    monkeypatch.setattr("grace.automation.computer_use._screen_state", lambda: None)
    monkeypatch.setattr("grace.automation.computer_use._settled_state", lambda: None)
    monkeypatch.setattr("grace.automation.computer_use._invalidate_graph", lambda: None)

    res_enter = cu._press_key({"key": "Enter"}, confirmed=False)
    assert res_enter["sent"] is False
    assert not hotkey.called and not press.called
    assert res_enter["error"] == "The window in front is a terminal (cmd.exe); typing there needs the user's yes."

    hotkey.reset_mock()
    press.reset_mock()
    res_down = cu._press_key({"key": "Down"}, confirmed=False)
    assert res_down["ok"] is True and res_down["sent"] is True
    assert hotkey.called or press.called

    # Confirmed Enter goes through.
    hotkey.reset_mock()
    press.reset_mock()
    res_enter_confirmed = cu._press_key({"key": "Enter"}, confirmed=True)
    assert res_enter_confirmed["ok"] is True and res_enter_confirmed["sent"] is True
    assert hotkey.called or press.called
