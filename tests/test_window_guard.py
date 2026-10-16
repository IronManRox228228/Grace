"""Tests for window guard focus checks and race condition prevention (R11, R12, R22).

Before typing or clicking (and right before the physical dispatch in _click_at),
Grace re-checks that the target window is still the foreground window, and aborts
the action if focus changed.
"""

from unittest import mock
import pytest

from grace.automation.computer_use import ComputerUse, _wrong_window


class FakeAutoGui:
    def __init__(self):
        self.calls = []

    def write(self, text, interval=None):
        self.calls.append(("write", text))

    def hotkey(self, *keys):
        self.calls.append(("hotkey", *keys))

    def press(self, key):
        self.calls.append(("press", key))

    def click(self, *a, **k):
        self.calls.append(("click", a, k))


class TestWindowGuard:
    def test_wrong_window_refuses_when_named_window_differs_from_foreground(self, monkeypatch):
        monkeypatch.setattr("grace.automation.computer_use._foreground_title", lambda: "Calculator")

        refused = _wrong_window({"window": {"title": "Notepad"}}, "click")
        assert refused is not None
        assert refused["ok"] is False
        assert refused["status"] == "wrong_window"
        assert "Notepad" in refused["error"]
        assert "Calculator" in refused["error"]

    def test_wrong_window_detects_focus_shift_from_initial_window(self, monkeypatch):
        monkeypatch.setattr("grace.automation.computer_use._foreground_title", lambda: "Browser")

        refused = _wrong_window({}, "type_text", initial_title="Word")
        assert refused is not None
        assert refused["ok"] is False
        assert refused["status"] == "wrong_window"
        assert "Word" in refused["error"]
        assert "Browser" in refused["error"]

    def test_click_at_race_aborts_if_foreground_window_changes_before_click(self, monkeypatch):
        """R22: _click_at re-checks immediately prior to the physical click."""
        fake_gui = FakeAutoGui()
        monkeypatch.setattr("pyautogui.click", fake_gui.click)

        # Foreground window is "Notepad" initially, but changes to "Malicious Pop-up" right before click
        monkeypatch.setattr("grace.automation.computer_use._foreground_title", lambda: "Malicious Pop-up")

        cu = ComputerUse()
        res = cu._click_at(
            x=100, y=100, click_count=1, method="test",
            params={"window": {"title": "Notepad"}},
            initial_fg="Notepad"
        )

        assert res["ok"] is False
        assert res["status"] == "wrong_window"
        assert len(fake_gui.calls) == 0  # No click was dispatched

    def test_type_text_aborts_if_foreground_window_changes_before_typing(self, monkeypatch):
        """R11/R12: _type_text re-checks immediately prior to keystrokes."""
        fake_gui = FakeAutoGui()
        monkeypatch.setattr("pyautogui.write", fake_gui.write)
        monkeypatch.setattr("pyautogui.hotkey", fake_gui.hotkey)
        monkeypatch.setattr("pyautogui.press", fake_gui.press)

        titles = ["Notepad", "Command Prompt"]  # Changes focus mid-action

        def shifting_title():
            return titles.pop(0) if titles else "Command Prompt"

        monkeypatch.setattr("grace.automation.computer_use._foreground_title", shifting_title)

        cu = ComputerUse()
        res = cu._type_text({"text": "confidential text", "window": {"title": "Notepad"}})

        assert res["ok"] is False
        assert res["status"] == "wrong_window"
        assert len(fake_gui.calls) == 0  # No keystrokes were dispatched

    def test_press_key_aborts_if_foreground_window_changes_before_key(self, monkeypatch):
        fake_gui = FakeAutoGui()
        monkeypatch.setattr("pyautogui.hotkey", fake_gui.hotkey)
        monkeypatch.setattr("pyautogui.press", fake_gui.press)

        titles = ["Word", "Terminal"]

        def shifting_title():
            return titles.pop(0) if titles else "Terminal"

        monkeypatch.setattr("grace.automation.computer_use._foreground_title", shifting_title)

        cu = ComputerUse()
        res = cu._press_key({"key": "enter", "window": {"title": "Word"}})

        assert res["ok"] is False
        assert res["status"] == "wrong_window"
        assert len(fake_gui.calls) == 0  # No key was dispatched
