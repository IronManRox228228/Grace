"""Tests for window guard focus checks and race condition prevention (R11, R12, R22).

Before typing or clicking (and right before the physical dispatch in _click_at),
Grace re-checks that the target window is still the foreground window, and aborts
the action if focus changed.
"""

from unittest import mock
import pytest

from grace.automation.computer_use import ComputerUse, _wrong_window


@pytest.fixture(autouse=True)
def _no_terminal_in_front(monkeypatch):
    # The send-time terminal check reads the real foreground window; a test run
    # started from a terminal would otherwise refuse every keystroke.
    import grace.automation.computer_use as cu
    monkeypatch.setattr(cu, "_terminal_foreground", lambda: None)


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
        assert refused["status"] == "window_not_found"
        assert "Notepad" in refused["error"]
        assert "Calculator" in refused["error"]

    def test_wrong_window_detects_focus_shift_from_initial_window(self, monkeypatch):
        monkeypatch.setattr("grace.automation.computer_use._foreground_title", lambda: "Browser")

        refused = _wrong_window({}, "type_text", initial_title="Word")
        assert refused is not None
        assert refused["ok"] is False
        assert refused["status"] == "window_not_found"
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
        assert res["status"] == "window_not_found"
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
        assert res["status"] == "window_not_found"
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
        assert res["status"] == "window_not_found"
        assert len(fake_gui.calls) == 0  # No key was dispatched


class TestWindowTitleNormalization:
    """Zero-width characters in window titles (real-desktop.md #9).

    Edge Beta embeds a U+200B in its title ("Microsoft​ Edge"), so raw string
    comparison made Grace refuse the right window. Titles are normalised
    everywhere they're compared or shown to the planner.
    """

    ZERO_WIDTH_TITLE = "New tab - Personal - Microsoft\u200b Edge Beta"
    PLAIN_TITLE = "New tab - Personal - Microsoft Edge Beta"

    def test_normalize_title_strips_zero_width_and_collapses_whitespace(self):
        from grace.automation.computer_use import normalize_title

        assert normalize_title(self.ZERO_WIDTH_TITLE) == self.PLAIN_TITLE
        # Each zero-width / format char is stripped (not replaced by a space).
        assert normalize_title("a\u200bb") == "ab"      # U+200B
        assert normalize_title("a\u200cb") == "ab"      # U+200C
        assert normalize_title("a\u200db") == "ab"      # U+200D
        assert normalize_title("a\u2060b") == "ab"      # U+2060
        assert normalize_title("\ufeffab") == "ab"      # U+FEFF
        # NFKC + whitespace collapse.
        assert normalize_title("  Microsoft   Edge  ") == "Microsoft Edge"
        assert normalize_title("") == ""

    def test_foreground_title_is_normalised(self, monkeypatch):
        import win32gui

        monkeypatch.setattr(win32gui, "GetForegroundWindow", lambda: 1234)
        monkeypatch.setattr(win32gui, "GetWindowText", lambda hwnd: self.ZERO_WIDTH_TITLE)
        from grace.automation.computer_use import _foreground_title

        assert _foreground_title() == self.PLAIN_TITLE

    def test_wrong_window_matches_despite_zero_width_char_in_foreground(self, monkeypatch):
        """The planner names the window without the U+200B: must not refuse."""
        monkeypatch.setattr(
            "grace.automation.computer_use._foreground_title",
            lambda: self.ZERO_WIDTH_TITLE,
        )

        refused = _wrong_window({"window": {"title": self.PLAIN_TITLE}}, "click")
        assert refused is None

    def test_wrong_window_still_refuses_genuinely_different_title(self, monkeypatch):
        monkeypatch.setattr("grace.automation.computer_use._foreground_title", lambda: "Calculator")

        refused = _wrong_window({"window": {"title": self.PLAIN_TITLE}}, "click")
        assert refused is not None
        assert refused["ok"] is False
        assert refused["status"] == "window_not_found"
        assert self.PLAIN_TITLE in refused["error"]
        assert "Calculator" in refused["error"]
