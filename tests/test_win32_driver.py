"""Unit tests for Win32Driver background message dispatcher and Hermes Agent integration."""

import sys
import os
import time
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.automation.win32_driver import Win32Driver, MOUSEEVENTF_MOVE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.automation.ui_inspector import UIElement


class TestWin32Driver:
    """Test suite for Win32Driver background message dispatcher."""

    @patch("win32gui.GetForegroundWindow", return_value=0x1234)
    @patch("win32gui.IsWindowVisible", return_value=True)
    @patch("win32gui.WindowFromPoint", return_value=0x1234)
    @patch("ctypes.windll.user32.SetCursorPos", return_value=1)
    @patch("ctypes.windll.user32.SendInput", return_value=3)
    def test_win32_driver_click_at_does_not_raise(
        self, mock_send_input, mock_set_cursor, mock_from_point, mock_visible, mock_fg
    ):
        # WindowFromPoint must not read the real desktop: a fake hwnd that
        # GetForegroundWindow matches keeps the focus step hermetic (the test
        # passed or failed depending on whichever real window sat at (100, 100)).
        res = Win32Driver.click_at(100, 100)
        assert res is True
        assert mock_send_input.called
        assert mock_set_cursor.called

    def test_perception_to_markdown_includes_hermes_tree_index(self):
        elem = UIElement(index=14, name="Play", control_type="Button", bounds=(700, 245, 780, 285), center=(740, 265))
        snapshot = ScreenSnapshot(
            active_window=WindowInfo(hwnd=123, title="YouTube Music - Microsoft Edge", class_name="Chrome_WidgetWin_1", rect=(0,0,1000,800)),
            ocr_lines=[],
            width=1920,
            height=1080,
            ui_elements=[elem],
        )

        md = snapshot.to_markdown()
        assert "[14] Button: \"Play\"" in md
        # The window hint is now appended: cua_click(element_index=14, window="...")
        assert "cua_click(element_index=14" in md

    def test_click_at_returns_true_without_win32gui(self):
        """Bug #1 fix: missing pywin32 must not flip click_at to False."""
        import builtins
        real_import = builtins.__import__

        def _block(name, *a, **k):
            if name == "win32gui":
                raise ImportError("simulated missing pywin32")
            return real_import(name, *a, **k)

        with patch.object(builtins, "__import__", side_effect=_block):
            with patch("ctypes.windll.user32.SetCursorPos", return_value=1):
                with patch.object(Win32Driver, "send_input_click") as fake_send:
                    assert Win32Driver.click_at(250, 250) is True
                    fake_send.assert_called_once_with(250, 250)

    def test_click_at_moves_cursor_before_sendinput(self):
        """Bug #3 fix: SetCursorPos must run before SendInput so hover fires."""
        with patch("ctypes.windll") as windll:
            user32 = MagicMock()
            windll.user32 = user32
            with patch.object(Win32Driver, "send_input_click") as fake_send:
                # Block win32gui import path -> focus step skipped, cursor still moves
                import builtins
                real_import = builtins.__import__

                def _block(name, *a, **k):
                    if name == "win32gui":
                        raise ImportError("no pywin32")
                    return real_import(name, *a, **k)

                with patch.object(builtins, "__import__", side_effect=_block):
                    Win32Driver.click_at(320, 480)

            # SetCursorPos called with the click coordinates
            assert user32.SetCursorPos.called
            args = user32.SetCursorPos.call_args[0]
            assert args[0] == 320 and args[1] == 480
            # send_input_click runs after cursor positioning
            fake_send.assert_called_once_with(320, 480)

    def test_ensure_foreground_retries_on_mismatch(self):
        """Bug #1 fix: focus is verified and retried once via SetForegroundWindow."""
        from unittest.mock import MagicMock
        gui = MagicMock()
        con = MagicMock()
        gui.IsWindowVisible.return_value = True
        gui.GetForegroundWindow.side_effect = [0, 99]  # first not active, then active
        gui.IsWindow.return_value = True

        with patch.object(Win32Driver, "_send_alt_tap"), patch.dict(
            sys.modules, {"win32gui": gui, "win32con": con}
        ):
            assert Win32Driver._ensure_foreground(99) is True
            assert gui.SetForegroundWindow.call_count >= 1

    def test_ensure_foreground_uses_attach_thread_input_and_allow_set_foreground_window(self):
        """Item 3 fix: Uses AttachThreadInput and AllowSetForegroundWindow to focus terminal-launched targets."""
        gui = MagicMock()
        con = MagicMock()
        gui.GetForegroundWindow.side_effect = [10, 99]
        gui.GetAncestor.return_value = None

        with patch("ctypes.windll") as windll, \
             patch.object(Win32Driver, "_send_alt_tap"), \
             patch.dict(sys.modules, {"win32gui": gui, "win32con": con}):
            user32 = MagicMock()
            kernel32 = MagicMock()
            windll.user32 = user32
            windll.kernel32 = kernel32

            kernel32.GetCurrentThreadId.return_value = 1001
            user32.GetWindowThreadProcessId.return_value = 2002
            user32.AttachThreadInput.return_value = 1

            assert Win32Driver._ensure_foreground(99) is True

            # Verified AttachThreadInput attached then detached
            user32.AttachThreadInput.assert_any_call(1001, 2002, True)
            user32.AttachThreadInput.assert_any_call(1001, 2002, False)
            user32.AllowSetForegroundWindow.assert_called_with(-1)

    def test_ensure_foreground_resolves_child_window_to_root(self):
        """Child render window (e.g. Chrome/Edge widget) resolves to root top-level window."""
        gui = MagicMock()
        con = MagicMock()
        # 555 is child window, 99 is root window
        gui.GetAncestor.side_effect = lambda hwnd, flag: 99 if hwnd == 555 else None
        gui.GetForegroundWindow.return_value = 99  # Already foreground on root

        with patch.dict(sys.modules, {"win32gui": gui, "win32con": con}):
            assert Win32Driver._ensure_foreground(555) is True


class TestDragPath:
    """The drag must be a real human-style drag: down, small held moves, up.

    Nothing here touches the real desktop: SetCursorPos and SendInput are
    faked, and the generated input sequence is asserted instead.
    """

    def _block_win32gui(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _block(name, *a, **k):
            if name == "win32gui":
                raise ImportError("drag tests do not load pywin32")
            return real_import(name, *a, **k)

        return patch.object(builtins, "__import__", side_effect=_block)

    def test_drag_sends_down_then_small_moves_then_up(self, monkeypatch):
        sent = []
        sleeps = []

        def fake_send_mouse(x, y, flag):
            sent.append((int(x), int(y), int(flag)))
            return True

        monkeypatch.setattr(Win32Driver, "_send_mouse", staticmethod(fake_send_mouse))
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(float(s)))

        points = [(100, 100), (230, 100), (230, 200)]
        with patch("ctypes.windll") as windll, self._block_win32gui(monkeypatch):
            windll.user32 = MagicMock()
            assert Win32Driver.drag_path(points) is True

        # Exactly one press and one release.
        downs = [s for s in sent if s[2] & MOUSEEVENTF_LEFTDOWN]
        ups = [s for s in sent if s[2] & MOUSEEVENTF_LEFTUP]
        assert len(downs) == 1 and len(ups) == 1
        assert downs[0] == (100, 100, MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTDOWN)
        assert ups[-1] == (230, 200, MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTUP)

        # The first move sits on the start point with the button held.
        assert sent[1] == (100, 100, MOUSEEVENTF_MOVE)

        # Every intermediate event is a plain move (button held from the down).
        moves = [s for s in sent[1:-1] if not (s[2] & MOUSEEVENTF_LEFTDOWN or s[2] & MOUSEEVENTF_LEFTUP)]
        assert all(flag == MOUSEEVENTF_MOVE for _, _, flag in moves)

        # Steps are at most 10 px apart, including across segment boundaries,
        # and the last move ends on the final point.
        move_points = [(x, y) for x, y, _ in sent[1:-1]]
        assert move_points[-1] == (230, 200)
        for (x1, y1), (x2, y2) in zip(move_points, move_points[1:]):
            assert max(abs(x2 - x1), abs(y2 - y1)) <= 10

        # The held pauses: 10 ms cursor settle, ~50 ms after the button-down,
        # ~6 ms per step, ~50 ms before the release.
        assert sleeps[0] == 10 / 1000.0
        assert sleeps[1] == 50 / 1000.0
        assert sleeps[-1] == 50 / 1000.0
        assert all(s == 6 / 1000.0 for s in sleeps[2:-1])

    def test_drag_short_segment_is_one_move(self, monkeypatch):
        sent = []
        monkeypatch.setattr(Win32Driver, "_send_mouse", staticmethod(lambda x, y, f: sent.append((x, y, f)) or True))
        monkeypatch.setattr(time, "sleep", lambda s: None)

        with patch("ctypes.windll") as windll, self._block_win32gui(monkeypatch):
            windll.user32 = MagicMock()
            assert Win32Driver.drag_path([(0, 0), (5, 5)]) is True

        assert sent == [
            (0, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTDOWN),
            (0, 0, MOUSEEVENTF_MOVE),
            (5, 5, MOUSEEVENTF_MOVE),
            (5, 5, MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTUP),
        ]

    def test_drag_returns_false_when_the_button_down_fails(self, monkeypatch):
        monkeypatch.setattr(Win32Driver, "_send_mouse", staticmethod(lambda x, y, f: False))
        monkeypatch.setattr(time, "sleep", lambda s: None)

        with patch("ctypes.windll") as windll, self._block_win32gui(monkeypatch):
            windll.user32 = MagicMock()
            assert Win32Driver.drag_path([(0, 0), (50, 50)]) is False
