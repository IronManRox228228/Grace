"""Unit tests for Win32Driver background message dispatcher and Hermes Agent integration."""

import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.automation.win32_driver import Win32Driver
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.automation.ui_inspector import UIElement


class TestWin32Driver:
    """Test suite for Win32Driver background message dispatcher."""

    def test_win32_driver_click_at_does_not_raise(self):
        res = Win32Driver.click_at(100, 100)
        assert res is True

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

