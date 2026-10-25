"""Diagnostic script to test window activation techniques from a background shell.

Usage:
    python scripts/diag_activate.py <title-substring>

Do not run in automated test suites: it drives the real desktop.
"""

import ctypes
import sys
import time

user32 = ctypes.windll.user32
VK_MENU = 0x12
KEYEVENTF_KEYUP = 0x0002
SW_MINIMIZE = 6
SW_RESTORE = 9


def find_target_window(query: str) -> tuple[int, str]:
    import win32gui

    matches = []

    def enum_cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd) or ""
            if query.lower() in title.lower():
                matches.append((hwnd, title))

    win32gui.EnumWindows(enum_cb, None)
    if not matches:
        return 0, ""
    return matches[0]


def verify_active(target_hwnd: int, delay_s: float = 0.3) -> tuple[bool, int, str]:
    import win32gui

    time.sleep(delay_s)
    # Re-poll transient 0 for up to 500ms
    start = time.monotonic()
    fg = 0
    while time.monotonic() - start < 0.5:
        fg = win32gui.GetForegroundWindow()
        if fg == target_hwnd:
            break
        if fg != 0:
            break
        time.sleep(0.05)
    title = win32gui.GetWindowText(fg) if fg else ""
    return (fg == target_hwnd), fg, title


def try_technique_1(hwnd: int) -> None:
    """1. plain SetForegroundWindow"""
    import win32gui

    win32gui.SetForegroundWindow(hwnd)


def try_technique_2(hwnd: int) -> None:
    """2. AttachThreadInput to the foreground thread, then SetForegroundWindow + BringWindowToTop"""
    import win32api
    import win32gui
    import win32process

    fg_hwnd = win32gui.GetForegroundWindow()
    fg_thread = 0
    if fg_hwnd:
        fg_thread, _ = win32process.GetWindowThreadProcessId(fg_hwnd)
    cur_thread = win32api.GetCurrentThreadId()

    attached = False
    if fg_thread and fg_thread != cur_thread:
        try:
            attached = bool(user32.AttachThreadInput(cur_thread, fg_thread, True))
        except Exception:
            attached = False

    try:
        win32gui.SetForegroundWindow(hwnd)
        win32gui.BringWindowToTop(hwnd)
    finally:
        if attached:
            try:
                user32.AttachThreadInput(cur_thread, fg_thread, False)
            except Exception:
                pass


def try_technique_3(hwnd: int) -> None:
    """3. synthetic Alt down/up (VK_MENU), then SetForegroundWindow"""
    import win32gui

    user32.keybd_event(VK_MENU, 0, 0, 0)
    user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
    win32gui.SetForegroundWindow(hwnd)


def try_technique_4(hwnd: int) -> None:
    """4. ShowWindow(SW_MINIMIZE) then SW_RESTORE"""
    import win32gui

    win32gui.ShowWindow(hwnd, SW_MINIMIZE)
    time.sleep(0.05)
    win32gui.ShowWindow(hwnd, SW_RESTORE)


def try_technique_5(hwnd: int) -> None:
    """5. SwitchToThisWindow(hwnd, True)"""
    user32.SwitchToThisWindow(hwnd, True)


def try_technique_6(hwnd: int) -> None:
    """6. UIA SetFocus on the window element"""
    try:
        from grace.perception.uia_provider import _get_uia

        provider = _get_uia()
        if provider and provider.uia:
            elem = provider.uia.ElementFromHandle(hwnd)
            if elem:
                elem.SetFocus()
    except Exception as e:
        print(f"  [UIA error: {e}]")


TECHNIQUES = [
    ("1. plain SetForegroundWindow", try_technique_1),
    ("2. AttachThreadInput + SetForegroundWindow + BringWindowToTop", try_technique_2),
    ("3. synthetic Alt down/up + SetForegroundWindow", try_technique_3),
    ("4. ShowWindow(SW_MINIMIZE) -> SW_RESTORE", try_technique_4),
    ("5. SwitchToThisWindow(hwnd, True)", try_technique_5),
    ("6. UIA SetFocus on window element", try_technique_6),
]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("Usage: python scripts/diag_activate.py <title-substring>")
        sys.exit(1)

    query = sys.argv[1].strip()
    target_hwnd, target_title = find_target_window(query)

    if not target_hwnd:
        print(f"Error: No visible window matching '{query}' found.")
        sys.exit(1)

    print("=" * 80)
    print(f"Window Activation Diagnostic for target:")
    print(f"  HWND:  {target_hwnd} (0x{target_hwnd:X})")
    print(f"  Title: '{target_title}'")
    print("=" * 80)
    print(f"{'#':<3} | {'Technique':<50} | {'Success':<8} | {'FG HWND':<10} | {'Foreground Title'}")
    print("-" * 105)

    import win32gui
    orig_fg = 0
    try:
        orig_fg = win32gui.GetForegroundWindow()
    except Exception:
        pass

    for idx, (name, fn) in enumerate(TECHNIQUES, start=1):
        if orig_fg and win32gui.IsWindow(orig_fg):
            try:
                user32.SwitchToThisWindow(orig_fg, True)
                time.sleep(0.2)
            except Exception:
                pass

        try:
            fn(target_hwnd)
        except Exception as e:
            print(f"{idx:<3} | {name:<50} | ERROR    | {'-':<10} | Raised: {e}")
            continue

        ok, fg_hwnd, fg_title = verify_active(target_hwnd, delay_s=0.3)
        res_str = "YES" if ok else "NO"
        title_snip = (fg_title[:30] + "...") if len(fg_title) > 30 else fg_title
        print(f"{idx:<3} | {name:<50} | {res_str:<8} | {fg_hwnd:<10} | {title_snip}")

    print("=" * 80)


if __name__ == "__main__":
    main()
