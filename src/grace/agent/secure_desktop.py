"""Windows security screens and secure desktop detection.

Grace hands off Windows security screens (UAC prompts, lock screen, sign-in)
to Windows Voice Access, which is designed and trusted to operate on the secure desktop.
Grace never attempts to bypass or touch the secure desktop.
"""

from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
from typing import Optional

logger = logging.getLogger("grace.secure_desktop")

# Spoken aloud: NEVER contains command phrases like 'click', 'voice access wake up', or 'hello'
# so speaker output cannot accidentally be heard and executed by Voice Access.
SAFE_HANDOFF_SPOKEN = "That's a Windows security screen. Voice Access can handle it."

# Shown in the UI pill only:
SAFE_HANDOFF_PILL_TEXT = 'Say "voice access wake up", then "click Yes" or "click No"'

ACCESSIBILITY_REG_KEY = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Accessibility"
ACCESSIBILITY_REG_VALUE = "Configuration"

_mock_desktop_name: Optional[str] = None
_mock_session_locked: Optional[bool] = None


def set_mock_desktop_name(name: Optional[str]) -> None:
    """Set mock input desktop name for testing."""
    global _mock_desktop_name
    _mock_desktop_name = name


def set_mock_session_locked(locked: Optional[bool]) -> None:
    """Set mock session locked state for testing."""
    global _mock_session_locked
    _mock_session_locked = locked


def get_input_desktop_name() -> str:
    """Return the name of the current input desktop ('Default', 'Winlogon', etc.).

    If OpenInputDesktop fails (e.g. access denied on a secure desktop), returns 'SecureDesktop'.
    """
    global _mock_desktop_name
    if _mock_desktop_name is not None:
        return _mock_desktop_name

    try:
        user32 = ctypes.windll.user32
        # DESKTOP_READOBJECTS = 0x0001
        h_desk = user32.OpenInputDesktop(0, False, 0x0001)
        if not h_desk:
            return "SecureDesktop"
        buf = ctypes.create_unicode_buffer(256)
        needed = wintypes.DWORD()
        # UOI_NAME = 2
        ok = user32.GetUserObjectInformationW(h_desk, 2, buf, 512, ctypes.byref(needed))
        user32.CloseDesktop(h_desk)
        if ok and buf.value:
            return buf.value
        return "SecureDesktop"
    except Exception as exc:
        logger.debug("Failed to query input desktop: %s", exc)
        return "Default"


def is_input_desktop_secure() -> bool:
    """Return True if the input desktop is NOT 'Default' (e.g. UAC prompt, lock screen, sign-in)."""
    name = get_input_desktop_name()
    return name.strip().lower() != "default"


def is_session_locked() -> bool:
    """Return True if the session is locked."""
    global _mock_session_locked
    if _mock_session_locked is not None:
        return _mock_session_locked
    return is_input_desktop_secure()


def is_deaf() -> bool:
    """Security rule: Grace stops speech recognition completely on secure or locked screens.

    Nothing is transcribed, buffered, logged or written to tapes while deaf.
    """
    return is_input_desktop_secure() or is_session_locked()


def get_signin_accessibility_config() -> str:
    """Read HKLM Accessibility Configuration value."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, ACCESSIBILITY_REG_KEY, 0, winreg.KEY_READ) as key:
            val, _ = winreg.QueryValueEx(key, ACCESSIBILITY_REG_VALUE)
            return val or ""
    except Exception as exc:
        logger.debug("Failed reading Accessibility Configuration: %s", exc)
        return ""


def is_voice_access_enabled_at_signin() -> bool:
    """Check if 'voiceaccess' is in the sign-in accessibility configuration."""
    val = get_signin_accessibility_config()
    items = [x.strip().lower() for x in val.split(",") if x.strip()]
    return "voiceaccess" in items


def merge_voice_access_config(current: str) -> str:
    """Add 'voiceaccess' if missing, keeping existing entries untouched and in order."""
    if not current:
        return "voiceaccess"
    parts = [p.strip() for p in current.split(",") if p.strip()]
    if any(p.lower() == "voiceaccess" for p in parts):
        return current
    parts.append("voiceaccess")
    return ",".join(parts)


def enable_voice_access_at_signin_registry() -> bool:
    """Helper that modifies HKLM Configuration to include voiceaccess.

    Only registry write Grace is permitted to make. Reads, adds voiceaccess,
    writes back, re-reads to verify, and returns True only if verified.
    """
    import winreg

    try:
        current = get_signin_accessibility_config()
        merged = merge_voice_access_config(current)
        if merged != current:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                ACCESSIBILITY_REG_KEY,
                0,
                winreg.KEY_SET_VALUE | winreg.KEY_READ,
            ) as key:
                winreg.SetValueEx(key, ACCESSIBILITY_REG_VALUE, 0, winreg.REG_SZ, merged)
        # Re-read and verify
        verified = get_signin_accessibility_config()
        items = [x.strip().lower() for x in verified.split(",") if x.strip()]
        return "voiceaccess" in items
    except Exception as exc:
        logger.error("Failed to write/verify Voice Access configuration in registry: %s", exc)
        return False


def request_enable_voice_access_elevation() -> bool:
    """Request elevation via ShellExecute runas to run grace.exe --enable-voice-access-at-signin."""
    import sys

    try:
        if getattr(sys, "frozen", False):
            exe = sys.executable
            args = "--enable-voice-access-at-signin"
        else:
            exe = sys.executable
            args = "-m grace.main --enable-voice-access-at-signin"
        shell32 = ctypes.windll.shell32
        res = shell32.ShellExecuteW(None, "runas", exe, args, None, 1)
        return res > 32
    except Exception as exc:
        logger.error("Failed to request elevation for Voice Access setup: %s", exc)
        return False
