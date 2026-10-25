"""Harm covers that need to know what is on screen.

`SafetyGuard.evaluate` sees only the action and its params. These two covers
also need the window, the page and the user's goal, so they live here and the
loop calls them next to the guard:

  * sign-in pages: stop and ask the user rather than pick an account or type
    an identity the user never gave;
  * terminals: typing into a console runs a command, so it needs a spoken yes.

Both are harm covers, not judgements of the model's choice. Everything else the
model does on these screens is left alone.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger("grace.agent.context_covers")

TEXT_ENTRY_TOOLS = ("cua_type_text", "cua_set_value")

# -- sign-in pages ------------------------------------------------------------

_AUTH_URL_RE = re.compile(
    r"accounts\.google\.com|login\.microsoftonline\.com"
    r"|/login|/signin|/sign-in|/oauth|/authorize",
    re.IGNORECASE,
)
_AUTH_TITLE_RE = re.compile(r"\b(?:sign[\s-]?in|log[\s-]?in|login)\b", re.IGNORECASE)

# The user asked for it, so the sign-in cover stands down.
_GOAL_SIGNIN_RE = re.compile(
    r"\b(?:sign|log)(?:\s|-)?(?:in|into|on)\b|\blogin\b", re.IGNORECASE
)

# Buttons that choose an identity or grant access.
_IDENTITY_CLICK_RE = re.compile(
    r"use another account|add(?:\s+another)?\s+account|choose an account"
    r"|\bcontinue with\b|\bsign(?:ing)?[\s-]?in with\b|\blog(?:ging)?[\s-]?in with\b"
    r"|\bsign in as\b|\bcontinue as\b"
    r"|^\s*(?:allow|authori[sz]e|accept)\b",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
# Only phone-shaped text: a leading + or ( , or 10+ digits split by spaces or
# dashes. A bare number, a date or a decimal is not a phone number.
_PHONE_RE = re.compile(r"(?<![\w.])(?:\+|\()?\d[\d\s()\-]{5,}\d(?![\w])")

# -- terminals ----------------------------------------------------------------

TERMINAL_PROCESSES = frozenset({
    "windowsterminal.exe", "powershell.exe", "pwsh.exe", "cmd.exe", "conhost.exe",
    "wsl.exe", "bash.exe", "mintty.exe",
})
TERMINAL_CLASSES = frozenset({"ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS"})

# Keys that submit or paste into a terminal. Arrow keys and the like are not
# covered: they run nothing.
_TERMINAL_KEYS = {"enter", "return", "ctrl+v", "ctrl+shift+v", "shift+insert"}


@dataclass
class CoverDecision:
    """What a cover wants done instead of running the step."""

    kind: str  # "ask_user" | "confirm" | "refuse"
    message: str  # spoken question, or the fact returned to the model


def window_process_name(hwnd: int) -> str:
    """Lower-case exe name of the process owning `hwnd`, or "" if unknown."""
    if not hwnd:
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
        if not pid.value:
            return ""
        handle = kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buf))
            if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return buf.value.replace("/", "\\").rsplit("\\", 1)[-1].lower()
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        pass
    return ""


def _elements(snapshot: Any) -> list[Any]:
    graph = getattr(snapshot, "graph", None)
    els = getattr(graph, "elements", None) if graph is not None else None
    if els is None:
        els = getattr(snapshot, "ui_elements", None)
    return list(els or [])


def _get(el: Any, attr: str, default: Any = "") -> Any:
    if isinstance(el, dict):
        return el.get(attr, default)
    return getattr(el, attr, default)


def _screen_text(snapshot: Any) -> str:
    """Everything legible on screen, for 'was this shown to the user'."""
    parts: list[str] = []
    window = getattr(snapshot, "active_window", None)
    if window is not None:
        parts.append(str(getattr(window, "title", "") or ""))
    for line in getattr(snapshot, "ocr_lines", None) or []:
        parts.append(str(getattr(line, "text", "") or ""))
    for el in _elements(snapshot):
        if _get(el, "is_password", False):
            continue
        parts.append(str(_get(el, "name", "") or ""))
        parts.append(str(_get(el, "value", "") or ""))
    return "\n".join(p for p in parts if p)


def _has_password_field(snapshot: Any) -> bool:
    from grace.perception.elements import is_password_element

    for el in _elements(snapshot):
        if _get(el, "is_password", False):
            return True
        role = str(_get(el, "role", "") or "").lower()
        if role == "password":
            return True
        if role in ("edit", "textbox", "text", "combobox") and is_password_element(
            role, _get(el, "name", ""), _get(el, "placeholder", ""),
            _get(el, "automation_id", ""), _get(el, "source", ""),
        ):
            return True
    return False


def _site_name(title: str, url: Optional[str]) -> str:
    if url:
        host = urlparse(url if "//" in url else "//" + url).hostname or ""
        host = host[4:] if host.startswith("www.") else host
        if host:
            return host
    first = (title or "").split(" - ")[0].strip()
    # "Sign in – Google accounts" -> "Google accounts"
    for dash in ("–", "—"):
        if dash in first:
            first = first.split(dash)[-1].strip()
    return first or "This page"


def is_auth_screen(snapshot: Any, url: Optional[str]) -> bool:
    """True on a sign-in page, by URL, title or a password field.

    The title and URL patterns apply to browser-like windows only, so a
    Notepad file called "login notes" is not mistaken for a sign-in page. A
    password field counts anywhere.
    """
    from grace.perception.uia_provider import is_browser_window

    window = getattr(snapshot, "active_window", None)
    title = str(getattr(window, "title", "") or "")
    class_name = str(getattr(window, "class_name", "") or "")
    if url and _AUTH_URL_RE.search(url):
        return True
    if (url or is_browser_window(title, class_name)) and _AUTH_TITLE_RE.search(title):
        return True
    return _has_password_field(snapshot)


def _targets_identity(params: dict[str, Any], snapshot: Any) -> bool:
    names: list[str] = []
    for key in ("target_name", "describe", "name", "label"):
        val = params.get(key)
        if isinstance(val, str):
            names.append(val)
    graph = getattr(snapshot, "graph", None)
    raw_id = params.get("element_id", params.get("element_index"))
    if graph is not None and raw_id is not None:
        try:
            node = graph.by_id(int(raw_id))
        except (ValueError, TypeError):
            node = None
        if node is not None:
            names.append(str(_get(node, "name", "") or ""))
    return any(_IDENTITY_CLICK_RE.search(n) or _EMAIL_RE.search(n) for n in names if n)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _unsourced_contact(text: str, haystack: str) -> Optional[tuple[str, str]]:
    """First ("email"|"phone", value) in `text` that `haystack` does not contain."""
    lower = haystack.lower()
    hay_digits = _digits(haystack)
    for match in _EMAIL_RE.findall(text):
        if match.lower() not in lower:
            return "email", match
    for match in _PHONE_RE.findall(text):
        digits = _digits(match)
        marked = match.lstrip().startswith(("+", "("))
        split = bool(re.search(r"[\s\-]", match.strip()))
        shaped = (marked and 7 <= len(digits) <= 15) or (split and 10 <= len(digits) <= 15)
        if shaped and digits not in hay_digits:
            # A date-like run of digits is not a phone number.
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", match.strip()):
                continue
            return "phone", match.strip()
    return None


def _terminal_window(snapshot: Any, process: Optional[str]) -> bool:
    window = getattr(snapshot, "active_window", None)
    if window is None:
        return False
    class_name = str(getattr(window, "class_name", "") or "")
    title = str(getattr(window, "title", "") or "")
    if class_name in TERMINAL_CLASSES:
        return True
    proc = process if process is not None else window_process_name(getattr(window, "hwnd", 0))
    if proc and proc.lower() in TERMINAL_PROCESSES:
        return True
    # VS Code's integrated terminal: the window is VS Code and focus is in it.
    if "visual studio code" in title.lower():
        for el in _elements(snapshot):
            if _get(el, "focused", False) and "terminal" in str(_get(el, "name", "") or "").lower():
                return True
    return False


def _terminal_key(raw: Any) -> bool:
    from grace.agent.safety import SafetyGuard

    return SafetyGuard._normalise_key(raw) in {
        SafetyGuard._normalise_key(k) for k in _TERMINAL_KEYS
    }


def evaluate_context(
    action: str,
    params: dict[str, Any],
    snapshot: Any,
    *,
    goal: str = "",
    url: Optional[str] = None,
    facts_text: str = "",
    process: Optional[str] = None,
) -> Optional[CoverDecision]:
    """Run the screen-aware covers; None means go ahead."""
    # Terminals: the same yes/no flow `delete` uses.
    if action in TEXT_ENTRY_TOOLS or action == "cua_press_key":
        if _terminal_window(snapshot, process):
            if action == "cua_press_key":
                if _terminal_key(params.get("key")):
                    return CoverDecision(
                        "confirm", f"Press {params.get('key')} in the terminal? Say yes or no."
                    )
            else:
                text = params.get("text", params.get("value", ""))
                return CoverDecision(
                    "confirm", f"Run this in the terminal: '{text}'? Say yes or no."
                )

    # Sign-in pages.
    asked_to_sign_in = bool(_GOAL_SIGNIN_RE.search(goal or ""))
    if not asked_to_sign_in and is_auth_screen(snapshot, url):
        clicking = action in ("cua_click", "cua_secondary_action") and _targets_identity(
            params, snapshot
        )
        if clicking or action in TEXT_ENTRY_TOOLS:
            window = getattr(snapshot, "active_window", None)
            site = _site_name(str(getattr(window, "title", "") or ""), url)
            return CoverDecision(
                "ask_user",
                f"{site} wants you to sign in. Do you want to sign in yourself first?",
            )

    # Anywhere: an email or phone number nobody supplied.
    if action in TEXT_ENTRY_TOOLS:
        text = params.get("text", params.get("value", ""))
        if isinstance(text, str) and text:
            haystack = "\n".join((goal or "", facts_text or "", _screen_text(snapshot)))
            found = _unsourced_contact(text, haystack)
            if found:
                noun = "email address" if found[0] == "email" else "phone number"
                return CoverDecision(
                    "refuse", f"That {noun} was not given by the user or shown on screen."
                )
    return None
