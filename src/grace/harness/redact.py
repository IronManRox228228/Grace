"""Redaction of secrets, credentials, and user identity for session tapes.

Enforces Red-Team findings:
- R21: Tapes must redact typed text from password fields and secret patterns.
- R44: Tapes must not leak the Windows username or home-directory path.
"""

from __future__ import annotations

import getpass
import os
import re
from typing import Any

# Secret patterns (API keys, tokens, credentials, payment cards)
_SECRET_PATTERNS = [
    # API keys and bearer tokens
    re.compile(r"\b(?:sk-[a-zA-Z0-9_\-]{20,})\b", re.IGNORECASE),
    re.compile(r"\b(?:ghp_[a-zA-Z0-9]{30,}|github_pat_[a-zA-Z0-9_]{30,})\b", re.IGNORECASE),
    re.compile(r"\b(?:bearer\s+[a-zA-Z0-9_\-\.]{16,})\b", re.IGNORECASE),
    # Credit/debit card numbers (13-19 digits, with optional spaces, dashes, commas, or dots - R25)
    re.compile(r"\b(?:\d{4}[,\.\s-]+){3}\d{1,4}\b"),
]

_PASSPHRASE_PATTERN = re.compile(r"\b(?=[a-zA-Z_\-\.]{14,})(?:[a-zA-Z]{3,}[-_.]){2,}[a-zA-Z]{3,}\b")


_PASSWORD_HINTS = (
    "password", "passwd", "pwd", "secret", "pin", "token", "api_key", "apikey",
    "cvv", "cvc", "ssn", "credit_card", "passphrase",
)


def get_user_and_home() -> tuple[str, str]:
    """Retrieve username and home path safely."""
    username = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    if not username:
        try:
            username = getpass.getuser()
        except Exception:
            username = ""
    home = os.path.expanduser("~")
    return username, home


def _is_password_field(role: Optional[str], name: Optional[str]) -> bool:
    role_str = (role or "").lower()
    name_str = (name or "").lower()
    if role_str == "password":
        return True
    return any(hint in name_str for hint in ("password", "passwd", "pwd", "pin"))


def redact_string(text: str, username: str = "", home: str = "") -> str:
    """Redact username, home directory, and secrets from a string."""
    if not text:
        return text

    if not username or not home:
        u, h = get_user_and_home()
        username = username or u
        home = home or h

    out = text

    # 1. Redact home path (both backslash and forward slash forms, case-insensitive)
    if home and len(home) > 3:
        home_bs = home
        home_fs = home.replace("\\", "/")
        out = re.sub(re.escape(home_bs), "<HOME>", out, flags=re.IGNORECASE)
        out = re.sub(re.escape(home_fs), "<HOME>", out, flags=re.IGNORECASE)

    # 2. Redact username
    if username and len(username) > 1:
        out = re.sub(r"\b" + re.escape(username) + r"\b", "<USER>", out, flags=re.IGNORECASE)

    # 3. Redact secret patterns
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("<redacted>", out)

    # 4. Redact passphrases (excluding known tools and untrusted delimiters)
    try:
        from grace.intent.tools import ALL_TOOLS
        excluded = {t.name for t in ALL_TOOLS}
    except Exception:
        excluded = set()
    excluded |= {
        "UNTRUSTED_DATA_START", "UNTRUSTED_DATA_END",
        "untrusted_data_start", "untrusted_data_end",
        "set_speech_rate", "cua_list_windows", "cua_secondary_action",
    }
    def _sub_passphrase(m: re.Match) -> str:
        word = m.group(0)
        if word in excluded or word.lower() in excluded:
            return word
        if word.startswith("cua_") or "UNTRUSTED_DATA" in word:
            return word
        if re.search(r"\.(?:com|org|net|edu|gov|io|ai|app|dev|co|uk|de|jp|fr|ca|au|in|ru|ch|it|nl|se|no|es|mil|xyz|site|online|tech)\b", word, re.IGNORECASE):
            return word
        return "<redacted>"
    out = _PASSPHRASE_PATTERN.sub(_sub_passphrase, out)

    return out


def redact_data(data: Any, username: str = "", home: str = "") -> Any:
    """Recursively redact secrets, user info, and password fields in arbitrary JSON data."""
    if not username or not home:
        u, h = get_user_and_home()
        username = username or u
        home = home or h

    if isinstance(data, str):
        return redact_string(data, username=username, home=home)

    if isinstance(data, dict):
        new_dict = {}
        is_type_tool = data.get("tool") in ("type_text", "cua_type", "cua_type_text")

        # Check if this object represents an element node with a password role
        role = data.get("role")
        name = data.get("name")
        is_pwd_element = _is_password_field(role, name) or data.get("is_password")
        is_element_node = "id" in data and "role" in data and "rect" in data

        for k, v in data.items():
            # Tool names and contract event types are never user secrets
            if k in ("tool", "type"):
                new_dict[k] = v
                continue

            k_lower = str(k).lower()


            # Redact password field values in element snapshots
            if is_pwd_element and k_lower in ("value", "text"):
                new_dict[k] = (
                    "[password field, filled]"
                    if (v and v != "[password field, empty]")
                    else "[password field, empty]"
                )
                continue

            # Keep ordinary text field values visible in element snapshots unless secret-shaped
            if is_element_node and k_lower in ("value", "text") and not is_pwd_element:
                from grace.agent.facts import looks_like_secret, looks_like_secret_value
                if looks_like_secret(name or role or "", str(v)) or looks_like_secret_value(str(v)):
                    new_dict[k] = "<redacted>"
                else:
                    new_dict[k] = v
                continue

            # Redact password fields in tool dispatch params
            if is_type_tool and k == "params" and isinstance(v, dict):
                p_copy = dict(v)
                target_role = p_copy.get("role") or p_copy.get("element_role")
                target_name = p_copy.get("target") or p_copy.get("target_name") or p_copy.get("name")
                if _is_password_field(target_role, target_name) or any(h in k_lower for h in _PASSWORD_HINTS):
                    if "text" in p_copy:
                        p_copy["text"] = "<redacted>"
                new_dict[k] = redact_data(p_copy, username=username, home=home)
                continue

            # Keys explicitly named password/secret/token
            if any(h == k_lower or f"_{h}" in k_lower or f"{h}_" in k_lower for h in _PASSWORD_HINTS):
                if isinstance(v, str):
                    new_dict[k] = "<redacted>"
                    continue

            new_dict[k] = redact_data(v, username=username, home=home)
        return new_dict

    if isinstance(data, list):
        return [redact_data(item, username=username, home=home) for item in data]

    if isinstance(data, tuple):
        return tuple(redact_data(item, username=username, home=home) for item in data)

    return data
