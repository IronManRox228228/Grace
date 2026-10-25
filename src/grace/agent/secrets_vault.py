"""Saved-secret vault with site-bound typing and eTLD+1 verification.

Guarantees:
- Secrets are encrypted at rest via DPAPI in the PersistentMemoryStore.
- Secrets are never logged, traced, written to tapes, put in memory facts, or put in any prompt.
- Typing requires:
  1. Focused element is a password field.
  2. Domain secrets: active browser tab domain matches site's eTLD+1.
  3. App secrets: foreground process exe matches site.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any, Optional

logger = logging.getLogger("grace.secrets_vault")

# Common two-part public suffixes
MULTI_PART_TLDS = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "me.uk", "net.uk", "sch.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.jp", "ne.jp", "ac.jp", "go.jp", "ed.jp",
    "co.nz", "net.nz", "org.nz", "govt.nz",
    "com.br", "net.br", "org.br", "gov.br",
    "com.cn", "net.cn", "org.cn", "gov.cn",
    "com.sg", "edu.sg", "gov.sg", "net.sg",
    "co.in", "net.in", "org.in", "gen.in", "firm.in", "ind.in",
    "com.tw", "org.tw", "idv.tw", "gov.tw",
    "com.hk", "edu.hk", "gov.hk", "net.hk",
    "co.za", "org.za", "web.za", "gov.za",
}


def extract_etld_plus_one(url_or_domain: str) -> str:
    """Extract the effective top-level domain plus one (eTLD+1) from a URL or hostname."""
    if not url_or_domain:
        return ""

    raw = url_or_domain.strip().lower()
    # Strip protocol scheme if present
    if "://" in raw:
        parsed = urllib.parse.urlparse(raw)
        host = parsed.netloc or parsed.path
    else:
        # Might start with path or query
        host = raw.split("/")[0].split("?")[0].split("#")[0]

    # Strip port if present
    if ":" in host:
        host = host.split(":")[0]

    host = host.strip(".").strip()
    if not host:
        return ""

    parts = host.split(".")
    if len(parts) <= 2:
        return host

    # Check two-part suffix (e.g. example.co.uk)
    two_part = f"{parts[-2]}.{parts[-1]}"
    if two_part in MULTI_PART_TLDS and len(parts) >= 3:
        return f"{parts[-3]}.{two_part}"

    return f"{parts[-2]}.{parts[-1]}"


def classify_site(site_arg: str) -> tuple[str, str]:
    """Classify a site argument into (site_kind, site).

    Returns:
        ("app", process_exe) if ending in .exe, else ("domain", etld_plus_one).
    """
    cleaned = site_arg.strip()
    if cleaned.lower().endswith(".exe"):
        return "app", cleaned.lower()
    return "domain", extract_etld_plus_one(cleaned)


def type_saved_secret_guard(
    secret_info: dict[str, Any],
    focused_is_password: bool,
    active_domain: Optional[str] = None,
    active_process_exe: Optional[str] = None,
) -> tuple[bool, str]:
    """Deterministically check if typing the secret is safe and permitted.

    Returns (allowed, refusal_sentence).
    Refusal sentence must not speak or leak secret values or sites.
    """
    name = secret_info.get("name", "password")

    # Guard 1: Must be a password field
    if not focused_is_password:
        return False, "The focused field is not a password field."

    site_kind = secret_info.get("site_kind", "domain")
    expected_site = (secret_info.get("site") or "").strip().lower()

    # Guard 2: Domain secret check
    if site_kind == "domain":
        if not active_domain:
            return False, f"I couldn't verify the active website for {name}."
        active_etld = extract_etld_plus_one(active_domain).lower()
        if not active_etld or active_etld != expected_site:
            return False, f"The active website does not match the site saved for {name}."

    # Guard 3: App secret check
    elif site_kind == "app":
        if not active_process_exe:
            return False, f"I couldn't verify the active application for {name}."
        proc_cleaned = active_process_exe.strip().lower()
        if proc_cleaned != expected_site:
            return False, f"The active application does not match the app saved for {name}."

    return True, ""
