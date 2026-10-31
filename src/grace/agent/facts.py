"""Facts and user memory helpers (PLAN.md §12.1).

Deterministic memory pattern detection, pronoun flipping,
provenance safety guards, and secret detectors ported from
crates/grace-memory/src/facts.rs and crates/grace-memory/src/secret.rs.
"""

import hashlib
import re
import string
from typing import Any, Optional

SECRET_TOPIC_HINTS = (
    "password", "passwd", "pwd", "secret", "token", "pin", "otp", "cvv",
    "cvc", "ssn", "api_key", "apikey", "credit_card", "passphrase",
)


def looks_like_secret(topic: str, value: str) -> bool:
    """Port of crates/grace-memory/src/secret.rs looks_like_secret.

    Rejects anything with a secret-shaped topic name or secret-shaped value.
    """
    topic_lower = (topic or "").lower()
    if any(hint in topic_lower for hint in SECRET_TOPIC_HINTS):
        return True
    return looks_like_secret_value(value)


def looks_like_secret_value(value: str) -> bool:
    """Check if value matches secret patterns (cards, passphrases, complex tokens)."""
    trimmed = (value or "").strip()
    if not trimmed:
        return False

    if looks_like_card_number(trimmed):
        return True

    if looks_like_passphrase(trimmed):
        return True

    for token in trimmed.split():
        clean = token.strip(string.punctuation)
        if len(clean) >= 8:
            lower = any(c.islower() for c in clean)
            upper = any(c.isupper() for c in clean)
            digit = any(c.isdigit() for c in clean)
            symbol = any(not c.isalnum() for c in clean)
            digit_count = sum(1 for c in clean if c.isdigit())
            letter_count = sum(1 for c in clean if c.isalpha())
            classes = sum(1 for b in (lower, upper, digit, symbol) if b)
            if classes >= 3:
                return True
            if (lower or upper) and digit and digit_count >= 2 and letter_count >= 3:
                return True

    return False


def looks_like_passphrase(s: str) -> bool:
    """Passphrase heuristic (R25): 3+ words separated by -, _, or . with length >= 14."""
    if not any(c.isspace() for c in s):
        for delim in ('-', '_', '.'):
            if delim in s:
                parts = s.split(delim)
                if len(parts) >= 3 and all(len(p) >= 3 and p.isalpha() for p in parts) and len(s) >= 14:
                    if delim == '.' and parts[-1].lower() in (
                        "com", "org", "net", "edu", "gov", "io", "ai", "app", "dev", "co",
                        "uk", "de", "jp", "fr", "ca", "au", "in", "ru", "ch", "it", "nl",
                        "se", "no", "es", "mil", "xyz", "site", "online", "tech",
                    ):
                        return False
                    return True
    return False


def looks_like_card_number(s: str) -> bool:
    """Run of 13-19 digits with optional spaces/dashes/commas/periods."""
    digits = [c for c in s if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    return all(c.isdigit() or c in (' ', '-', ',', '.') for c in s)


def redact_value(value: str, salt: Optional[str] = None) -> str:
    """Redact fact value for logs and traces (PLAN.md §12.1)."""
    raw = value or ""
    salt_str = salt or "grace_redact_salt"
    h = hashlib.sha256(f"{salt_str}:{raw}".encode("utf-8")).hexdigest()[:16]
    return f"[REDACTED len={len(raw)} hash={h}]"


PRONOUN_MAP = {
    "i": "you",
    "me": "you",
    "my": "your",
    "mine": "yours",
    "myself": "yourself",
    "am": "are",
    "i'm": "you're",
    "im": "you're",
    "ive": "you've",
    "i've": "you've",
}


def pronoun_flip(text: str) -> str:
    """Flip 1st person pronouns to 2nd person for confirmation questions."""
    tokens = re.split(r"(\w+(?:'\w+)?)", text)
    result = []
    for token in tokens:
        low = token.lower()
        if low in PRONOUN_MAP:
            result.append(PRONOUN_MAP[low])
        else:
            result.append(token)
    return "".join(result)


def is_value_in_transcript(value: str, transcript: str) -> bool:
    """Provenance safety check: verify that fact value was spoken in current transcript."""
    if not value or not transcript:
        return False
    norm_val = re.sub(r"[^\w\s]", "", value.lower()).strip()
    norm_trans = re.sub(r"[^\w\s]", "", transcript.lower()).strip()
    if norm_val in norm_trans:
        return True
    # Check if all content words of value are in transcript
    val_words = [w for w in norm_val.split() if len(w) > 2]
    if val_words and all(w in norm_trans.split() for w in val_words):
        return True
    return False


def detect_remember_pattern(transcript: str) -> Optional[dict[str, Any]]:
    """Detect deterministic memory intent from user transcript.

    Returns dict with {topic, value, confirmation_prompt, spoken_fact} or None.
    Handles:
    - "my name is Sam" / "remember my name is Sam"
    - "remember that I prefer Firefox" / "I prefer Firefox"
    - "remember my usual song is Signal by Home" / "my usual song is Signal by Home"
    - General "remember that ..." / "remember ..."
    """
    if not transcript or not transcript.strip():
        return None

    raw = transcript.strip()
    clean = re.sub(r"[^\w\s]", "", raw.lower()).strip()

    # Exclude recall, forget, stop, repeat, greetings
    if clean.startswith("what do you remember") or clean.startswith("what do you know"):
        return None
    if clean.startswith("forget"):
        return None
    if clean in ("stop", "repeat"):
        return None

    # 1. "my name is <name>" / "remember [that] my name is <name>"
    m_name = re.match(
        r"^(?:remember\s+(?:that\s+)?)?my\s+name\s+is\s+(.+)$",
        raw,
        re.IGNORECASE,
    )
    if m_name:
        name = m_name.group(1).strip().strip(string.punctuation)
        if name:
            return {
                "topic": "name",
                "value": name,
                "confirmation_prompt": f"Should I remember that your name is {name}?",
                "spoken_fact": f"your name is {name}",
            }

    # 2. "remember [that] my usual song is <song>" / "my usual song is <song>"
    m_song = re.match(
        r"^(?:remember\s+(?:that\s+)?)?my\s+usual\s+song\s+is\s+(.+)$",
        raw,
        re.IGNORECASE,
    )
    if m_song:
        song = m_song.group(1).strip().strip(string.punctuation)
        if song:
            return {
                "topic": "usual_song",
                "value": song,
                "confirmation_prompt": f"Should I remember that your usual song is {song}?",
                "spoken_fact": f"your usual song is {song}",
            }

    # 3. "remember [that] I prefer <item>" / "I prefer <item>"
    m_pref = re.match(
        r"^(?:remember\s+(?:that\s+)?)?i\s+prefer\s+(.+)$",
        raw,
        re.IGNORECASE,
    )
    if m_pref:
        pref = m_pref.group(1).strip().strip(string.punctuation)
        if pref:
            pref_low = pref.lower()
            if pref_low in ("firefox", "chrome", "edge", "safari", "brave", "opera"):
                topic = "preference:browser"
            else:
                slug_pref = re.sub(r"\s+", "_", pref_low)
                topic = f"preference:{slug_pref}"
            return {
                "topic": topic,
                "value": pref,
                "confirmation_prompt": f"Should I remember that you prefer {pref}?",
                "spoken_fact": f"you prefer {pref}",
            }

    # 4. General "remember that <fact>" or "remember <fact>"
    m_rem = re.match(r"^remember\s+(?:that\s+)?(.+)$", raw, re.IGNORECASE)
    if m_rem:
        body = m_rem.group(1).strip()
        # Don't capture command reminders like "remember to ..."
        if body.lower().startswith("to "):
            return None
        # Clean trailing punctuation
        body_clean = body.rstrip(string.punctuation)
        flipped = pronoun_flip(body_clean)
        # Generate a slug topic
        topic = "fact:" + re.sub(r"[^\w]+", "_", body_clean.lower())[:32].strip("_")
        return {
            "topic": topic,
            "value": body_clean,
            "confirmation_prompt": f"Should I remember that {flipped}?",
            "spoken_fact": flipped,
        }

    return None


def format_fact_for_speech(topic: str, value: str) -> str:
    """Format a stored fact for speaking aloud naturally."""
    if topic == "name":
        return f"your name is {value}"
    if topic == "usual_song":
        return f"your usual song is {value}"
    if topic == "preference:browser":
        return f"you prefer {value}"
    if topic.startswith("preference:"):
        return f"you prefer {value}"
    if topic.startswith("fact:"):
        return pronoun_flip(value)
    clean_topic = topic.replace("_", " ").replace(":", " ")
    return f"{clean_topic} is {value}"


def is_recall_command(clean: str) -> bool:
    """Check if utterance asks Grace to recall facts about the user."""
    return clean in (
        "what do you remember about me",
        "what do you remember",
        "what do you know about me",
        "what do you know",
        "what have you remembered",
        "tell me what you remember",
        "tell me what you know about me",
    ) or clean.startswith("what do you remember") or clean.startswith("what do you know about me")


def detect_forget_fact_pattern(
    transcript: str, confirmed_facts: Optional[list[dict[str, Any]]] = None
) -> Optional[tuple[str, str]]:
    """Detect voice command to forget a specific fact (PLAN.md §12.1).

    Returns (topic, human_label) or None.
    Handles:
    - "forget my name" / "forget that my name is Sam"
    - "forget that I prefer Firefox" / "forget I prefer Firefox"
    - "forget my usual song" / "forget that my usual song is Signal by Home"
    - "forget <fact value or topic name>"
    """
    if not transcript or not transcript.strip():
        return None

    raw = transcript.strip()
    clean = re.sub(r"[^\w\s]", "", raw.lower()).strip()

    if not clean.startswith("forget"):
        return None

    # Exclude broad session wipes handled elsewhere
    if clean in ("forget today", "forget what i did today", "forget everything", "forget all"):
        return None
    if clean.startswith("forget what i did in "):
        return None

    target = re.sub(r"^forget\s+(?:that\s+)?", "", clean).strip()
    if not target:
        return None

    # 1. Name
    if target in ("my name", "name") or target.startswith("my name is ") or target.startswith("name is "):
        return ("name", "your name")

    # 2. Usual song
    if target in ("my usual song", "usual song") or target.startswith("my usual song is "):
        return ("usual_song", "your usual song")

    # 3. Preferences (e.g. browser or generic prefer)
    if target in ("browser preference", "my browser", "my preferred browser"):
        return ("preference:browser", "your browser preference")

    m_pref = re.search(r"\b(?:i\s+)?prefer\s+(.+)$", target)
    if m_pref:
        pref = m_pref.group(1).strip()
        if pref in ("firefox", "chrome", "edge", "safari", "brave", "opera"):
            return ("preference:browser", f"that you prefer {pref.capitalize()}")
        slug = re.sub(r"\s+", "_", pref)
        return (f"preference:{slug}", f"that you prefer {pref}")

    # 4. Match against currently confirmed facts if available
    if confirmed_facts:
        for f in confirmed_facts:
            topic = f.get("topic", "")
            val = str(f.get("value", "")).lower()
            val_clean = re.sub(r"[^\w\s]", "", val).strip()
            # If value appears in utterance (e.g. "firefox" or "signal by home")
            if val_clean and val_clean in target:
                label = format_fact_for_speech(topic, f.get("value", ""))
                return (topic, label)
            # If topic words match target
            clean_topic = topic.replace("_", " ").replace(":", " ")
            if clean_topic in target or target in clean_topic:
                label = format_fact_for_speech(topic, f.get("value", ""))
                return (topic, label)

    return None

