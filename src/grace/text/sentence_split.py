"""Shared sentence-splitting utility for TTS chunking.

Used by both the Kokoro engine and the response generator to break
long text into speakable sentence chunks without splitting on
abbreviations (e.g. "Dr. Smith").
"""

import re

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "eg", "ie", "vs", "etc", "approx", "dept", "est", "govt",
    "capt", "lt", "col", "gen", "sgt", "vol", "no",
    "co", "inc", "ltd", "corp",
}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, merging false splits on abbreviations."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    merged = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue
        tail = part.split()[-1].rstrip(".").lower() if part.split() else ""
        if tail in ABBREVIATIONS and i + 1 < len(parts):
            part = part + " " + parts[i + 1].strip()
            i += 1
        merged.append(part)
        i += 1
    return merged
