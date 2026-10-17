"""Text normalizer for ASR benchmark evaluation (PLAN.md §8, Task 09).

Ensures identical, fair text normalization across all evaluated models
(Whisper, Parakeet-TDT, Moonshine, etc.).

Normalization Pipeline:
1. Unicode normalization (NFKC).
2. Lowercase transformation.
3. Removal of transcriber metadata / acoustic tokens in brackets (e.g. `[cough]`, `<laughter>`).
4. Typographic symbol standardization (smart quotes, em-dashes, hyphens).
5. Number standardization (cardinal digits 0-20 mapped to spoken word equivalents).
6. Punctuation removal (stripping all punctuation characters).
7. Whitespace normalization (collapsing multiple spaces, stripping leading/trailing whitespace).
"""

from __future__ import annotations

import re
import unicodedata

# Single-digit and common number mapping to words
_DIGIT_TO_WORD = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "16": "sixteen",
    "17": "seventeen",
    "18": "eighteen",
    "19": "nineteen",
    "20": "twenty",
}

# Regex to find bracketed metadata like [cough], <applause>, (laughter)
_BRACKETED_PATTERN = re.compile(r"\[.*?\]|\<.*?\>|\(.*?\)")

# Characters to replace with space (e.g. hyphens, slashes)
_PUNCT_TO_SPACE = re.compile(r"[-_/]")

# All remaining punctuation to strip completely
_PUNCT_TO_STRIP = re.compile(r"""[.,!?:;"'`~@#$%^&*()+=<>{}\[\]|\\~]""")

# Number token pattern (isolated digits 0-20)
_NUMBER_PATTERN = re.compile(r"\b(\d+)\b")


def _replace_number(match: re.Match[str]) -> str:
    num_str = match.group(1)
    return _DIGIT_TO_WORD.get(num_str, num_str)


def normalize_text(text: str) -> str:
    """Normalize reference or hypothesis transcript for WER evaluation.

    Args:
        text: Raw transcript text from prompt or model output.

    Returns:
        Cleaned, lowercased, punctuation-free string.
    """
    if not text:
        return ""

    # 1. Unicode normalization (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Strip bracketed acoustic tokens or metadata
    text = _BRACKETED_PATTERN.sub(" ", text)

    # 4. Standardize quotes and apostrophes
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')

    # 5. Convert hyphens/slashes to spaces (e.g., "twenty-one" -> "twenty one")
    text = _PUNCT_TO_SPACE.sub(" ", text)

    # 6. Replace digits 0-20 with words
    text = _NUMBER_PATTERN.sub(_replace_number, text)

    # 7. Strip remaining punctuation
    text = _PUNCT_TO_STRIP.sub("", text)

    # 8. Collapse whitespace
    text = " ".join(text.split())

    return text
