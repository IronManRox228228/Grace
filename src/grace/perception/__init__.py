"""Structured desktop perception: the element graph the agent reasons over."""

from grace.perception.elements import (
    FRAME_APP,
    FRAME_CHROME,
    FRAME_PAGE,
    ElementNode,
    elements_to_json,
    elements_to_prompt,
)

__all__ = [
    "ElementNode",
    "FRAME_APP",
    "FRAME_CHROME",
    "FRAME_PAGE",
    "elements_to_json",
    "elements_to_prompt",
]
