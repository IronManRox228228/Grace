"""The grounder: turns a described target into screen coordinates.

Works with any vision model. Given a screenshot and a description of one
control, the model identifies where that control is. It is called only when
the element graph could not resolve the planner's target, so on a well-behaved
accessibility tree it is never called at all.
"""

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("grace.agent.grounder")

# Canonical vision grounding instruction for any vision model.
GROUNDER_SYSTEM_PROMPT = """You are a GUI visual grounding assistant. You are shown a screenshot and asked to locate a single element.
Locate the requested element and return its center coordinates.
Respond in this format:
Thought: <brief description of where the element is>
Action: click(point=(x, y))

If the element is not visible or cannot be found, respond with:
Action: wait()"""


@dataclass
class GroundedPoint:
    """A located target, in screen coordinates."""

    x: int
    y: int
    thought: str = ""
    source: str = "vision"


def parse_grounder_response(response_text: Optional[str]) -> Optional[tuple[float, float, str]]:
    """Parse raw text from a vision grounder into (x, y, thought).

    Robustly handles:
    - Thought: ... Action: click(point=(x, y))
    - Action: click((x, y)) or click(x, y)
    - Action: click(start_box='(x, y)') or box tokens
    - JSON: {"point": [x, y]} or {"x": x, "y": y}
    - Action: wait() or not visible -> None
    - Malformed output -> None
    """
    if not response_text or not isinstance(response_text, str):
        return None

    clean_text = response_text.strip()
    if not clean_text:
        return None

    # Separate thought if present
    thought = ""
    if "Thought:" in clean_text:
        parts = clean_text.split("Action:", 1) if "Action:" in clean_text else clean_text.split("\n", 1)
        thought = parts[0].replace("Thought:", "").strip()

    # Explicit wait or element not found
    if re.search(r"\bwait\s*\(\s*\)", clean_text, re.IGNORECASE):
        return None

    # 1. Try parsing JSON if present
    json_candidates = re.findall(r"\{[^{}]*\}", clean_text)
    for candidate in json_candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                candidate_thought = data.get("thought", thought)
                pt = data.get("point") or data.get("coordinates") or data.get("click")
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    return float(pt[0]), float(pt[1]), str(candidate_thought)
                if "x" in data and "y" in data:
                    return float(data["x"]), float(data["y"]), str(candidate_thought)
        except Exception:
            pass

    _COORD = r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"

    # 2. Action click / point pattern
    m = re.search(
        rf"(?:click|move|point)\s*\(\s*(?:(?:point|start_box|target|location|coord|coordinates)\s*=\s*)?['\"]?(?:<\|box_start\|>)?\s*\(?\s*{_COORD}\s*\)?\s*(?:<\|box_end\|>)?['\"]?\s*\)",
        clean_text,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1)), float(m.group(2)), thought

    # 3. Box tokens: <|box_start|>(x, y)<|box_end|>
    m = re.search(rf"<\|box_start\|>\s*\(?\s*{_COORD}\s*\)?\s*<\|box_end\|>", clean_text)
    if m:
        return float(m.group(1)), float(m.group(2)), thought

    # 4. Explicit point/coords: point: (x, y) or point = [x, y]
    m = re.search(rf"(?:point|coord|coordinate|coordinates|location)\s*[:=]\s*[\(\[]\s*{_COORD}\s*[\)\]]", clean_text, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2)), thought

    # 5. Bare coordinate pair in Action line: Action: (x, y) or Action: [x, y]
    m = re.search(rf"Action:\s*(?:click\s*)?[\(\[]\s*{_COORD}\s*[\)\]]", clean_text, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2)), thought

    return None


class Grounder:
    """Locates a described control in a screenshot via vision LLM."""

    def __init__(self, vision_llm, on_demand_start=None, coordinate_format: str = "0-1000"):
        self._llm = vision_llm
        self._on_demand_start = on_demand_start
        self._coordinate_format = coordinate_format
        self._started = False
        self._start_lock = asyncio.Lock()
        self.calls_made = 0

    async def _ensure_backend(self) -> bool:
        if self._started or self._on_demand_start is None:
            return True
        async with self._start_lock:
            if self._started:
                return True
            try:
                result = self._on_demand_start()
                if asyncio.iscoroutine(result):
                    result = await result
                self._started = bool(result) if result is not None else True
            except Exception as e:
                logger.error(f"Could not start the grounding backend: {e}")
                self._started = False
            return self._started

    async def locate(
        self,
        description: str,
        png_bytes: bytes,
        image_size: tuple[int, int],
        screen_size: tuple[int, int],
    ) -> Optional[GroundedPoint]:
        """Find `description` in the screenshot and return screen coordinates."""
        if not png_bytes or not description:
            return None
        if not await self._ensure_backend():
            return None

        self.calls_made += 1
        prompt = f"Locate this element: {description}"
        try:
            raw = await self._llm.generate_text(
                prompt=prompt,
                system_prompt=GROUNDER_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=2048,
                image_b64=base64.b64encode(png_bytes).decode("utf-8"),
            )
        except Exception as e:
            logger.error(f"Grounding request failed: {e}")
            return None

        if not raw:
            logger.warning(f"Grounder returned nothing for '{description}'")
            return None

        parsed = parse_grounder_response(raw)
        if not parsed:
            logger.warning(f"Grounder output was not parseable: {raw[:150]}")
            return None

        x, y, thought = parsed
        scaled = scale_to_screen(x, y, image_size, screen_size, self._coordinate_format)
        if scaled is None:
            logger.warning(f"Grounder coordinates ({x}, {y}) out of bounds for format '{self._coordinate_format}'")
            return None

        sx, sy = scaled
        logger.info(f"Grounder located '{description}' at ({x},{y}) [{self._coordinate_format}] -> screen ({sx},{sy})")
        return GroundedPoint(x=sx, y=sy, thought=thought)


def scale_to_screen(
    x: float,
    y: float,
    image_size: tuple[int, int],
    screen_size: tuple[int, int],
    coordinate_format: str = "0-1000",
) -> Optional[tuple[int, int]]:
    """Map a coordinate from the model back to the physical screen.

    Supported coordinate_format values:
    - '0-1000' (or 'norm_1000'): normalized integers in [0, 1000]
    - '0-1' (or 'norm_1'): normalized floats in [0.0, 1.0]
    - 'pixels': image pixels in [0, width] x [0, height]
    """
    img_w, img_h = image_size
    scr_w, scr_h = screen_size
    if not img_w or not img_h or not scr_w or not scr_h:
        return int(round(x)), int(round(y))

    fmt = (coordinate_format or "0-1000").strip().lower()

    if fmt in ("0-1000", "norm_1000", "1000"):
        if x < 0 or x > 1000 or y < 0 or y > 1000:
            return None
        return int(round(x * scr_w / 1000.0)), int(round(y * scr_h / 1000.0))

    if fmt in ("0-1", "norm_1", "normalized", "percent"):
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            return None
        return int(round(x * scr_w)), int(round(y * scr_h))

    # Pixels format
    if x < 0 or y < 0:
        return None

    if x > img_w or y > img_h:
        # Overflow net: if model emitted 0-1000 coordinates despite pixel prompt
        if 0 <= x <= 1000 and 0 <= y <= 1000:
            logger.warning(
                f"Grounder returned ({x},{y}) outside the {img_w}x{img_h} image it was "
                f"given; treating as 0-1000 normalised."
            )
            return int(round(x * scr_w / 1000.0)), int(round(y * scr_h / 1000.0))
        return None

    return int(round(x * scr_w / img_w)), int(round(y * scr_h / img_h))


def describe_target(params: dict[str, Any]) -> str:
    """Build a natural-language description from the planner's params."""
    described = params.get("describe")
    if described:
        return str(described)

    name = params.get("target_name") or params.get("name") or ""
    role = params.get("role") or ""
    frame = params.get("frame") or ""

    parts = []
    if role:
        parts.append(str(role))
    if name:
        parts.append(f"labelled '{name}'")
    if frame == "chrome":
        parts.append("in the browser's toolbar")
    elif frame == "page":
        parts.append("in the web page content")

    return " ".join(parts) if parts else str(name)
