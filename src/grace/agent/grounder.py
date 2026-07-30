"""The grounder: turns a described target into screen coordinates.

UI-TARS is a GUI grounding model. Given a screenshot and a description of one
control, it is very good at saying where that control is. It was previously
also being asked to decompose goals and pick from Grace's 26-tool schema, which
is not what it was trained for and is where the confusion came from.

Here it does one job and gets its native prompt and action space to do it in.
It is called only when the element graph could not resolve the planner's
target, so on a well-behaved accessibility tree it is never called at all.
"""

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Any, Optional

from grace.agent.ui_tars_parser import UITarsParser

logger = logging.getLogger("grace.agent.grounder")

# The canonical UI-TARS grounding instruction. Deliberately not Grace's tool
# schema: the model was trained against this action space.
GROUNDER_SYSTEM_PROMPT = """You are a GUI grounding model. You are shown a screenshot and asked to locate a single element.

Respond in exactly this format and nothing else:

Thought: <one short sentence identifying the element>
Action: click(start_box='(x,y)')

x and y are pixel coordinates in the image you were given, measured from its top-left corner. Point at the centre of the element. If the element is not visible in the screenshot, respond with Action: wait()."""


@dataclass
class GroundedPoint:
    """A located target, in screen coordinates."""

    x: int
    y: int
    thought: str = ""
    source: str = "ui_tars"


class Grounder:
    """Locates a described control in a screenshot via UI-TARS."""

    def __init__(self, vision_llm, on_demand_start=None):
        self._llm = vision_llm
        # Called before the first grounding request so llama-server can be
        # started lazily instead of blocking startup for up to 120 seconds.
        self._on_demand_start = on_demand_start
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
        """Find `description` in the screenshot and return screen coordinates.

        image_size is the size of the PNG actually sent; screen_size is the real
        desktop. The model answers in image space, so the result must be scaled
        back - Grace previously sent a downscaled image and then clicked the
        returned coordinates as if they were screen pixels.
        """
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
                max_tokens=128,
                image_b64=base64.b64encode(png_bytes).decode("utf-8"),
            )
        except Exception as e:
            logger.error(f"Grounding request failed: {e}")
            return None

        if not raw:
            logger.warning(f"Grounder returned nothing for '{description}'")
            return None

        parsed = UITarsParser.parse_response(raw)
        if not parsed:
            logger.warning(f"Grounder output was not parseable: {raw[:150]}")
            return None

        params = parsed.get("params") or {}
        x, y = params.get("x"), params.get("y")
        if x is None or y is None:
            logger.info(f"Grounder could not see '{description}' (action={parsed.get('action')})")
            return None

        sx, sy = scale_to_screen(int(x), int(y), image_size, screen_size)
        logger.info(f"Grounder located '{description}' at image ({x},{y}) -> screen ({sx},{sy})")
        return GroundedPoint(x=sx, y=sy, thought=parsed.get("thought", ""))


def scale_to_screen(
    x: int,
    y: int,
    image_size: tuple[int, int],
    screen_size: tuple[int, int],
) -> tuple[int, int]:
    """Map a coordinate in the sent image back to the physical screen.

    UI-TARS also emits normalised 0-1000 coordinates in some prompt formats;
    those are detected by the coordinate exceeding the image it was given.
    """
    img_w, img_h = image_size
    scr_w, scr_h = screen_size
    if not img_w or not img_h or not scr_w or not scr_h:
        return x, y

    if x > img_w or y > img_h:
        # 0-1000 normalised space rather than image pixels.
        return int(round(x * scr_w / 1000.0)), int(round(y * scr_h / 1000.0))

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
