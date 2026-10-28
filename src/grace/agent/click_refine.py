"""Zoom refine for single-mode clicks (cockpit 9b, part C).

Holo's 0-1000 grid is coarse on a wide screen (one step is 1.9 px, and Holo
rounds to multiples of 5, about 9.6 px on 1920). After Holo picks a click point
Grace crops a small full-resolution patch around it, enlarges it, and asks Holo
once more where the centre of its target is inside that patch.

Coordinates only. Holo makes both choices (what to click, and where in the
zoom); Grace crops and does arithmetic. No element ids, names or snapping.
Any failure (not_here, unparseable reply, error, timeout) means: click Holo's
first point unchanged.
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("grace.agent.click_refine")

REFINE_TIMEOUT_S = 5.0
DEFAULT_CROP_PX = 300
DEFAULT_SCALE = 3

_SYSTEM_PROMPT = "You are a GUI pointing assistant. You answer with one JSON object only."


@dataclass
class RefineResult:
    first: tuple[int, int]
    refined: Optional[tuple[int, int]] = None
    rect: Optional[tuple[int, int, int, int]] = None  # left, top, width, height
    raw: str = ""
    reason: str = ""  # ok | not_here | unparseable | error | timeout
    png: Optional[bytes] = None
    elapsed_ms: int = 0

    @property
    def point(self) -> tuple[int, int]:
        """The point to click: the refined one, else Holo's first."""
        return self.refined if self.refined is not None else self.first

    def to_record(self) -> dict[str, Any]:
        return {
            "first_point": list(self.first),
            "refined_point": list(self.refined) if self.refined is not None else None,
            "crop_rect": list(self.rect) if self.rect is not None else None,
            "raw_reply": self.raw,
            "outcome": self.reason,
            "elapsed_ms": self.elapsed_ms,
        }


def _truthy(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return default


def refine_settings(config: Any = None) -> tuple[bool, int, int]:
    """(enabled, crop_px, scale). Only real values on `config` are honoured;
    anything else falls through to the environment, then the defaults."""
    def pick(attr: str, env: str):
        val = getattr(config, attr, None) if config is not None else None
        if isinstance(val, (bool, int, str)):
            return val
        return os.getenv(env)

    def as_int(val: Any, default: int) -> int:
        try:
            n = int(val)
            return n if n > 0 else default
        except (TypeError, ValueError):
            return default

    enabled_raw = pick("click_refine", "CLICK_REFINE")
    enabled = False if enabled_raw is None else _truthy(enabled_raw, False)
    crop = as_int(pick("click_refine_crop_px", "CLICK_REFINE_CROP_PX"), DEFAULT_CROP_PX)
    scale = as_int(pick("click_refine_scale", "CLICK_REFINE_SCALE"), DEFAULT_SCALE)
    return enabled, crop, scale


def crop_rect(px: int, py: int, screen_w: int, screen_h: int, crop_px: int) -> tuple[int, int, int, int]:
    """A crop_px square centred on (px, py), clamped inside the screen.

    Returns (left, top, width, height). A screen smaller than the crop gives
    the whole screen on that axis.
    """
    w = min(crop_px, screen_w)
    h = min(crop_px, screen_h)
    left = min(max(px - crop_px // 2, 0), screen_w - w)
    top = min(max(py - crop_px // 2, 0), screen_h - h)
    return int(left), int(top), int(w), int(h)


def map_back(x: float, y: float, rect: tuple[int, int, int, int]) -> tuple[int, int]:
    """Zoom-grid (0-1000) point back to screen pixels: left + x/1000*width."""
    left, top, w, h = rect
    return int(round(left + x / 1000.0 * w)), int(round(top + y / 1000.0 * h))


def build_prompt(thought: str) -> str:
    from grace.agent.planner import wrap_untrusted

    target = " ".join((thought or "").split()) or "the control you meant to click"
    return (
        f"Your target: {wrap_untrusted(target)}\n"
        "This is a zoomed view around the point you chose. "
        "Point at the exact centre of your target.\n"
        'Reply with JSON only: {"x": .., "y": ..} on the 0-1000 grid of this zoomed image, '
        'or {"not_here": true}.'
    )


def parse_reply(raw: Optional[str]) -> tuple[Optional[tuple[float, float]], str]:
    """(point or None, reason). reason is 'ok', 'not_here' or 'unparseable'."""
    if not raw or not isinstance(raw, str):
        return None, "unparseable"
    for candidate in re.findall(r"\{[^{}]*\}", raw):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("not_here") is True:
            return None, "not_here"
        try:
            x, y = float(data["x"]), float(data["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if x != x or y != y or not (0 <= x <= 1000 and 0 <= y <= 1000):
            continue
        return (x, y), "ok"
    return None, "unparseable"


def grab_region(rect: tuple[int, int, int, int]):
    """Full-resolution capture of just `rect`. The real screen: tests inject
    a fake instead of calling this."""
    import pyautogui

    left, top, w, h = rect
    return pyautogui.screenshot(region=(left, top, w, h))


def _enlarged_png(image: Any, size: tuple[int, int]) -> bytes:
    from PIL import Image

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    big = image.convert("RGB").resize(size, resample)
    buf = io.BytesIO()
    big.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


async def refine_click(
    llm: Any,
    point: tuple[int, int],
    screen_size: tuple[int, int],
    thought: str,
    *,
    crop_px: int = DEFAULT_CROP_PX,
    scale: int = DEFAULT_SCALE,
    capture: Optional[Callable[[tuple[int, int, int, int]], Any]] = None,
    timeout: float = REFINE_TIMEOUT_S,
) -> RefineResult:
    """One zoomed second look at a click point. Never raises."""
    started = time.monotonic()
    result = RefineResult(first=(int(point[0]), int(point[1])))
    try:
        rect = crop_rect(result.first[0], result.first[1], screen_size[0], screen_size[1], crop_px)
        result.rect = rect
        grab = capture or grab_region
        prompt = build_prompt(thought)

        async def _go() -> str:
            image = await asyncio.to_thread(grab, rect)
            png = await asyncio.to_thread(_enlarged_png, image, (rect[2] * scale, rect[3] * scale))
            result.png = png
            raw = await llm.generate_text(
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=256,
                image_b64=base64.b64encode(png).decode("utf-8"),
            )
            return raw or ""

        raw = await asyncio.wait_for(_go(), timeout=timeout)
        result.raw = str(raw)
        parsed, reason = parse_reply(result.raw)
        result.reason = reason
        if parsed is not None:
            sx, sy = map_back(parsed[0], parsed[1], rect)
            result.refined = (
                min(max(sx, 0), max(screen_size[0] - 1, 0)),
                min(max(sy, 0), max(screen_size[1] - 1, 0)),
            )
    except asyncio.TimeoutError:
        result.reason = "timeout"
    except asyncio.CancelledError:
        raise
    except Exception as e:  # capture or model failure: click the first point
        result.reason = "error"
        result.raw = f"{type(e).__name__}: {e}"
    result.elapsed_ms = int((time.monotonic() - started) * 1000)
    logger.info(
        "click refine: first=%s refined=%s rect=%s outcome=%s %dms",
        result.first, result.refined, result.rect, result.reason, result.elapsed_ms,
    )
    return result
