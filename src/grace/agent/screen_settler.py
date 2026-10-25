"""Screen settling detector for web pages and applications.

After an action that can change the page (such as Return/Enter, click,
type plus submit, open_app, activate), the desktop or web application
(e.g. YouTube Music, SPAs) may still be transitioning/loading.

This module polls a cheap screen signal every 250ms (downscaled grayscale
hash/pixels, plus UIA element count if available) and proceeds when two
consecutive samples are stable (pixel diff < 1%). If not stable after a
configurable timeout (default 3000ms), it proceeds anyway and notes
"screen may still be loading" in the planner's last-step result.
"""

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

logger = logging.getLogger("grace.agent.screen_settler")

DEFAULT_SETTLE_MAX_MS = 3000
DEFAULT_POLL_INTERVAL_MS = 250
DEFAULT_PIXEL_DIFF_THRESHOLD = 0.01  # 1%


def action_can_change_page(action: Optional[str], params: Optional[dict[str, Any]] = None) -> bool:
    """Check whether a tool action is capable of triggering page transitions/loads."""
    if not action:
        return False
    act = action.lower().strip()
    p = params or {}

    # Key presses like Return / Enter often submit searches, navigate forms, etc.
    if act in ("cua_press_key", "press_key"):
        key = str(p.get("key", "")).lower().strip()
        return key in ("return", "enter", "enter_key")

    # Mouse clicks often navigate links or trigger actions
    if act in ("cua_click", "click"):
        return True

    # Text typing with submit/enter flag
    if act in ("cua_type_text", "type_text"):
        return bool(
            p.get("submit")
            or p.get("press_enter")
            or p.get("enter")
            or p.get("press_return")
        )

    # Launching apps or URLs
    if act in ("open_app", "cua_open_app", "launch"):
        return True

    # Switching / activating windows
    if act in ("cua_activate", "activate"):
        return True

    return False


def pixel_diff_ratio(b1: bytes, b2: bytes) -> float:
    """Calculate the average absolute difference between two byte buffers, normalised to 0.0 - 1.0.

    Returns 0.0 if both are identical, up to 1.0 if maximally different.
    """
    if not b1 and not b2:
        return 0.0
    if not b1 or not b2 or len(b1) != len(b2):
        return 1.0
    total_diff = sum(abs(x - y) for x, y in zip(b1, b2))
    return total_diff / (255.0 * len(b1))


@dataclass
class SettleResult:
    """Outcome of waiting for the screen to settle."""

    settled: bool
    elapsed_ms: float
    samples_taken: int
    note: str = ""


class ScreenSettler:
    """Polls cheap screen state until two consecutive reads are stable or timeout expires."""

    def __init__(
        self,
        max_ms: Optional[int] = None,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        pixel_diff_threshold: float = DEFAULT_PIXEL_DIFF_THRESHOLD,
        sample_fn: Optional[Callable] = None,
        sleep_fn: Optional[Callable] = None,
    ):
        if max_ms is not None:
            self.max_ms = max_ms
        else:
            try:
                from grace.config import Config
                self.max_ms = Config().observe_settle_max_ms
            except Exception:
                self.max_ms = DEFAULT_SETTLE_MAX_MS

        self.poll_interval_ms = poll_interval_ms
        self.pixel_diff_threshold = pixel_diff_threshold
        self._sample_fn = sample_fn or self._default_sample
        self._sleep_fn = sleep_fn or asyncio.sleep

    async def _get_sample(self) -> tuple[bytes, Optional[int]]:
        """Retrieve a cheap sample as (downscaled_bytes, optional_element_count)."""
        res = self._sample_fn()
        if inspect.isawaitable(res):
            res = await res
        if isinstance(res, tuple):
            pixels = res[0] if len(res) > 0 else b""
            count = res[1] if len(res) > 1 else None
            return (pixels if isinstance(pixels, bytes) else bytes(pixels)), count
        if isinstance(res, (bytes, bytearray)):
            return bytes(res), None
        return b"", None

    def _default_sample(self) -> tuple[bytes, Optional[int]]:
        """Live capture: downscaled 64x64 grayscale desktop frame + fast UIA count."""
        pixels = b""
        try:
            import mss
            from PIL import Image

            with mss.MSS() as sct:
                mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                raw = sct.grab(mon)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                thumb = img.convert("L").resize((64, 64))
                pixels = thumb.tobytes()
        except Exception:
            try:
                import pyautogui
                img = pyautogui.screenshot()
                thumb = img.convert("L").resize((64, 64))
                pixels = thumb.tobytes()
            except Exception:
                pixels = b""

        uia_count = None
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                from grace.perception.uia_provider import _get_uia
                uia = _get_uia()
                if uia and hasattr(uia, "cache_request"):
                    elem = uia.uia.ElementFromHandleBuildCache(hwnd, uia.cache_request)
                    if elem:
                        children = elem.GetCachedChildren()
                        if children:
                            uia_count = children.Length
        except Exception:
            uia_count = None

        return pixels, uia_count

    def _log_trace(self, res: SettleResult) -> None:
        """Log the settle time to the active turn trace."""
        try:
            from grace.util.timing import mark
            mark(
                "settle_screen",
                round(res.elapsed_ms, 1),
                detail=f"settled={res.settled} in {res.elapsed_ms:.1f}ms ({res.samples_taken} samples)",
            )
        except Exception:
            pass

    async def wait_for_settle(self) -> SettleResult:
        """Wait until two consecutive samples are within the pixel diff threshold or cap hits."""
        start_time = time.monotonic()

        sample1 = await self._get_sample()
        samples_taken = 1

        if self.max_ms <= 0:
            res = SettleResult(settled=True, elapsed_ms=0.0, samples_taken=1)
            self._log_trace(res)
            return res

        while True:
            now = time.monotonic()
            elapsed_ms = (now - start_time) * 1000.0

            if elapsed_ms >= self.max_ms:
                res = SettleResult(
                    settled=False,
                    elapsed_ms=elapsed_ms,
                    samples_taken=samples_taken,
                    note="screen may still be loading",
                )
                self._log_trace(res)
                return res

            remaining_ms = self.max_ms - elapsed_ms
            wait_s = min(self.poll_interval_ms / 1000.0, max(0.0, remaining_ms / 1000.0))
            if wait_s > 0:
                sleep_res = self._sleep_fn(wait_s)
                if inspect.isawaitable(sleep_res):
                    await sleep_res

            sample2 = await self._get_sample()
            samples_taken += 1

            now = time.monotonic()
            elapsed_ms = (now - start_time) * 1000.0

            pix_diff = pixel_diff_ratio(sample1[0], sample2[0])
            uia_stable = (sample1[1] is None or sample2[1] is None or sample1[1] == sample2[1])

            if pix_diff < self.pixel_diff_threshold and uia_stable:
                res = SettleResult(
                    settled=True,
                    elapsed_ms=elapsed_ms,
                    samples_taken=samples_taken,
                    note="",
                )
                self._log_trace(res)
                return res

            if elapsed_ms >= self.max_ms:
                res = SettleResult(
                    settled=False,
                    elapsed_ms=elapsed_ms,
                    samples_taken=samples_taken,
                    note="screen may still be loading",
                )
                self._log_trace(res)
                return res

            sample1 = sample2
