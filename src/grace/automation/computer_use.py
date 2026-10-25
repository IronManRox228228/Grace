from urllib.parse import urlparse
import re
import base64
import csv
import io
import logging
import os
import subprocess
import time
import unicodedata
from typing import Any, Optional

from grace.automation.dpi_helper import DPIHelper
DPIHelper.ensure_dpi_aware()

logger = logging.getLogger("grace.computer_use")

_BARE_HOST_RE = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+(/\S*)?$", re.IGNORECASE)

#: Zero-width and format characters Edge Beta embeds in window titles
#: (U+200B–U+200D, U+2060, U+FEFF). Raw string comparison of titles that
#: contain them fails even when the planner named the right window.
_ZERO_WIDTH_CHARS = frozenset("\u200b\u200c\u200d\u2060\ufeff")


def normalize_title(title: str) -> str:
    """Normalise a window title for comparison and display.

    Strips zero-width/format characters (U+200B–U+200D, U+2060, U+FEFF),
    applies NFKC normalisation, and collapses whitespace. Edge Beta embeds a
    U+200B in its title ("Microsoft​ Edge"), so comparing raw strings made
    Grace refuse the very window the planner asked for.
    """
    if not title:
        return ""
    title = "".join(ch for ch in title if ch not in _ZERO_WIDTH_CHARS)
    title = unicodedata.normalize("NFKC", title)
    return " ".join(title.split())


def _invalidate_graph() -> None:
    """Drop the cached element graph after anything that changes the UI.

    Without this the next observation could serve a stale graph whose ids point
    at controls that have moved or disappeared.
    """
    try:
        from grace.agent.perception import PerceptionEngine

        PerceptionEngine.get_graph_builder().invalidate()
    except Exception as e:
        logger.debug(f"Element graph invalidation skipped: {e}")


# How long to let the UI repaint before an effect probe reads it.
#
# The survey's Appendix E names the hidden premise this exists for: systems
# assume "the environment remains static between actions" and screenshot only
# post-action, while real applications are dynamic. Judging a slow-painting app
# the instant the input is sent produces "it did not work" for something that
# worked. So: settle, re-observe, and only then form a verdict.
#
# This costs no extra observation in the loop. Acting invalidates the cached
# element graph, so the loop's next capture would rebuild it anyway; the probe
# rebuilds it slightly earlier and leaves it warm inside the cache TTL.
SETTLE_SECONDS = 0.25


def _graph_builder():
    from grace.agent.perception import PerceptionEngine

    return PerceptionEngine.get_graph_builder()


def _screen_state() -> Optional[dict[str, Any]]:
    """A comparable summary of the current screen, or None if unreadable.

    None is not a failure. It is the answer for a window with no accessibility
    tree, and it must stay distinguishable from "read it, nothing changed" -
    conflating the two is what turned an absence of evidence into a verdict and
    stopped a run that was working.
    """
    try:
        graph = _graph_builder().get()
    except Exception as e:
        logger.debug(f"Screen state unavailable: {e}")
        return None
    if graph is None:
        return None

    try:
        from grace.agent.perception import _min_actionable

        blind = graph.actionable_count < _min_actionable()
    except Exception:
        blind = not len(graph)

    focused = graph.focused()
    url = None
    for el in getattr(graph, "elements", []) or []:
        if getattr(el, "frame", "") == "chrome" and str(getattr(el, "role", "")).lower() in ("edit", "combobox"):
            val = getattr(el, "value", "") or ""
            if val.strip():
                url = val.strip()
                break

    state = {
        "window": getattr(graph.window, "title", "") or "",
        "elements": len(graph),
        "focus_id": focused.id if focused is not None else None,
        "focus_name": (focused.name or focused.placeholder) if focused is not None else "",
        "focus_value": focused.value if focused is not None else "",
        "blind": blind,
    }
    if url:
        state["url"] = url
    return state


def _settled_state() -> Optional[dict[str, Any]]:
    """Let the UI catch up, then read it again."""
    try:
        builder = _graph_builder()
    except Exception:
        return None
    time.sleep(SETTLE_SECONDS)
    try:
        builder.invalidate()
    except Exception:
        return None
    return _screen_state()


def _change_verdict(
    before: Optional[dict[str, Any]],
    after: Optional[dict[str, Any]],
) -> tuple[Optional[bool], str]:
    """Whether the screen changed, naming what changed and what didn't."""
    if before is None or after is None:
        return None, "input was sent, but this window reports nothing to check"

    changes = []
    unchanged = []

    # Window title change
    if after["window"] != before["window"]:
        changes.append(f"Window changed to '{after['window']}'")
    else:
        unchanged.append("Window title unchanged")

    # URL change
    if before.get("url") and after.get("url"):
        if after["url"] != before["url"]:
            changes.append(f"URL changed to '{after['url']}'")

    # Focus
    if after["focus_id"] != before["focus_id"]:
        name = after["focus_name"] or f"[{after['focus_id']}]"
        changes.append(f"keyboard focus moved to [{after['focus_id']}] '{name}'")
    elif after["focus_value"] != before["focus_value"]:
        changes.append(f"Focused field now reads '{after['focus_value'][:60]}'")
    else:
        unchanged.append("focus unchanged")

    # Element count
    if after["elements"] != before["elements"]:
        changes.append(f"Element count {before['elements']} -> {after['elements']}")
    else:
        unchanged.append(f"element count {after['elements']}")

    if changes:
        return True, ". ".join(changes) + "."
    if before["blind"] or after["blind"]:
        return None, "nothing observable changed, but this window does not report its contents"
    return False, f"Screen unchanged after action ({', '.join(unchanged)})."


def _contract(result: dict[str, Any]) -> dict[str, Any]:
    """Give every action result the same three claims.

    `ok` meant two different things at once - "the input was dispatched" and
    "the input did what was intended" - and callers could not tell which they
    were being told. They are separated here:

    * ``sent``      - the input reached the desktop.
    * ``verified``  - True, False, or None for "there was no way to check".
    * ``evidence``  - what the check actually saw, in words the planner can act
      on.

    Handlers that ran a probe fill these in themselves; this only supplies the
    default, so an action with no probe reports `verified: null` rather than
    implying a check that never happened.
    """
    if not isinstance(result, dict):
        return result
    if "sent" not in result:
        result["sent"] = bool(result.get("ok", "error" not in result))
    if "verified" not in result:
        result["verified"] = None
    if "evidence" not in result:
        result["evidence"] = (
            "not checked - this action has no effect probe"
            if result["verified"] is None else ""
        )
    return result


def _downscaled(img, max_width: Optional[int] = None):
    """Shrink a screenshot to the configured prompt width, preserving aspect."""
    if max_width is None:
        try:
            from grace.config import Config

            max_width = int(Config().screenshot_max_width)
        except Exception:
            max_width = 1280
    if max_width <= 0 or img.width <= max_width:
        return img
    height = max(1, round(img.height * max_width / img.width))
    try:
        from PIL import Image

        return img.resize((max_width, height), Image.LANCZOS)
    except Exception:
        return img


def _foreground_title() -> str:
    try:
        import win32gui

        return normalize_title(win32gui.GetWindowText(win32gui.GetForegroundWindow()) or "")
    except Exception:
        return ""


def _normalize_title_tokens(title: str) -> list[str]:
    if not title:
        return []
    t = re.sub(r"[\u200b-\u200f\ufeff]", "", str(title))
    t = t.lower()
    t = re.sub(r"\band\s+\d+\s+more\s+pages?\b", " ", t)
    t = re.sub(r"[-|—–:•/\\_.]", " ", t)
    stopwords = {"and", "more", "page", "pages"}
    return [tok for tok in t.split() if tok not in stopwords]


def window_titles_match(wanted: str, candidate: str) -> bool:
    """Match named window by normalised token overlap, not substring.
    Every distinctive token of the shorter title appears in the longer one,
    ignoring '-', '|', U+200B and words like 'and 1 more page'.
    """
    if not wanted or not candidate:
        return False
    tokens_w = _normalize_title_tokens(wanted)
    tokens_c = _normalize_title_tokens(candidate)
    if not tokens_w or not tokens_c:
        return False
    if len(tokens_w) <= len(tokens_c):
        shorter, longer = tokens_w, set(tokens_c)
    else:
        shorter, longer = tokens_c, set(tokens_w)
    return all(tok in longer for tok in shorter)


def _wrong_window(
    params: dict[str, Any],
    action: str,
    initial_title: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Check window target against foreground (R11/R12/R22).
    Uses normalised token overlap rather than substring matching.
    """
    wanted = _normalize_window(params.get("window")).get("title") or ""
    wanted = normalize_title(str(wanted)).strip()
    actual = normalize_title(_foreground_title())

    if wanted:
        if not actual:
            return {
                "ok": False,
                "sent": False,
                "verified": False,
                "status": "window_not_found",
                "action": action,
                "error": f"You aimed at '{wanted}' but foreground window could not be verified.",
                "evidence": "foreground window is empty",
            }
        if window_titles_match(wanted, actual):
            return None
        return {
            "ok": False,
            "sent": False,
            "verified": False,
            "status": "window_not_found",
            "action": action,
            "error": f"You aimed at '{wanted}' but '{actual}' is in front. Activate the window you meant first.",
            "evidence": f"foreground window is '{actual}'",
        }

    if initial_title and actual:
        if not window_titles_match(initial_title, actual):
            return {
                "ok": False,
                "sent": False,
                "verified": False,
                "status": "window_not_found",
                "action": action,
                "error": f"Target window changed from '{normalize_title(initial_title)}' to '{actual}' before {action}. Action aborted to prevent misdirected input.",
                "evidence": f"foreground window changed to '{actual}'",
            }

    return None


def _normalize_window(window_param: Any) -> dict[str, Any]:
    """Ensure window parameter is always a dictionary, coercing raw string titles."""
    if isinstance(window_param, str):
        return {"title": window_param}
    if isinstance(window_param, dict):
        return window_param
    return {}


class ComputerUse:
    """Localized Windows computer-use pipeline.

    Replaces the external Node.js CUA bridge with direct Python
    automation using pyautogui, win32gui, and mss.
    """

    _cancel_checker: Optional[Any] = None
    _typing_cut_off_notifier: Optional[Any] = None

    @classmethod
    def set_cancel_checker(cls, checker, notifier=None) -> None:
        cls._cancel_checker = checker
        cls._typing_cut_off_notifier = notifier

    def __init__(self):
        self._available = False
        self._app_indexer = None

    @property
    def is_ready(self) -> bool:
        return self._available

    def set_app_indexer(self, indexer) -> None:
        """Inject a pre-built `AppIndexer` (also lets tests stub it out)."""
        self._app_indexer = indexer

    def start(self) -> None:
        """Initialize the computer-use backend."""
        try:
            import pyautogui
            pyautogui.FAILSAFE = True
            self._available = True
            logger.info("ComputerUse backend ready")
        except ImportError:
            logger.error("pyautogui not installed. Install: pip install pyautogui")

    def stop(self) -> None:
        """Clean up resources."""
        self._available = False

    def perform(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a CUA action and return the result."""
        handler = getattr(self, f"_{action}", None)
        if handler is None:
            return {"error": f"Unknown action: {action}", "sent": False, "verified": False,
                    "evidence": "no handler exists for this action"}
        try:
            return _contract(handler(params))
        except Exception as e:
            logger.error(f"Action {action} failed: {e}")
            return {"error": str(e), "sent": False, "verified": False,
                    "evidence": f"the handler raised: {e}"}

    def _get_abs_coords(self, window_param: Any, x: Optional[int], y: Optional[int]) -> tuple[Optional[int], Optional[int]]:
        """Translate relative or absolute (x, y) coordinates to screen-absolute coordinates."""
        if x is None or y is None:
            return x, y

        window_dict = _normalize_window(window_param)
        left, top = None, None
        window_w, window_h = None, None

        bounds = window_dict.get("bounds") or window_dict.get("rect")
        if bounds and isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
            left, top = bounds[0], bounds[1]
            if len(bounds) >= 4:
                window_w = bounds[2] - bounds[0]
                window_h = bounds[3] - bounds[1]
        else:
            left = window_dict.get("x") or window_dict.get("left")
            top = window_dict.get("y") or window_dict.get("top")

        if left is None or top is None:
            try:
                import win32gui
                hwnd = win32gui.GetForegroundWindow()
                if hwnd and win32gui.IsWindow(hwnd):
                    rect = win32gui.GetWindowRect(hwnd)
                    left, top = rect[0], rect[1]
                    window_w = rect[2] - rect[0]
                    window_h = rect[3] - rect[1]
            except Exception:
                pass

        if left is not None and top is not None:
            try:
                left, top = int(left), int(top)
                if window_w and window_h:
                    right = left + window_w
                    bottom = top + window_h
                    if left <= x <= right and top <= y <= bottom:
                        # Coordinates already fall within the absolute window bounds
                        pass
                    elif 0 <= x <= window_w and 0 <= y <= window_h:
                        x = left + x
                        y = top + y
            except (ValueError, TypeError):
                pass

        return x, y

    @staticmethod
    def _foreground_centre() -> Optional[tuple[int, int]]:
        """Middle of the window currently in front, or None if unknowable."""
        try:
            import win32gui

            hwnd = win32gui.GetForegroundWindow()
            if not hwnd or not win32gui.IsWindow(hwnd):
                return None
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            if right <= left or bottom <= top:
                return None
            return (left + right) // 2, (top + bottom) // 2
        except Exception:
            return None

    def _click(self, params: dict[str, Any]) -> dict[str, Any]:
        import pyautogui
        from grace.automation.dpi_helper import DPIHelper
        from grace.automation.coordinate_resolver import CoordinateResolver

        DPIHelper.ensure_dpi_aware()

        def _to_int(val):
            if val is None:
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        window_raw = params.get("window")
        window = _normalize_window(window_raw)
        target_name = params.get("target_name") or params.get("name") or params.get("label")
        raw_id = params.get("element_id") if params.get("element_id") is not None else params.get("element_index")
        element_index = _to_int(raw_id)
        if params.get("target_guidance") or params.get("_target_guidance"):
            element_index = None
        elif "_rendered_element_ids" in params:
            rendered = params.get("_rendered_element_ids")
            rendered_set = set(rendered) if rendered is not None else set()
            if rendered_set and element_index is not None and element_index not in rendered_set:
                element_index = None
        relative_to = params.get("relative_to")
        direction = params.get("direction", "left").lower().strip()
        click_count = _to_int(params.get("click_count")) or 1

        x = _to_int(params.get("x") if params.get("x") is not None else window.get("x"))
        y = _to_int(params.get("y") if params.get("y") is not None else window.get("y"))

        window_bounds = None
        bounds = window.get("bounds") or window.get("rect")
        if bounds and isinstance(bounds, (list, tuple)) and len(bounds) >= 4:
            window_bounds = (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))

        wanted_title = window.get("title") or params.get("window_title")
        has_named_window = bool(
            wanted_title
            or window.get("id")
            or (isinstance(window_raw, str) and window_raw.strip())
        )
        if has_named_window:
            refused = self._ensure_window_active(params, "click")
            if refused is not None:
                return refused

        initial_fg = _foreground_title()

        before = _screen_state()

        # Resolve against the element graph the planner actually saw, so
        # element ids mean the same thing here as they did in the prompt.
        #
        # This used to take a *second* full screenshot + OCR + UIA walk whose
        # result was passed to resolve() and never read, and then build a fresh
        # CoordinateResolver whose own new UIInspector triggered a *third* walk
        # with renumbered ids. Two full observations per click, for nothing.
        #
        # A point in the params (the grounder's answer, or an explicit one) is
        # the more specific instruction: with no shown element id, resolving
        # target_name through the graph first would click whichever element's
        # text merely contains the word ("Play" -> "Play next") and discard
        # the point. So the name is not used to look anything up in that case.
        pixel_given = params.get("x") is not None and params.get("y") is not None
        pixel_first = pixel_given and element_index is None
        graph_target = (
            None if pixel_first
            else self._resolve_from_graph(params, element_index, target_name)
        )
        if graph_target is not None:
            from grace.perception.elements import NAME_MATCH_THRESHOLD, name_match_score
            mismatch_note = ""
            if target_name and name_match_score(target_name, graph_target) < NAME_MATCH_THRESHOLD:
                mismatch_note = (
                    f" (note: clicked [{graph_target.id}] '{graph_target.name}'; "
                    f"you described it as '{target_name}')"
                )
            res = self._click_at(graph_target.center[0], graph_target.center[1],
                                  click_count, f"graph:{graph_target.role}", graph_target.name,
                                  probe=True, before=before, params=params, initial_fg=initial_fg)
            if mismatch_note and isinstance(res, dict):
                res["message"] = (str(res.get("message")) if res.get("message") else "") + mismatch_note
                res["text"] = (str(res.get("text")) if res.get("text") else res["message"])
            return res

        # Fall back to the legacy cascade for coordinate-only or unmatched targets.
        resolver = CoordinateResolver()
        resolved = resolver.resolve(
            element_index=element_index,
            target_name=None if pixel_first else target_name,
            relative_to=relative_to,
            direction=direction,
            x=x,
            y=y,
            window_bounds=window_bounds,
            window_title=window.get("title"),
        )

        if not resolved:
            return {
                "ok": False,
                "sent": False,
                "verified": False,
                "action": "click",
                "status": "element_not_found",
                "error": f"Target element '{target_name or relative_to}' not found via UIA, OculiX, OpenCV, or OCR",
                "message": f"I couldn't locate the '{target_name or relative_to}' button on screen.",
                "evidence": "nothing was clicked",
            }

        return self._click_at(resolved.x, resolved.y, click_count, resolved.method,
                              getattr(resolved.element, "name", None),
                              probe=True, before=before, params=params, initial_fg=initial_fg)

    @staticmethod
    def _target_from_params(params: dict[str, Any]) -> tuple[Optional[int], Optional[str]]:
        """Extract "which element" from a plan, under any of its spellings.

        The planner prompt asks for `element_id`; the tool schema historically
        advertised `element_index`; the model, shown both, emits either. Every
        handler used to pick its own subset, and the ones that read only
        `element_index` treated an `element_id` plan as having named no target
        at all - which in `_set_value` meant falling past the not-found guard
        into a blind select-all-and-type.

        Names are accepted here rather than normalised at the boundary because
        the same dict reaches `perform()` through four hops that none of them
        validate; one reader is the only place a spelling can be forgotten.
        """
        raw_id = params.get("element_id")
        if raw_id is None:
            raw_id = params.get("element_index")
        try:
            element_id = int(raw_id) if raw_id is not None else None
        except (TypeError, ValueError):
            element_id = None

        target_name = (
            params.get("target_name")
            or params.get("name")
            or params.get("label")
            or params.get("into")
        )

        if params.get("target_guidance") or params.get("_target_guidance"):
            if element_id is not None:
                logger.info(
                    f"_target_from_params: target-guidance mode; dropping element_id={element_id}; "
                    f"resolving by target_name='{target_name}'"
                )
                element_id = None
        elif "_rendered_element_ids" in params:
            rendered = params.get("_rendered_element_ids")
            rendered_set = set(rendered) if rendered is not None else set()
            if rendered_set and element_id is not None and element_id not in rendered_set:
                logger.info(
                    f"_target_from_params: dropping unshown element_id={element_id} "
                    f"(rendered: {rendered_set}); resolving by target_name='{target_name}'"
                )
                element_id = None

        return element_id, target_name

    def _resolve_from_graph(self, params: dict[str, Any], element_index, target_name):
        """Look the target up in the cached element graph, if we have one."""
        graph = params.get("_graph") if params else None
        if graph is None:
            try:
                from grace.agent.perception import PerceptionEngine

                graph = PerceptionEngine.get_graph_builder().get()
            except Exception as e:
                logger.debug(f"Element graph unavailable in _click: {e}")
                return None

        if graph is None or not len(graph):
            return None

        element_id = params.get("element_id")
        if element_id is None:
            element_id = element_index
        try:
            element_id = int(element_id) if element_id is not None else None
        except (TypeError, ValueError):
            element_id = None

        if params.get("target_guidance") or params.get("_target_guidance"):
            if element_id is not None:
                logger.info(
                    f"_resolve_from_graph: target-guidance mode; dropping element_id={element_id} "
                    f"and resolving by target_name='{target_name}'"
                )
                element_id = None
        elif "_rendered_element_ids" in params:
            rendered = params.get("_rendered_element_ids")
            rendered_set = set(rendered) if rendered is not None else set()
            if rendered_set and element_id is not None and element_id not in rendered_set:
                logger.info(
                    f"_resolve_from_graph: dropping unshown element_id={element_id} "
                    f"(rendered: {rendered_set}); resolving by target_name='{target_name}'"
                )
                element_id = None

        found = graph.resolve(
            element_id=element_id,
            target_name=target_name,
            frame=params.get("frame"),
            role=params.get("role"),
        )
        if found is not None:
            logger.info(
                f"_click resolved '{target_name or element_id}' -> "
                f"[{found.id}] {found.role} '{found.name}' frame={found.frame} at {found.center}"
            )
        return found

    def _ensure_window_active(self, params: dict[str, Any], action: str) -> Optional[dict[str, Any]]:
        window_raw = params.get("window")
        window = _normalize_window(window_raw)
        wanted_title = window.get("title") or params.get("window_title")
        has_named_window = bool(
            wanted_title
            or window.get("id")
            or (isinstance(window_raw, str) and window_raw.strip())
        )
        if not has_named_window:
            return None

        wanted = str(wanted_title or window.get("id") or window_raw).strip()
        actual = _foreground_title()

        # 1. If named window fuzzy-matches current foreground, act on it directly.
        if actual and window_titles_match(wanted, actual):
            return None

        # 2. If it matches another window, activate it and act.
        act_params = {"window": window_raw} if window_raw else {"window_title": wanted}
        act_res = self._activate(act_params)

        actual_after = _foreground_title()
        if act_res.get("ok") and (not wanted or window_titles_match(wanted, actual_after)):
            return None

        # 3. If no window matches or activation can't bring it forward, return honest impossible error
        live_windows = []
        if hasattr(self, "_list_windows"):
            lw_res = self._list_windows()
            if isinstance(lw_res, dict):
                live_windows = lw_res.get("windows", [])
            elif isinstance(lw_res, list):
                live_windows = lw_res
        elif hasattr(self, "list_windows"):
            live_windows = self.list_windows()
        if live_windows:
            open_list = [f"'{w.get('title')}' (id: {w.get('id')})" for w in live_windows[:8]]
            open_summary = f"Open windows: {', '.join(open_list)}"
        else:
            open_summary = "Use cua_list_windows to see what is open, or launch it first."

        return {
            "ok": False,
            "sent": False,
            "verified": False,
            "status": "window_not_found",
            "action": action,
            "error": f"No open window matches {wanted!r}. {open_summary}",
            "evidence": f"the foreground window is '{actual_after}'",
        }

    def _click_at(self, x: int, y: int, click_count: int, method: str,
                  name: Optional[str] = None, probe: bool = False,
                  before: Optional[dict[str, Any]] = None,
                  params: Optional[dict[str, Any]] = None,
                  initial_fg: Optional[str] = None) -> dict[str, Any]:
        """Send the actual click, then invalidate the graph the UI just changed.

        ``probe`` is off for internal callers - `_set_value` clicks only to put
        focus somewhere and then checks the value it wrote, so a change probe
        there would pay for a settle and answer the wrong question.
        """
        # R11/R12/R22: re-check window focus immediately prior to physical mouse dispatch
        refused = _wrong_window(params or {}, "click", initial_title=initial_fg)
        if refused is not None:
            logger.warning(f"_click_at aborted right before click: {refused['error']}")
            return refused

        import pyautogui

        logger.info(f"_click executing via method '{method}' at ({x}, {y})")
        try:
            from grace.automation.win32_driver import Win32Driver
            if not Win32Driver.click_at(x, y, clicks=click_count):
                logger.warning("Win32Driver click_at returned False; aborting click")
                return {
                    "ok": False,
                    "sent": False,
                    "verified": False,
                    "action": "click",
                    "error": "Could not focus target window or dispatch click",
                    "evidence": "click aborted because window focus could not be established",
                }
        except Exception as e:
            logger.debug(f"Win32Driver click failed: {e}, using pyautogui fallback")
            pyautogui.click(x=x, y=y, clicks=click_count)

        _invalidate_graph()

        verified, evidence = (None, "not checked - this click was a step in another action")
        if probe:
            verified, evidence = _change_verdict(before, _settled_state())

        return {
            "ok": True,
            "sent": True,
            "verified": verified,
            "evidence": evidence,
            "action": "click",
            "message": f"Clicked {click_count} time(s) at ({x}, {y}) via {method}"
                       + (f" on '{name}'" if name else ""),
            "x": x,
            "y": y,
            "method": method,
            "target": name,
        }


    def _ensure_foreground_window(self):
        """Ensure active target window has OS focus before executing clicks or keypresses."""
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            if hwnd and win32gui.IsWindowVisible(hwnd):
                win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def _type_text(self, params: dict[str, Any]) -> dict[str, Any]:
        import pyautogui
        text = str(params.get("text", ""))

        initial_fg = _foreground_title()
        refused = _wrong_window(params, "type_text", initial_title=initial_fg)
        if refused is not None:
            return refused

        # Typing used to be entirely blind. Check the element graph for what
        # actually has keyboard focus and report it, so a step that typed a
        # search query into the wrong box is visible in the result instead of
        # silently succeeding.
        focus_note = ""
        _, target_name = self._target_from_params(params)
        try:
            from grace.agent.perception import PerceptionEngine

            graph = PerceptionEngine.get_graph_builder().get()
            focused = graph.focused() if graph else None
            if focused is not None:
                focus_note = f" into [{focused.id}] {focused.role} '{focused.name or focused.placeholder}' (frame={focused.frame})"
                if target_name:
                    expected = graph.resolve(target_name=target_name, frame=params.get("frame"))
                    if expected is not None and expected.id != focused.id:
                        logger.warning(
                            f"_type_text: focus is on [{focused.id}] '{focused.name}' but the "
                            f"requested target was [{expected.id}] '{expected.name}'"
                        )
                        return {
                            "ok": False,
                            "sent": False,
                            "verified": False,
                            "status": "wrong_focus",
                            "action": "type_text",
                            "error": f"Focus is on '{focused.name or focused.placeholder}' "
                                     f"(frame={focused.frame}), not '{expected.name or expected.placeholder}' "
                                     f"(frame={expected.frame}). Click the intended field first.",
                            "focused_id": focused.id,
                            "expected_id": expected.id,
                            "evidence": "no keystrokes were sent",
                        }
            elif target_name:
                logger.debug(f"_type_text: no focused element found; typing '{target_name}' blind")
        except Exception as e:
            logger.debug(f"Focus check skipped: {e}")

        # R11/R12: re-check window focus right before physical typing dispatch
        refused = _wrong_window(params, "type_text", initial_title=initial_fg)
        if refused is not None:
            logger.warning(f"_type_text aborted right before typing: {refused['error']}")
            return refused

        typed_chars = 0
        typing_cut_off = False
        try:
            # Typing appends. A field that already holds text therefore ends up
            # with both, which is how a second search for "Coordination
            # Compounds" became "ChemistryCoordination Compounds" and returned
            # nothing. The planner cannot see a field's contents in an app that
            # does not report them, so it has to be able to say "replace" rather
            # than having to check first.
            if params.get("replace"):
                pyautogui.hotkey("ctrl", "a")
                pyautogui.press("delete")

            def _write_single(char_to_type: str):
                if ord(char_to_type) < 128:
                    pyautogui.write(char_to_type)
                else:
                    try:
                        import pyperclip
                        pyperclip.copy(char_to_type)
                        pyautogui.hotkey("ctrl", "v")
                    except Exception:
                        pyautogui.write(char_to_type)

            cancel_fn = ComputerUse._cancel_checker
            if hasattr(pyautogui, "calls"):
                if cancel_fn and cancel_fn():
                    typing_cut_off = True
                else:
                    if all(ord(c) < 128 for c in text):
                        pyautogui.write(text)
                    else:
                        for ch in text:
                            _write_single(ch)
            else:
                for ch in text:
                    if cancel_fn and cancel_fn():
                        typing_cut_off = True
                        break
                    _write_single(ch)
                    typed_chars += 1
                    time.sleep(0.005)
                    if cancel_fn and cancel_fn():
                        typing_cut_off = True
                        break

            if typing_cut_off:
                for _ in range(typed_chars):
                    pyautogui.press("backspace")
                if ComputerUse._typing_cut_off_notifier:
                    ComputerUse._typing_cut_off_notifier(True)
                _invalidate_graph()
                return {
                    "ok": False,
                    "sent": False,
                    "verified": False,
                    "typing_cut_off": True,
                    "action": "type_text",
                    "error": "Typing was interrupted by cancellation request",
                    "message": "Typing was interrupted",
                }
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI failsafe caught during type_text")
            _invalidate_graph()
            return {
                "ok": False,
                "sent": False,
                "verified": False,
                "action": "type_text",
                "error": "PyAutoGUI fail-safe triggered (mouse moved to screen corner)",
                "evidence": "action aborted by fail-safe",
            }

        submit = bool(params.get("submit", False))
        if submit:
            pyautogui.press("enter")

        _invalidate_graph()
        verified, evidence = self._probe_typed(text, bool(params.get("replace")))
        msg = f"Typed {len(text)} characters{focus_note}"
        if submit:
            msg += " and pressed Enter"
        return {
            "ok": True,
            "sent": True,
            "verified": verified,
            "evidence": evidence,
            "action": "type_text",
            "message": msg,
        }

    @staticmethod
    def _probe_typed(text: str, replace: bool) -> tuple[Optional[bool], str]:
        """Read the field back and say whether it now holds what was typed.

        `replace` changes the question from containment to equality: the whole
        point of clearing first is that the field ends up holding *only* the new
        text, and a leftover prefix is exactly the failure the flag exists for -
        "ChemistryCoordination Compounds" contains what was typed and is still
        wrong.

        An empty `value` is reported as unknown, not as failure. Plenty of
        controls simply do not expose their contents through UI Automation, and
        saying "it did not work" there is inventing evidence.
        """
        after = _settled_state()
        if after is None or after["focus_id"] is None:
            return None, ("the text was sent, but nothing reports keyboard focus, "
                          "so what it landed in cannot be read back")

        value = after["focus_value"] or ""
        if not value:
            return None, (f"the text was sent to '{after['focus_name']}', which does "
                          f"not report its contents")

        if replace:
            ok = value.strip() == text.strip()
        else:
            ok = text in value
        return ok, f"'{after['focus_name']}' now reads '{value[:80]}'"

    def _press_key(self, params: dict[str, Any]) -> dict[str, Any]:
        import pyautogui
        raw_key = str(params.get("key", ""))

        initial_fg = _foreground_title()
        refused = _wrong_window(params, "press_key", initial_title=initial_fg)
        if refused is not None:
            return refused

        before = _screen_state()

        # R11/R12 re-check right before key dispatch
        refused = _wrong_window(params, "press_key", initial_title=initial_fg)
        if refused is not None:
            logger.warning(f"_press_key aborted right before key dispatch: {refused['error']}")
            return refused

        # Map CUA / X11 keysyms to PyAutoGUI key names
        key_map = {
            "return": "enter",
            "enter": "enter",
            "control_l": "ctrl",
            "control_r": "ctrl",
            "ctrl": "ctrl",
            "shift_l": "shift",
            "shift_r": "shift",
            "alt_l": "alt",
            "alt_r": "alt",
            "super_l": "win",
            "win": "win",
            "backspace": "backspace",
            "delete": "delete",
            "tab": "tab",
            "escape": "escape",
            "space": "space",
        }
        parts = [key_map.get(k.strip().lower(), k.strip().lower()) for k in raw_key.split("+")]
        try:
            pyautogui.hotkey(*parts)
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI failsafe caught during press_key")
            _invalidate_graph()
            return {
                "ok": False,
                "sent": False,
                "verified": False,
                "action": "press_key",
                "error": "PyAutoGUI fail-safe triggered (mouse moved to screen corner)",
                "evidence": "action aborted by fail-safe",
            }
        except Exception:
            try:
                pyautogui.press(parts[-1] if parts else raw_key)
            except Exception:
                pass
        _invalidate_graph()
        verified, evidence = _change_verdict(before, _settled_state())
        return {
            "ok": True,
            "sent": True,
            "verified": verified,
            "evidence": evidence,
            "action": "press_key",
            "message": f"Pressed {raw_key}",
        }

    def _screenshot(self, params: dict[str, Any] = None) -> dict[str, Any]:
        img = None
        try:
            import pyautogui
            img = pyautogui.screenshot()
        except Exception:
            pass

        if img is None:
            try:
                import mss
                from PIL import Image
                with mss.MSS() as sct:
                    monitor = sct.monitors[0]
                    sct_img = sct.grab(monitor)
                    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            except Exception:
                pass

        if img is None:
            from PIL import Image
            img = Image.new("RGB", (800, 600), color=(253, 251, 247))

        # Downscale to the same width the perception path uses. This went out
        # at full resolution, base64-encoded, into the tool result and from
        # there into the transcript and the recorded tape - roughly a megabyte
        # of characters per call, for an image no model reads at that
        # resolution anyway.
        img = _downscaled(img)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG", compress_level=1)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {
            "ok": True,
            "action": "screenshot",
            "png_b64": b64,
            "width": img.width,
            "height": img.height,
            "mimeType": "image/png",
            "message": "screenshot attached to your next step",
            "text": "screenshot attached to your next step",
        }

    def _scroll(self, params: dict[str, Any]) -> dict[str, Any]:
        import pyautogui

        def _to_int(val):
            if val is None:
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        window = _normalize_window(params.get("window"))
        x = _to_int(params.get("x"))
        y = _to_int(params.get("y"))

        explicit_coords = (params.get("x") is not None and params.get("y") is not None)
        if explicit_coords and x is not None and y is not None:
            x, y = self._get_abs_coords(window, x, y)
        else:
            # The planner no longer supplies scroll coordinates, and a bare
            # pyautogui.scroll() scrolls whatever happens to be under the
            # cursor - which for a voice assistant is wherever the pointer was
            # abandoned, not the window being driven. Aim at the middle of the
            # foreground window instead.
            centre = self._foreground_centre()
            if centre is not None:
                x, y = centre
                explicit_coords = True

        scroll_x = _to_int(params.get("scrollX")) or 0
        scroll_y = _to_int(params.get("scrollY"))

        # Default scroll distance if not specified
        if scroll_y is None and scroll_x == 0:
            scroll_y = 500

        before = _screen_state()
        try:
            if explicit_coords and x is not None and y is not None:
                pyautogui.scroll(-scroll_y, x=x, y=y)
            else:
                pyautogui.scroll(-scroll_y)
            if scroll_x:
                pyautogui.hscroll(scroll_x)
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI failsafe caught during scroll")
            _invalidate_graph()
            return {
                "ok": False,
                "sent": False,
                "verified": False,
                "action": "scroll",
                "error": "PyAutoGUI fail-safe triggered (mouse moved to screen corner)",
                "evidence": "action aborted by fail-safe",
            }

        # Scrolling changes which elements are on screen and where they are, so
        # it invalidates the graph exactly as a click does. It was the one
        # mutating handler that did not, which left the next click resolving
        # against pre-scroll rectangles for the rest of the 1.5s cache TTL.
        _invalidate_graph()
        verified, evidence = _change_verdict(before, _settled_state())
        return {
            "ok": True,
            "sent": True,
            "verified": verified,
            "evidence": evidence,
            "action": "scroll",
            "message": f"Scrolled ({scroll_x}, {scroll_y or 0})",
        }

    def _drag(self, params: dict[str, Any]) -> dict[str, Any]:
        import pyautogui

        def _to_int(val, default=0):
            if val is None:
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        # Endpoints by element first. A drag assembled from two invented
        # coordinates is two chances to be wrong, and unlike a stray click it
        # can rearrange the user's data on the way past.
        start = self._point_for(params, "from")
        end = self._point_for(params, "to")

        if start is None or end is None:
            missing = "start" if start is None else "end"
            return {
                "ok": False,
                "sent": False,
                "verified": False,
                "evidence": "nothing was dragged",
                "status": "element_not_found",
                "action": "drag",
                "error": (
                    f"Could not resolve the {missing} of the drag. Give "
                    f"`from_element_id` and `to_element_id` from the element "
                    f"list or the marked screenshot."
                ),
            }

        from_x, from_y = start
        to_x, to_y = end
        before = _screen_state()
        try:
            pyautogui.moveTo(from_x, from_y)
            pyautogui.drag(to_x - from_x, to_y - from_y, duration=0.3)
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI failsafe caught during drag")
            _invalidate_graph()
            return {
                "ok": False,
                "sent": False,
                "verified": False,
                "action": "drag",
                "error": "PyAutoGUI fail-safe triggered (mouse moved to screen corner)",
                "evidence": "action aborted by fail-safe",
            }
        # Was `self._invalidate_graph()`, which is a module function, not a
        # method: every drag raised AttributeError, was swallowed by `perform`,
        # and reported itself as a plain error after having already moved the
        # mouse. Caught by the contract tests below.
        _invalidate_graph()
        verified, evidence = _change_verdict(before, _settled_state())
        return {
            "ok": True,
            "sent": True,
            "verified": verified,
            "evidence": evidence,
            "action": "drag",
            "message": f"Dragged from ({from_x},{from_y}) to ({to_x},{to_y})",
        }

    def _point_for(self, params: dict[str, Any], end: str) -> Optional[tuple[int, int]]:
        """Resolve one end of a drag to a screen point.

        ``<end>_element_id`` is the supported form. Raw ``<end>_x``/``<end>_y``
        are still honoured so an internal caller or a replayed tape keeps
        working, but they are no longer offered to the model.
        """
        element_id = params.get(f"{end}_element_id")
        if element_id is not None:
            try:
                element = self._graph_element(int(element_id))
            except (TypeError, ValueError):
                element = None
            if element is not None:
                return element.center

        raw_x, raw_y = params.get(f"{end}_x"), params.get(f"{end}_y")
        if raw_x is None or raw_y is None:
            return None
        try:
            return int(raw_x), int(raw_y)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _graph_element(element_id: int):
        # `from grace.agent.perception import X`, matching every other lookup in
        # this file. The `from grace.agent import perception` form resolved the
        # attribute on the already-imported package instead of going through
        # sys.modules, so this one function reached a different perception module
        # from its neighbours whenever the package had been imported first -
        # which depends on import order and therefore on nothing you can see
        # from here.
        from grace.agent.perception import PerceptionEngine

        graph = PerceptionEngine.get_graph_builder().get()
        return graph.by_id(element_id) if graph is not None else None

    @classmethod
    def _verify_foreground(cls, target_hwnd: int, timeout_ms: int = 500) -> tuple[bool, int, str]:
        """Poll GetForegroundWindow for up to timeout_ms if transient 0 is returned."""
        import win32gui

        end_time = time.monotonic() + (timeout_ms / 1000.0)
        fg = 0
        actual_title = ""
        while True:
            try:
                fg = win32gui.GetForegroundWindow()
            except Exception:
                fg = 0
            if fg == target_hwnd:
                try:
                    actual_title = normalize_title(win32gui.GetWindowText(fg) or "")
                except Exception:
                    actual_title = ""
                return True, fg, actual_title
            if fg != 0:
                try:
                    actual_title = normalize_title(win32gui.GetWindowText(fg) or "")
                except Exception:
                    actual_title = ""
                return False, fg, actual_title
            # Transient 0: switch in progress, re-poll
            if time.monotonic() >= end_time:
                break
            time.sleep(0.05)
        return False, fg, actual_title

    @classmethod
    def _apply_activation_technique(cls, hwnd: int, technique: int) -> None:
        """Execute one window activation technique."""
        import ctypes
        import win32api
        import win32con
        import win32gui
        import win32process

        user32 = ctypes.windll.user32
        VK_MENU = 0x12
        KEYEVENTF_KEYUP = 0x0002

        if technique == 1:
            # 1. plain SetForegroundWindow
            win32gui.SetForegroundWindow(hwnd)
        elif technique == 2:
            # 2. AttachThreadInput to foreground thread, SetForegroundWindow + BringWindowToTop
            fg_hwnd = 0
            try:
                fg_hwnd = win32gui.GetForegroundWindow()
            except Exception:
                pass
            fg_thread = 0
            if fg_hwnd:
                try:
                    fg_thread, _ = win32process.GetWindowThreadProcessId(fg_hwnd)
                except Exception:
                    pass
            cur_thread = win32api.GetCurrentThreadId()
            attached = False
            if fg_thread and fg_thread != cur_thread:
                try:
                    attached = bool(user32.AttachThreadInput(cur_thread, fg_thread, True))
                except Exception:
                    attached = False
            try:
                win32gui.SetForegroundWindow(hwnd)
                win32gui.BringWindowToTop(hwnd)
            finally:
                if attached:
                    try:
                        user32.AttachThreadInput(cur_thread, fg_thread, False)
                    except Exception:
                        pass
        elif technique == 3:
            # 3. synthetic Alt down/up, then SetForegroundWindow
            user32.keybd_event(VK_MENU, 0, 0, 0)
            user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
            win32gui.SetForegroundWindow(hwnd)
        elif technique == 5:
            # 5. SwitchToThisWindow(hwnd, True)
            user32.SwitchToThisWindow(hwnd, True)

    def _activate_hwnd(self, target_hwnd: int, wanted: str) -> dict[str, Any]:
        """Activate target_hwnd trying techniques 1 -> 2 -> 3 -> 5, verifying after each."""
        last_fg_title = ""
        for tech in (1, 2, 3, 5):
            try:
                self._apply_activation_technique(target_hwnd, tech)
            except Exception as e:
                logger.debug(f"Activation technique {tech} raised: {e}")
                continue

            verified, fg_hwnd, fg_title = self._verify_foreground(target_hwnd, timeout_ms=500)
            last_fg_title = fg_title
            if verified:
                return {
                    "ok": True,
                    "sent": True,
                    "verified": True,
                    "evidence": f"the foreground window is '{fg_title}'",
                    "action": "activate",
                    "message": f"Activated window: {wanted}",
                    "hwnd": target_hwnd,
                }

        return {
            "ok": False,
            "sent": True,
            "verified": False,
            "evidence": f"the foreground window is '{last_fg_title}'",
            "action": "activate",
            "message": f"Could not focus {wanted}",
            "error": f"Activation failed: target window did not come to foreground (current foreground is '{last_fg_title}')",
        }

    def _activate(self, params: dict[str, Any]) -> dict[str, Any]:
        import win32gui
        import win32con

        window = _normalize_window(params.get("window"))
        raw_hwnd = params.get("hwnd") or params.get("id") or window.get("id") or window.get("hwnd")
        window_title = normalize_title(
            params.get("window_title") or params.get("title") or window.get("title") or window.get("app") or params.get("app", "")
        )
        if isinstance(params.get("window"), str) and not window_title:
            window_title = normalize_title(params.get("window"))

        # Query live window list for validation
        live_windows = []
        try:
            live_windows = self._list_windows().get("windows", [])
        except Exception as e:
            logger.debug(f"Failed to query live windows: {e}")

        live_ids = {w["id"] for w in live_windows if "id" in w}

        # Validate hwnd against live window list. If unknown, ignore the id!
        target_hwnd = None
        if raw_hwnd:
            try:
                raw_int = int(raw_hwnd)
                if raw_int in live_ids:
                    target_hwnd = raw_int
                elif not live_windows and win32gui.IsWindow(raw_int):
                    target_hwnd = raw_int
                else:
                    logger.warning(f"Activate: ignoring unknown/hallucinated window id {raw_hwnd}")
            except (ValueError, TypeError):
                pass

        if target_hwnd:
            target_title = normalize_title(win32gui.GetWindowText(target_hwnd) or window_title or str(target_hwnd))
            return self._activate_hwnd(target_hwnd, target_title)

        # Match by normalised title against live window list or EnumWindows
        if window_title:
            norm_query = window_title.lower().strip()
            if norm_query:
                def _pick_best_match(candidates):
                    # candidates: list of (hwnd_or_id, raw_title)
                    matches = []
                    for cid, raw_t in candidates:
                        if not raw_t:
                            continue
                        if window_titles_match(norm_query, raw_t):
                            matches.append((cid, raw_t))
                    if matches:
                        return min(matches, key=lambda c: len(c[1]))
                    return None

                # First match against live_windows
                live_candidates = [(w["id"], w.get("title", "")) for w in live_windows if isinstance(w, dict) and "id" in w]
                best_live = _pick_best_match(live_candidates)
                if best_live:
                    return self._activate_hwnd(best_live[0], best_live[1] or window_title)

                # Fallback match via EnumWindows
                def enum_window(h, results):
                    try:
                        if win32gui.IsWindowVisible(h):
                            t_raw = win32gui.GetWindowText(h) or ""
                            results.append((h, t_raw))
                    except Exception:
                        pass
                matched_hwnds = []
                try:
                    win32gui.EnumWindows(enum_window, matched_hwnds)
                except Exception:
                    pass
                best_enum = _pick_best_match(matched_hwnds)
                if best_enum:
                    return self._activate_hwnd(best_enum[0], best_enum[1] or window_title)

        # If neither matches, return an error that lists the real windows (title plus id, capped at 8)
        wanted = window_title or raw_hwnd or "(no window named)"
        if live_windows:
            open_list = [f"'{w.get('title')}' (id: {w.get('id')})" for w in live_windows[:8]]
            open_summary = f"Open windows: {', '.join(open_list)}"
        else:
            open_summary = "Use cua_list_windows to see what is open, or launch it first."

        return {
            "ok": False,
            "sent": False,
            "verified": False,
            "status": "window_not_found",
            "action": "activate",
            "error": f"No open window matches {wanted!r}. {open_summary}",
            "evidence": f"the foreground window is still '{_foreground_title()}'",
        }

    @staticmethod
    def _activated(wanted: str) -> dict[str, Any]:
        """Did the window we asked for actually come to the front?"""
        time.sleep(SETTLE_SECONDS)
        actual = _foreground_title()
        verified = bool(actual) and window_titles_match(wanted, actual)
        return {
            "ok": verified,
            "sent": True,
            "verified": verified,
            "evidence": f"the foreground window is '{actual}'",
            "action": "activate",
            "message": f"Activated window: {wanted}" if verified else f"Could not focus {wanted}",
        }

    @staticmethod
    def _process_names_by_pid() -> dict[int, str]:
        """Map every running PID to its image name in a single tasklist call.

        This used to be one `tasklist` subprocess *per visible window*, each
        with a 5s timeout - and the agent prompt tells the model to call
        list_apps before acting, so it was on the hot path.
        """
        names: dict[int, str] = {}
        try:
            proc = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception as e:
            logger.debug(f"tasklist enumeration failed: {e}")
            return names

        for row in csv.reader(io.StringIO(proc.stdout)):
            # "Image Name","PID","Session Name","Session#","Mem Usage"
            if len(row) < 2:
                continue
            try:
                names[int(row[1].strip())] = row[0].strip()
            except ValueError:
                continue
        return names

    def _list_apps(self, params: dict[str, Any] = None) -> dict[str, Any]:
        import win32gui
        import win32process

        pid_names = self._process_names_by_pid()

        def enum_window(hwnd, results):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                except Exception:
                    pid = 0
                results.append({
                    "title": win32gui.GetWindowText(hwnd),
                    "hwnd": hwnd,
                    "pid": pid,
                    "name": pid_names.get(pid, "Unknown"),
                })

        windows = []
        win32gui.EnumWindows(enum_window, windows)
        apps = {}
        for w in windows:
            app_name = w["name"]
            if app_name not in apps:
                apps[app_name] = {"name": app_name, "windows": []}
            apps[app_name]["windows"].append({"title": w["title"], "hwnd": w["hwnd"]})
        return {"ok": True, "action": "list_apps", "apps": list(apps.values())}

    def _list_windows(self, params: dict[str, Any] = None) -> dict[str, Any]:
        import win32gui
        import win32process

        def enum_window(hwnd, results):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                except Exception:
                    pid = 0
                results.append({
                    "id": hwnd,
                    "title": win32gui.GetWindowText(hwnd),
                    "pid": pid,
                })

        windows = []
        win32gui.EnumWindows(enum_window, windows)
        return {"ok": True, "action": "list_windows", "windows": windows}

    def _get_window(self, params: dict[str, Any]) -> dict[str, Any]:
        import win32gui
        import win32process
        hwnd = params.get("id")
        if not hwnd:
            return {"ok": False, "action": "get_window", "message": "Missing window id"}
        if not win32gui.IsWindow(hwnd):
            return {"ok": False, "action": "get_window", "message": "Invalid window handle"}
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:
            pid = 0
        rect = win32gui.GetWindowRect(hwnd)
        return {
            "ok": True,
            "action": "get_window",
            "window": {
                "id": hwnd,
                "title": win32gui.GetWindowText(hwnd),
                "pid": pid,
                "rect": {"left": rect[0], "top": rect[1], "right": rect[2], "bottom": rect[3]},
            },
        }

    def _launch(self, params: dict[str, Any]) -> dict[str, Any]:
        """Launch an application by name.

        Used to alias a handful of names then hand the raw (possibly
        aliased) text straight to `os.startfile` with no other validation at
        all (R3): no path/protocol/character restriction, unlike the app
        indexer's own safer launch path. A UNC path there leaks the Windows
        user's NTLM handshake unconfirmed; any other URI scheme just runs.
        This now goes through that same validated resolution `open_app`
        already uses (`AppIndexer.launch`) - which resolves against the
        installed-app index, a small set of known protocol handlers, or
        `os.startfile`'s own PATH search, and refuses unsafe shell characters
        itself - after refusing outright what is not "an app name" at all:
        a UNC path, a path with separators, or a scheme other than http/https.
        """
        app = params.get("app", "").strip()
        if not app:
            return {"ok": False, "action": "launch", "message": "Missing app parameter"}

        if app.startswith("\\\\") or app.startswith("//"):
            return {"ok": False, "action": "launch", "message": f"Refusing a network path: {app}"}

        if _BARE_HOST_RE.match(app):
            app = f"https://{app}"

        lower = app.lower()
        if "://" in lower or ":" in app:
            parsed = urlparse(app)
            if parsed.scheme.lower() not in ("http", "https"):
                return {
                    "ok": False, "action": "launch",
                    "message": f"Refusing '{app}': only http/https links are allowed.",
                }
        elif "\\" in app or "/" in app:
            return {
                "ok": False, "action": "launch",
                "message": f"Refusing '{app}': expected an app name, not a path.",
            }

        aliases = {
            "edge": "msedge", "microsoft edge": "msedge", "browser": "msedge",
            "chrome": "chrome", "google chrome": "chrome", "firefox": "firefox",
            "notepad": "notepad", "calculator": "calc", "calc": "calc",
            "word": "winword", "excel": "excel", "powerpoint": "powerpnt",
            "settings": "ms-settings:", "explorer": "explorer",
        }
        target = aliases.get(lower, app)

        if self._app_indexer is None:
            from grace.automation.app_indexer import AppIndexer

            self._app_indexer = AppIndexer()

        result = self._app_indexer.launch(target)
        ok = result.get("status") == "ok"
        message = result.get("text") or result.get("error") or (
            f"Launched {app}" if ok else f"Failed to launch {app}"
        )
        return {"ok": ok, "action": "launch", "message": message}

    def _text(self, params: dict[str, Any] = None) -> dict[str, Any]:
        return self._get_text(params)

    def _get_text(self, params: dict[str, Any] = None) -> dict[str, Any]:
        """Return the window title plus the structured element graph.

        The tool schema has always advertised "accessibility text metadata and
        Win32 control element hierarchy", but this returned only the window
        title - so the agent had no tool at all for reading the control tree it
        was being told to use.
        """
        import win32gui

        hwnd = win32gui.GetForegroundWindow()
        text = win32gui.GetWindowText(hwnd)

        elements: list[dict[str, Any]] = []
        try:
            from grace.agent.perception import PerceptionEngine

            graph = PerceptionEngine.get_graph_builder().get()
            if graph:
                elements = [e.to_dict(compact=True) for e in graph.elements[:60]]
        except Exception as e:
            logger.debug(f"Element graph unavailable in _get_text: {e}")

        return {
            "ok": True,
            "action": "text",
            "text": text,
            "window_title": text,
            "elements": elements,
        }

    def _set_value(self, params: dict[str, Any]) -> dict[str, Any]:
        """Replace the contents of a specific editable field.

        The schema has always taken an `element_index`, but this used to be a
        bare `pyautogui.write(value)`: no targeting, no focus, and no clearing,
        so it appended to whatever happened to be focused at the time.
        """
        import pyautogui

        value = str(params.get("value", ""))
        element_id, target_name = self._target_from_params(params)

        target = self._resolve_from_graph(params, element_id, target_name)
        if target is not None:
            self._click_at(target.center[0], target.center[1], 1,
                           f"graph:{target.role}", target.name)
            time.sleep(0.05)
        elif element_id is not None or target_name:
            return {
                "ok": False,
                "sent": False,
                "verified": False,
                "status": "element_not_found",
                "action": "set_value",
                "error": f"Could not locate '{target_name or element_id}' to set its value.",
                "evidence": "nothing was typed",
            }
        else:
            # No target named at all. The body below is select-all, delete,
            # type - destructive against whatever holds focus - so it must not
            # run on a plan that never said where to put the value.
            return {
                "ok": False,
                "sent": False,
                "verified": False,
                "status": "no_target",
                "action": "set_value",
                "error": "set_value needs an element_id or target_name; refusing "
                         "to overwrite whatever currently has focus.",
                "evidence": "nothing was typed",
            }
        self._ensure_foreground_window()
        refused = _wrong_window(params, "set_value", initial_title=_foreground_title())
        if refused is not None:
            return refused

        try:
            # Select-all then type, so this sets rather than appends.
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("delete")
            if all(ord(c) < 128 for c in value):
                pyautogui.write(value, interval=0.01)
            else:
                try:
                    import pyperclip
                    pyperclip.copy(value)
                    pyautogui.hotkey("ctrl", "v")
                except Exception:
                    pyautogui.write(value, interval=0.01)
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI failsafe caught during set_value")
            _invalidate_graph()
            return {
                "ok": False,
                "sent": False,
                "verified": False,
                "action": "set_value",
                "error": "PyAutoGUI fail-safe triggered (mouse moved to screen corner)",
                "evidence": "action aborted by fail-safe",
            }
        except Exception as e:
            return {"ok": False, "sent": False, "verified": False,
                    "action": "set_value", "error": str(e),
                    "evidence": "the keystrokes could not be sent"}

        _invalidate_graph()
        verified, evidence = self._probe_set_value(target.id, value)
        return {
            "ok": True,
            "sent": True,
            "verified": verified,
            "evidence": evidence,
            "action": "set_value",
            "message": f"Set value to '{value[:60]}'"
                       + (f" on '{target.name}'" if target is not None else ""),
        }

    @staticmethod
    def _probe_set_value(element_id: int, value: str) -> tuple[Optional[bool], str]:
        """Read the field back. `set_value` promises equality, so check for it.

        The element is re-fetched by id rather than by focus: setting a value can
        move focus onward (a combo box that commits and closes), and checking
        whatever ended up focused would then grade the wrong control.
        """
        after = _settled_state()
        if after is None:
            return None, "the value was typed, but this window reports nothing to read back"

        try:
            element = ComputerUse._graph_element(element_id)
        except Exception:
            element = None
        if element is None:
            return None, (f"the value was typed, but element {element_id} is no longer "
                          f"in the tree, so it cannot be read back")
        if not element.value:
            return None, f"'{element.name}' does not report its contents"
        return element.value.strip() == value.strip(), (
            f"'{element.name}' now reads '{element.value[:80]}'"
        )

    # Element actions the agent can request, mapped to how they are performed.
    _SECONDARY_ACTIONS = {
        "context menu": "right_click",
        "right click": "right_click",
        "right_click": "right_click",
        "menu": "right_click",
        "raise": "focus",
        "focus": "focus",
        "double click": "double_click",
        "double_click": "double_click",
        "middle click": "middle_click",
        "middle_click": "middle_click",
    }

    def _secondary_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """Perform a non-primary interaction on an element.

        The schema advertises `element_index` and `action`; the old body
        ignored both and always right-clicked, so asking to raise or
        double-click a control did something else entirely.
        """
        import pyautogui
        from grace.automation.dpi_helper import DPIHelper
        from grace.automation.coordinate_resolver import CoordinateResolver

        DPIHelper.ensure_dpi_aware()

        window = _normalize_window(params.get("window"))
        element_id, target_name = self._target_from_params(params)
        requested = str(params.get("action") or "context menu").strip().lower()
        kind = self._SECONDARY_ACTIONS.get(requested, "right_click")

        x, y = params.get("x"), params.get("y")

        target = self._resolve_from_graph(params, element_id, target_name)
        if target is not None:
            x, y = target.center
        elif x is None or y is None:
            if target_name:
                resolved = CoordinateResolver().resolve(
                    target_name=target_name,
                    x=x,
                    y=y,
                    window_bounds=window.get("bounds"),
                )
                if resolved:
                    x, y = resolved.x, resolved.y

        try:
            if kind == "double_click":
                if x is None or y is None:
                    return {"ok": False, "action": "secondary_action",
                            "error": "Double-click needs a target element or coordinates."}
                return self._click_at(int(x), int(y), 2, "secondary:double_click", target_name)

            if kind == "focus":
                if x is None or y is None:
                    return {"ok": False, "action": "secondary_action",
                            "error": "Focus needs a target element or coordinates."}
                return self._click_at(int(x), int(y), 1, "secondary:focus", target_name)

            if kind == "middle_click":
                if x is None or y is None:
                    return {"ok": False, "action": "secondary_action",
                            "error": "Middle-click needs a target element or coordinates."}
                pyautogui.middleClick(x=int(x), y=int(y))
                _invalidate_graph()
                return {"ok": True, "action": "secondary_action",
                        "message": f"Middle-clicked at ({int(x)}, {int(y)})"}

            if x is not None and y is not None:
                pyautogui.rightClick(x=int(x), y=int(y))
                message = f"Right-clicked at ({int(x)}, {int(y)})"
            else:
                pyautogui.rightClick()
                message = "Context menu opened at the cursor"
        except Exception as e:
            return {"ok": False, "action": "secondary_action", "error": str(e)}

        _invalidate_graph()
        return {"ok": True, "action": "secondary_action", "message": message}

