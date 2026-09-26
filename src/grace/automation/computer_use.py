import base64
import csv
import io
import logging
import os
import subprocess
import time
from typing import Any, Optional

logger = logging.getLogger("grace.computer_use")


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
    return {
        "window": getattr(graph.window, "title", "") or "",
        "elements": len(graph),
        "focus_id": focused.id if focused is not None else None,
        "focus_name": (focused.name or focused.placeholder) if focused is not None else "",
        "focus_value": focused.value if focused is not None else "",
        "blind": blind,
    }


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
    """Whether the screen changed, in the three-valued sense `verified` needs."""
    if before is None or after is None:
        return None, ("the input was sent, but this window reports nothing that "
                      "could confirm it - check visually or by its side effects")

    changes = []
    if after["window"] != before["window"]:
        changes.append(f"the active window is now '{after['window']}'")
    if after["focus_id"] != before["focus_id"]:
        changes.append(f"keyboard focus moved to '{after['focus_name']}'")
    elif after["focus_value"] != before["focus_value"]:
        changes.append(f"the focused field now reads '{after['focus_value'][:60]}'")
    if after["elements"] != before["elements"]:
        changes.append(
            f"the window went from {before['elements']} to {after['elements']} elements"
        )

    if changes:
        return True, "; ".join(changes)
    if before["blind"] or after["blind"]:
        return None, ("nothing observable changed, but this window does not report "
                      "its contents, so that is not evidence either way")
    return False, ("nothing changed after settling: same window, same focus, "
                   "same elements")


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

        return win32gui.GetWindowText(win32gui.GetForegroundWindow()) or ""
    except Exception:
        return ""


def _wrong_window(params: dict[str, Any], action: str) -> Optional[dict[str, Any]]:
    """Refuse an action aimed at a window that is not in front.

    Modelled on the `wrong_focus` guard `_type_text` already had, which is the
    one precondition in the file that ever worked: it names both sides, and
    because `ok is False` the loop turns it into an explicit correction on the
    very next planner call at no extra cost.

    Only fires when the plan actually named a window. A missing or unmatched
    name means "wherever we are", which is how most steps are written and must
    keep working.
    """
    wanted = _normalize_window(params.get("window")).get("title") or ""
    wanted = str(wanted).strip()
    if not wanted:
        return None

    actual = _foreground_title()
    if not actual:
        return None

    a, b = wanted.lower(), actual.lower()
    if a in b or b in a:
        return None

    return {
        "ok": False,
        "sent": False,
        "verified": False,
        "status": "wrong_window",
        "action": action,
        "error": (f"You aimed at '{wanted}' but '{actual}' is in front. "
                  f"Activate the window you meant first."),
        "evidence": f"foreground window is '{actual}'",
    }


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

    def __init__(self):
        self._available = False

    @property
    def is_ready(self) -> bool:
        return self._available

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

        window = _normalize_window(params.get("window"))
        target_name = params.get("target_name") or params.get("name") or params.get("label")
        element_index = _to_int(params.get("element_index"))
        relative_to = params.get("relative_to")
        direction = params.get("direction", "left").lower().strip()
        click_count = _to_int(params.get("click_count")) or 1

        x = _to_int(params.get("x") if params.get("x") is not None else window.get("x"))
        y = _to_int(params.get("y") if params.get("y") is not None else window.get("y"))

        window_bounds = None
        bounds = window.get("bounds") or window.get("rect")
        if bounds and isinstance(bounds, (list, tuple)) and len(bounds) >= 4:
            window_bounds = (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))

        refused = _wrong_window(params, "click")
        if refused is not None:
            return refused

        before = _screen_state()

        # Resolve against the element graph the planner actually saw, so
        # element ids mean the same thing here as they did in the prompt.
        #
        # This used to take a *second* full screenshot + OCR + UIA walk whose
        # result was passed to resolve() and never read, and then build a fresh
        # CoordinateResolver whose own new UIInspector triggered a *third* walk
        # with renumbered ids. Two full observations per click, for nothing.
        graph_target = self._resolve_from_graph(params, element_index, target_name)
        if graph_target is not None:
            return self._click_at(graph_target.center[0], graph_target.center[1],
                                  click_count, f"graph:{graph_target.role}", graph_target.name,
                                  probe=True, before=before)

        # Fall back to the legacy cascade for coordinate-only or unmatched targets.
        resolver = CoordinateResolver()
        resolved = resolver.resolve(
            element_index=element_index,
            target_name=target_name,
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
                              probe=True, before=before)

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
        return element_id, target_name

    def _resolve_from_graph(self, params: dict[str, Any], element_index, target_name):
        """Look the target up in the cached element graph, if we have one."""
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

    def _click_at(self, x: int, y: int, click_count: int, method: str,
                  name: Optional[str] = None, probe: bool = False,
                  before: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Send the actual click, then invalidate the graph the UI just changed.

        ``probe`` is off for internal callers - `_set_value` clicks only to put
        focus somewhere and then checks the value it wrote, so a change probe
        there would pay for a settle and answer the wrong question.
        """
        import pyautogui

        logger.info(f"_click executing via method '{method}' at ({x}, {y})")
        try:
            from grace.automation.win32_driver import Win32Driver
            Win32Driver.click_at(x, y, clicks=click_count)
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

        refused = _wrong_window(params, "type_text")
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

            pyautogui.write(text, interval=0.01)
        except pyautogui.FailSafeException:
            logger.debug("PyAutoGUI failsafe caught during type_text")

        _invalidate_graph()
        verified, evidence = self._probe_typed(text, bool(params.get("replace")))
        return {
            "ok": True,
            "sent": True,
            "verified": verified,
            "evidence": evidence,
            "action": "type_text",
            "message": f"Typed {len(text)} characters{focus_note}",
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

        refused = _wrong_window(params, "press_key")
        if refused is not None:
            return refused

        before = _screen_state()

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
            logger.debug("PyAutoGUI failsafe caught during press_key")
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
            logger.debug("PyAutoGUI failsafe caught during scroll")

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
            logger.debug("PyAutoGUI failsafe caught during drag")
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

    def _activate(self, params: dict[str, Any]) -> dict[str, Any]:
        import win32gui
        import win32con

        window = _normalize_window(params.get("window"))
        hwnd = params.get("hwnd") or params.get("id") or window.get("id") or window.get("hwnd")
        window_title = params.get("window_title") or params.get("title") or window.get("title") or window.get("app") or params.get("app", "")
        if isinstance(params.get("window"), str) and not window_title:
            window_title = params.get("window")

        if hwnd:
            try:
                if win32gui.IsWindow(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    win32gui.SetForegroundWindow(hwnd)
                    return self._activated(win32gui.GetWindowText(hwnd) or str(hwnd))
            except Exception as e:
                logger.debug(f"Activate via hwnd failed: {e}")

        if window_title:
            def enum_window(h, results):
                try:
                    if win32gui.IsWindowVisible(h):
                        t = win32gui.GetWindowText(h)
                        if window_title.lower() in t.lower():
                            results.append(h)
                except Exception:
                    pass
            matched_hwnds = []
            try:
                win32gui.EnumWindows(enum_window, matched_hwnds)
            except Exception:
                pass
            if matched_hwnds:
                target_hwnd = matched_hwnds[0]
                try:
                    win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)
                    win32gui.SetForegroundWindow(target_hwnd)
                    return self._activated(window_title)
                except Exception as e:
                    return {"ok": False, "sent": True, "verified": False,
                            "action": "activate",
                            "message": f"Could not focus {window_title}: {e}",
                            "evidence": f"SetForegroundWindow raised: {e}"}

        # Nothing matched. This used to report success, which meant the planner
        # was told the window it asked for was in front and then planned its
        # next step against a screen belonging to some other app entirely.
        wanted = window_title or hwnd or "(no window named)"
        return {
            "ok": False,
            "sent": False,
            "verified": False,
            "status": "window_not_found",
            "action": "activate",
            "error": f"No open window matches {wanted!r}. Use cua_list_windows to see what is open, or launch it first.",
            "evidence": f"the foreground window is still '{_foreground_title()}'",
        }

    @staticmethod
    def _activated(wanted: str) -> dict[str, Any]:
        """Did the window we asked for actually come to the front?

        Windows refuses `SetForegroundWindow` from a process that does not own
        the current foreground window, and does so *silently* - the call returns
        and nothing happens. Every precondition that asks "is the right window
        in front" is downstream of this one, so it checks rather than assumes.
        """
        time.sleep(SETTLE_SECONDS)
        actual = _foreground_title()
        wanted_l, actual_l = wanted.lower(), actual.lower()
        verified = bool(actual) and (wanted_l in actual_l or actual_l in wanted_l)
        return {
            "ok": True,
            "sent": True,
            "verified": verified,
            "evidence": f"the foreground window is '{actual}'",
            "action": "activate",
            "message": f"Activated window: {wanted}",
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
        app = params.get("app", "").strip()
        if not app:
            return {"ok": False, "action": "launch", "message": "Missing app parameter"}
        
        aliases = {
            "edge": "msedge", "microsoft edge": "msedge", "browser": "msedge",
            "chrome": "chrome", "google chrome": "chrome", "firefox": "firefox",
            "notepad": "notepad", "calculator": "calc", "calc": "calc",
            "word": "winword", "excel": "excel", "powerpoint": "powerpnt",
            "settings": "ms-settings:", "explorer": "explorer",
        }
        target = aliases.get(app.lower(), app)
        try:
            os.startfile(target)
            return {"ok": True, "action": "launch", "message": f"Launched {app}"}
        except Exception as e:
            # No shell fallback: `target` can be model-generated text, and a
            # shell=True Popen re-parses it for &|<>^ - a launch request is not
            # supposed to be able to run a second, arbitrary command.
            return {"ok": False, "action": "launch", "message": f"Failed to launch {app}: {e}"}

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

        try:
            # Select-all then type, so this sets rather than appends.
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("delete")
            pyautogui.write(value, interval=0.01)
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

