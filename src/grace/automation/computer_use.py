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
            return {"error": f"Unknown action: {action}"}
        try:
            return handler(params)
        except Exception as e:
            logger.error(f"Action {action} failed: {e}")
            return {"error": str(e)}

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
                # Only add window left/top if x and y are within local window relative range (0..w, 0..h)
                # AND they are less than left/top (meaning they cannot already be absolute screen coordinates)
                if window_w and window_h and (0 <= x <= window_w) and (0 <= y <= window_h):
                    if x < left or y < top:
                        x = left + x
                        y = top + y
            except (ValueError, TypeError):
                pass

        return x, y

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
                                  click_count, f"graph:{graph_target.role}", graph_target.name)

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
                "status": "element_not_found",
                "error": f"Target element '{target_name or relative_to}' not found via UIA, OculiX, OpenCV, or OCR",
                "message": f"I couldn't locate the '{target_name or relative_to}' button on screen."
            }

        return self._click_at(resolved.x, resolved.y, click_count, resolved.method,
                              getattr(resolved.element, "name", None))

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
                  name: Optional[str] = None) -> dict[str, Any]:
        """Send the actual click, then invalidate the graph the UI just changed."""
        import pyautogui

        logger.info(f"_click executing via method '{method}' at ({x}, {y})")
        try:
            from grace.automation.win32_driver import Win32Driver
            Win32Driver.click_at(x, y, clicks=click_count)
        except Exception as e:
            logger.debug(f"Win32Driver click failed: {e}, using pyautogui fallback")
            pyautogui.click(x=x, y=y, clicks=click_count)

        _invalidate_graph()

        return {
            "ok": True,
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

        # Typing used to be entirely blind. Check the element graph for what
        # actually has keyboard focus and report it, so a step that typed a
        # search query into the wrong box is visible in the result instead of
        # silently succeeding.
        focus_note = ""
        target_name = params.get("target_name") or params.get("into")
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
                            "status": "wrong_focus",
                            "action": "type_text",
                            "error": f"Focus is on '{focused.name or focused.placeholder}' "
                                     f"(frame={focused.frame}), not '{expected.name or expected.placeholder}' "
                                     f"(frame={expected.frame}). Click the intended field first.",
                            "focused_id": focused.id,
                            "expected_id": expected.id,
                        }
            elif target_name:
                logger.debug(f"_type_text: no focused element found; typing '{target_name}' blind")
        except Exception as e:
            logger.debug(f"Focus check skipped: {e}")

        try:
            pyautogui.write(text, interval=0.01)
        except pyautogui.FailSafeException:
            logger.debug("PyAutoGUI failsafe caught during type_text")

        _invalidate_graph()
        return {
            "ok": True,
            "action": "type_text",
            "message": f"Typed {len(text)} characters{focus_note}",
        }

    def _press_key(self, params: dict[str, Any]) -> dict[str, Any]:
        import pyautogui
        raw_key = str(params.get("key", ""))
        
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
        return {"ok": True, "action": "press_key", "message": f"Pressed {raw_key}"}

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

        scroll_x = _to_int(params.get("scrollX")) or 0
        scroll_y = _to_int(params.get("scrollY"))

        # Default scroll distance if not specified
        if scroll_y is None and scroll_x == 0:
            scroll_y = 500

        try:
            if explicit_coords and x is not None and y is not None:
                pyautogui.scroll(-scroll_y, x=x, y=y)
            else:
                pyautogui.scroll(-scroll_y)
            if scroll_x:
                pyautogui.hscroll(scroll_x)
        except pyautogui.FailSafeException:
            logger.debug("PyAutoGUI failsafe caught during scroll")
        return {"ok": True, "action": "scroll", "message": f"Scrolled ({scroll_x}, {scroll_y or 0})"}

    def _drag(self, params: dict[str, Any]) -> dict[str, Any]:
        import pyautogui

        def _to_int(val, default=0):
            if val is None:
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        from_x = _to_int(params.get("from_x"), 0)
        from_y = _to_int(params.get("from_y"), 0)
        to_x = _to_int(params.get("to_x"), 0)
        to_y = _to_int(params.get("to_y"), 0)

        try:
            pyautogui.moveTo(from_x, from_y)
            pyautogui.drag(to_x - from_x, to_y - from_y, duration=0.3)
        except pyautogui.FailSafeException:
            logger.debug("PyAutoGUI failsafe caught during drag")
        return {"ok": True, "action": "drag", "message": f"Dragged from ({from_x},{from_y}) to ({to_x},{to_y})"}

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
                    return {"ok": True, "action": "activate", "message": f"Activated window hwnd: {hwnd}"}
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
                    return {"ok": True, "action": "activate", "message": f"Activated window: {window_title}"}
                except Exception as e:
                    return {"ok": False, "action": "activate", "message": f"Could not focus {window_title}: {e}"}

        return {"ok": True, "action": "activate", "message": "Window activated"}

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
        except Exception:
            try:
                subprocess.Popen(f'start "" "{target}"', shell=True)
                return {"ok": True, "action": "launch", "message": f"Launched {app}"}
            except Exception as e:
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
        element_index = params.get("element_index")
        target_name = params.get("target_name") or params.get("name")

        target = self._resolve_from_graph(params, element_index, target_name)
        if target is not None:
            self._click_at(target.center[0], target.center[1], 1,
                           f"graph:{target.role}", target.name)
            time.sleep(0.05)
        elif element_index is not None or target_name:
            return {
                "ok": False,
                "status": "element_not_found",
                "action": "set_value",
                "error": f"Could not locate '{target_name or element_index}' to set its value.",
            }

        try:
            # Select-all then type, so this sets rather than appends.
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("delete")
            pyautogui.write(value, interval=0.01)
        except Exception as e:
            return {"ok": False, "action": "set_value", "error": str(e)}

        _invalidate_graph()
        return {
            "ok": True,
            "action": "set_value",
            "message": f"Set value to '{value[:60]}'"
                       + (f" on '{target.name}'" if target is not None else ""),
        }

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
        target_name = params.get("target_name") or params.get("name")
        element_index = params.get("element_index")
        requested = str(params.get("action") or "context menu").strip().lower()
        kind = self._SECONDARY_ACTIONS.get(requested, "right_click")

        x, y = params.get("x"), params.get("y")

        target = self._resolve_from_graph(params, element_index, target_name)
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

