"""Windows UI Automation provider.

This is Grace's browser-agnostic DOM bridge. Chromium and Gecko both publish
their live accessibility tree through UIA, including ``AriaRole``,
``AriaProperties`` and ``ValuePattern.Value`` - i.e. real DOM semantics, in the
user's own browser profile, with no debug flags and no relaunch.

Three things this does that the previous inspector did not:

* **ControlViewWalker instead of RawViewWalker.** Raw view exposes every
  internal node Chromium creates and buries the useful controls.
* **One bulk cached property fetch per element.** The old code made three
  separate cross-process COM calls (``CurrentName``, ``CurrentBoundingRectangle``,
  ``CurrentControlType``) per element, for up to 250 elements per observation.
* **Chrome vs page classification.** Everything under the ``Document`` node is
  page content; everything else in a browser window is browser UI.
"""

import logging
import threading
from typing import Any, Optional

from grace.perception.elements import (
    FRAME_APP,
    FRAME_CHROME,
    FRAME_PAGE,
    SOURCE_UIA,
    ElementNode,
)

logger = logging.getLogger("grace.perception.uia")

MAX_ELEMENTS_DEFAULT = 120
MAX_ELEMENTS_BROWSER = 250
MAX_DEPTH_DEFAULT = 12
MAX_DEPTH_BROWSER = 18

MIN_SIZE_PX = 4
MAX_NAME_LEN = 160

BROWSER_CLASS_NAMES = {"Chrome_WidgetWin_1", "MozillaWindowClass"}
BROWSER_TITLE_KEYWORDS = ("edge", "chrome", "firefox", "brave", "opera", "vivaldi")

# Control type id -> canonical role name. Kept close to ARIA vocabulary so DOM
# and UIA sourced nodes describe themselves the same way.
CONTROL_TYPE_ROLES = {
    50000: "button", 50001: "calendar", 50002: "checkbox", 50003: "combobox",
    50004: "edit", 50005: "link", 50006: "image", 50007: "listitem",
    50008: "list", 50009: "menu", 50010: "menubar", 50011: "menuitem",
    50012: "progressbar", 50013: "radiobutton", 50014: "scrollbar", 50015: "slider",
    50016: "spinbutton", 50017: "statusbar", 50018: "tab", 50019: "tabitem",
    50020: "text", 50021: "toolbar", 50022: "tooltip", 50023: "tree",
    50024: "treeitem", 50025: "custom", 50026: "group", 50027: "thumb",
    50028: "datagrid", 50029: "dataitem", 50030: "document", 50031: "splitbutton",
    50032: "window", 50033: "pane", 50034: "header", 50035: "headeritem",
    50036: "table", 50037: "titlebar", 50038: "separator", 50039: "semantickey",
    50040: "appbar",
}

DOCUMENT_CONTROL_TYPE = 50030

INTERACTIVE_CONTROL_IDS = {
    50000, 50002, 50003, 50004, 50005, 50007, 50011, 50013, 50015, 50016,
    50018, 50019, 50021, 50024, 50029, 50031, 50035, 50040,
}

# Nodes we descend through but do not report as targets themselves.
CONTAINER_CONTROL_IDS = {50008, 50009, 50010, 50023, 50026, 50030, 50032, 50033, 50036, 50034}

# Containers whose name is worth inheriting as an element's `container`.
NAMED_REGION_CONTROL_IDS = {50021, 50026, 50030, 50033, 50008, 50009, 50010}

_thread_local = threading.local()


def is_browser_window(title: str = "", class_name: str = "") -> bool:
    """Classify a window as a browser (Chrome / Edge / Firefox / Brave / ...)."""
    if class_name and class_name in BROWSER_CLASS_NAMES:
        return True
    hay = (title or "").lower()
    return any(kw in hay for kw in BROWSER_TITLE_KEYWORDS)


class _Uia:
    """Per-thread cached IUIAutomation plus a prebuilt cache request."""

    def __init__(self):
        import comtypes.client

        try:
            from comtypes.gen import UIAutomationClient as client
        except Exception:
            comtypes.client.GetModule("UIAutomationCore.dll")
            from comtypes.gen import UIAutomationClient as client

        self.client = client
        self.uia = comtypes.client.CreateObject(
            client.CUIAutomation, interface=client.IUIAutomation
        )

        self.props = {
            "name": client.UIA_NamePropertyId,
            "control_type": client.UIA_ControlTypePropertyId,
            "rect": client.UIA_BoundingRectanglePropertyId,
            "automation_id": client.UIA_AutomationIdPropertyId,
            "class_name": client.UIA_ClassNamePropertyId,
            "focusable": client.UIA_IsKeyboardFocusablePropertyId,
            "focused": client.UIA_HasKeyboardFocusPropertyId,
            "enabled": client.UIA_IsEnabledPropertyId,
            "offscreen": client.UIA_IsOffscreenPropertyId,
            "help_text": client.UIA_HelpTextPropertyId,
            "aria_role": client.UIA_AriaRolePropertyId,
            "aria_props": client.UIA_AriaPropertiesPropertyId,
            "value": client.UIA_ValueValuePropertyId,
        }

        # One request, reused for every walk: each element then costs a single
        # cross-process round trip instead of one per property.
        request = self.uia.CreateCacheRequest()
        for pid in self.props.values():
            request.AddProperty(pid)
        request.TreeScope = client.TreeScope_Element
        self.cache_request = request

        # Control view hides the internal scaffolding Chromium generates.
        self.walker = self.uia.ControlViewWalker


def _get_uia() -> Optional[_Uia]:
    """Return this thread's cached automation object, creating it once."""
    existing = getattr(_thread_local, "uia", None)
    if existing is not None:
        return existing

    import pythoncom

    try:
        pythoncom.CoInitialize()
    except Exception:
        pass

    try:
        created = _Uia()
    except Exception as e:
        logger.debug(f"UIA initialisation failed: {e}")
        return None

    _thread_local.uia = created
    return created


def reset_thread_cache() -> None:
    """Drop this thread's cached automation object. For tests and teardown."""
    _thread_local.uia = None


def _cached(element, pid, default=None):
    try:
        value = element.GetCachedPropertyValue(pid)
    except Exception:
        return default
    return default if value is None else value


def _parse_aria_properties(raw: str) -> dict[str, str]:
    """Parse UIA's AriaProperties string, e.g. ``placeholder=Search;expanded=false``."""
    parsed: dict[str, str] = {}
    if not raw:
        return parsed
    for chunk in str(raw).split(";"):
        if "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        key = key.strip().lower()
        if key:
            parsed[key] = value.strip()
    return parsed


class UiaProvider:
    """Walks the UIA tree of a window into ElementNodes."""

    def snapshot(
        self,
        hwnd: int,
        title: str = "",
        class_name: str = "",
        start_id: int = 1,
    ) -> list[ElementNode]:
        """Collect interactive elements from the given window."""
        uia = _get_uia()
        if uia is None or not hwnd:
            return []

        browser = is_browser_window(title, class_name)
        cap = MAX_ELEMENTS_BROWSER if browser else MAX_ELEMENTS_DEFAULT
        max_depth = MAX_DEPTH_BROWSER if browser else MAX_DEPTH_DEFAULT
        base_frame = FRAME_CHROME if browser else FRAME_APP

        try:
            root = uia.uia.ElementFromHandleBuildCache(hwnd, uia.cache_request)
        except Exception as e:
            logger.debug(f"ElementFromHandle failed: {e}")
            return []
        if root is None:
            return []

        elements: list[ElementNode] = []
        counter = [start_id]
        try:
            self._walk(uia, root, elements, counter, 0, max_depth, cap, base_frame, "")
        except Exception as e:
            logger.debug(f"UIA walk aborted: {e}")

        return elements

    def _walk(
        self,
        uia: _Uia,
        node,
        elements: list[ElementNode],
        counter: list[int],
        depth: int,
        max_depth: int,
        cap: int,
        frame: str,
        container: str,
    ) -> None:
        if len(elements) >= cap or depth > max_depth:
            return

        try:
            child = uia.walker.GetFirstChildElementBuildCache(node, uia.cache_request)
        except Exception:
            return

        while child is not None:
            if len(elements) >= cap:
                return

            control_type = _cached(child, uia.props["control_type"], 0)
            name = str(_cached(child, uia.props["name"], "") or "").strip()

            # The document node marks the boundary between browser UI and the
            # rendered page - the distinction the planner needs most.
            child_frame = FRAME_PAGE if control_type == DOCUMENT_CONTROL_TYPE else frame

            if control_type in INTERACTIVE_CONTROL_IDS:
                node_obj = self._build_node(uia, child, control_type, name, child_frame, container, counter)
                if node_obj is not None:
                    elements.append(node_obj)

            if control_type in CONTAINER_CONTROL_IDS or control_type not in INTERACTIVE_CONTROL_IDS:
                child_container = container
                if control_type in NAMED_REGION_CONTROL_IDS and name:
                    child_container = name[:60]
                self._walk(
                    uia, child, elements, counter, depth + 1, max_depth, cap,
                    child_frame, child_container,
                )

            try:
                child = uia.walker.GetNextSiblingElementBuildCache(child, uia.cache_request)
            except Exception:
                return

    def _build_node(
        self,
        uia: _Uia,
        element,
        control_type: int,
        name: str,
        frame: str,
        container: str,
        counter: list[int],
    ) -> Optional[ElementNode]:
        rect = _cached(element, uia.props["rect"])
        if not rect or len(rect) < 4:
            return None

        left, top, width, height = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        if width <= MIN_SIZE_PX or height <= MIN_SIZE_PX:
            return None
        right, bottom = left + width, top + height

        aria_props = _parse_aria_properties(_cached(element, uia.props["aria_props"], "") or "")
        placeholder = aria_props.get("placeholder", "")
        if not placeholder:
            placeholder = str(_cached(element, uia.props["help_text"], "") or "").strip()

        aria_role = str(_cached(element, uia.props["aria_role"], "") or "").strip().lower()
        role = aria_role or CONTROL_TYPE_ROLES.get(control_type, f"control_{control_type}")

        # A control with no name at all is only useful if it has a placeholder
        # or an automation id to identify it by.
        automation_id = str(_cached(element, uia.props["automation_id"], "") or "").strip()
        if not name and not placeholder and not automation_id:
            return None
        if len(name) > MAX_NAME_LEN:
            name = name[:MAX_NAME_LEN]

        node = ElementNode(
            id=counter[0],
            role=role,
            name=name,
            rect=(left, top, right, bottom),
            center=(left + width // 2, top + height // 2),
            value=str(_cached(element, uia.props["value"], "") or "")[:120],
            placeholder=placeholder[:120],
            frame=frame,
            container=container,
            focused=bool(_cached(element, uia.props["focused"], False)),
            focusable=bool(_cached(element, uia.props["focusable"], False)),
            enabled=bool(_cached(element, uia.props["enabled"], True)),
            offscreen=bool(_cached(element, uia.props["offscreen"], False)),
            automation_id=automation_id[:60],
            source=SOURCE_UIA,
        )
        counter[0] += 1
        return node
