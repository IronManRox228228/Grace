"""Windows UI Automation (UIA) Control Inspector for Grace.

Inspects native Windows accessibility controls (Buttons, Edit boxes, Hyperlinks,
Tabs, MenuItems) in the active foreground window using native Windows APIs,
returning exact pixel center coordinates to eliminate coordinate guessing.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger("grace.automation.ui_inspector")

# Mapping from Windows COM UIA Control Type IDs to human-readable names
UIA_CONTROL_TYPES = {
    50000: "Button", 50001: "Calendar", 50002: "CheckBox", 50003: "ComboBox",
    50004: "Edit", 50005: "Hyperlink", 50006: "Image", 50007: "ListItem",
    50008: "List", 50009: "Menu", 50010: "MenuBar", 50011: "MenuItem",
    50012: "ProgressBar", 50013: "RadioButton", 50014: "ScrollBar", 50015: "Slider",
    50016: "Spinner", 50017: "StatusBar", 50018: "Tab", 50019: "TabItem",
    50020: "Text", 50021: "ToolBar", 50022: "ToolTip", 50023: "Tree",
    50024: "TreeItem", 50025: "Custom", 50026: "Group", 50027: "Thumb",
    50028: "DataGrid", 50029: "DataItem", 50030: "Document", 50031: "SplitButton",
    50032: "Window", 50033: "Pane", 50034: "Header", 50035: "HeaderItem",
    50036: "Table", 50037: "TitleBar", 50038: "Separator", 50039: "SemanticKey",
    50040: "AppBar",
}

# Control type IDs we treat as interactive (kept as Button/Edit/Hyperlink/etc.)
INTERACTIVE_CONTROL_IDS = {
    50000, 50002, 50003, 50004, 50005, 50006, 50007, 50011, 50013,
    50015, 50016, 50018, 50019, 50021, 50023, 50024, 50025, 50026,
    50029, 50031, 50035, 50040,
}
# Container types that are pruned mid-traversal (when nameless) to reach deep
# media controls buried under Shadow DOM / iframe wrappers.
CONTAINER_CONTROL_IDS = {50026: "Group", 50033: "Pane", 50032: "Window"}

# Fix Bug #4: Chromium web pages bury media buttons at depth 6-10 inside
# Shadow DOM / iframe wrappers. Default windows use a flat FindAll with a
# modest cap; browsers get a deep recursive TreeWalker with a higher cap.
MAX_ELEMENTS_DEFAULT = 100
MAX_ELEMENTS_BROWSER = 250
MAX_DEPTH_BROWSER = 8

BROWSER_CLASS_NAMES = {"Chrome_WidgetWin_1", "MozillaWindowClass"}
BROWSER_TITLE_KEYWORDS = ("Edge", "Chrome", "Firefox", "Google Chrome", "Brave")


@dataclass
class UIElement:
    """Metadata for a native Windows UI control element."""

    index: int
    name: str
    control_type: str
    bounds: Tuple[int, int, int, int]  # left, top, right, bottom
    center: Tuple[int, int]  # center_x, center_y


class UIInspector:
    """Inspects and resolves interactive UI controls in Windows applications."""

    def __init__(self):
        self._last_elements: List[UIElement] = []

    @staticmethod
    def _is_browser_window(hwnd: int, title: str = "", class_name: str = "") -> bool:
        """Classify a window as a browser (Edge / Chrome / Firefox / Brave)."""
        if class_name and class_name in BROWSER_CLASS_NAMES:
            return True
        # Fallback to title keyword matching for newer Edge variants
        hay = (title or "").lower()
        return any(kw.lower() in hay for kw in BROWSER_TITLE_KEYWORDS)

    def inspect_active_window(self) -> List[UIElement]:
        """Query interactive controls from the active foreground window."""
        elements: List[UIElement] = []
        idx = 1
        hwnd = 0
        win_title = ""
        win_class = ""

        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd or not win32gui.IsWindowVisible(hwnd):
                return []

            win_title = win32gui.GetWindowText(hwnd)
            win_class = win32gui.GetClassName(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            win_left, win_top, win_right, win_bottom = rect

            # Child window controls enumeration via Win32 EnumChildWindows
            def enum_child(child_hwnd, _):
                nonlocal idx
                try:
                    if win32gui.IsWindowVisible(child_hwnd):
                        txt = win32gui.GetWindowText(child_hwnd).strip()
                        cls = win32gui.GetClassName(child_hwnd).strip()
                        if txt and len(txt) < 100:
                            c_rect = win32gui.GetWindowRect(child_hwnd)
                            cl, ct, cr, cb = c_rect
                            cw = cr - cl
                            ch = cb - ct
                            if cw > 5 and ch > 5 and cl >= win_left - 10 and ct >= win_top - 10:
                                center_x = cl + cw // 2
                                center_y = ct + ch // 2
                                elem = UIElement(
                                    index=idx,
                                    name=txt,
                                    control_type=cls,
                                    bounds=(cl, ct, cr, cb),
                                    center=(center_x, center_y),
                                )
                                elements.append(elem)
                                idx += 1
                except Exception:
                    pass

            win32gui.EnumChildWindows(hwnd, enum_child, None)

        except Exception as e:
            logger.debug(f"UIInspector win32 enumeration failed: {e}")

        # Native COM CUIAutomation query (< 10ms) for Chromium / WPF / UWP controls
        if hwnd:
            is_browser = self._is_browser_window(hwnd, win_title, win_class)
            uia_elements = self._query_native_com_uia(
                hwnd, start_index=idx, is_browser=is_browser
            )
            if uia_elements:
                elements.extend(uia_elements)

        self._last_elements = elements
        return elements

    def _query_native_com_uia(
        self,
        hwnd: int,
        start_index: int = 1,
        is_browser: bool = False,
    ) -> List[UIElement]:
        """In-memory native COM CUIAutomation inspection (< 10ms execution time)."""
        elements: List[UIElement] = []
        import pythoncom
        pythoncom.CoInitialize()

        try:
            import comtypes.client
            try:
                from comtypes.gen import UIAutomationClient
            except Exception:
                comtypes.client.GetModule("UIAutomationCore.dll")
                from comtypes.gen import UIAutomationClient

            uia = comtypes.client.CreateObject(
                UIAutomationClient.CUIAutomation, interface=UIAutomationClient.IUIAutomation
            )
            elem = uia.ElementFromHandle(hwnd)
            if not elem:
                return []

            cap = MAX_ELEMENTS_BROWSER if is_browser else MAX_ELEMENTS_DEFAULT

            if is_browser:
                # Deep recursive walk so Shadow DOM / iframe media buttons
                # (depth 6-10) surface instead of being cut off.
                self._walk_uia_tree(
                    uia, elem, elements,
                    depth=0, max_depth=MAX_DEPTH_BROWSER, cap=cap,
                )
            else:
                condition = uia.CreateTrueCondition()
                controls = elem.FindAll(UIAutomationClient.TreeScope_Descendants, condition)
                idx = start_index
                for i in range(controls.Length):
                    if len(elements) >= cap:
                        break
                    try:
                        c = controls.GetElement(i)
                        self._maybe_append(elements, c)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Native COM UIA query skipped: {e}")
        finally:
            # FIX BUG #10: Always uninitialize COM to prevent handle leaks
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

        return elements

    def _maybe_append(
        self,
        elements: List[UIElement],
        c,
        start_index_offset: Optional[int] = None,
    ) -> None:
        """Append a COM UIA element to the result list if it is interactive."""
        try:
            name = c.CurrentName
            rect = c.CurrentBoundingRectangle
            raw_ctl_type = c.CurrentControlType

            ctl_type = UIA_CONTROL_TYPES.get(raw_ctl_type, f"Control_{raw_ctl_type}")

            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if name and name.strip() and len(name.strip()) < 120 and w > 5 and h > 5:
                cx = rect.left + w // 2
                cy = rect.top + h // 2
                elements.append(
                    UIElement(
                        index=len(elements) + 1,
                        name=name.strip(),
                        control_type=ctl_type,
                        bounds=(rect.left, rect.top, rect.right, rect.bottom),
                        center=(cx, cy),
                    )
                )
        except Exception:
            pass

    def _walk_uia_tree(
        self,
        uia,
        node,
        elements: List[UIElement],
        depth: int,
        max_depth: int,
        cap: int,
    ) -> None:
        """Depth-limited recursive TreeWalker over the UIA tree.

        Prunes nameless container nodes (Group/Pane/Window) mid-traversal so
        deep media buttons surface without exploding the element count.
        """
        if len(elements) >= cap or depth > max_depth:
            return

        try:
            walker = uia.RawViewWalker
            child = walker.GetFirstChildElement(node)
        except Exception:
            return

        while child is not None:
            if len(elements) >= cap:
                return

            raw_ctl_type = None
            try:
                raw_ctl_type = child.CurrentControlType
            except Exception:
                raw_ctl_type = None

            # Append interactive elements; recurse into named containers.
            if raw_ctl_type in INTERACTIVE_CONTROL_IDS:
                self._maybe_append(elements, child)
            elif raw_ctl_type in CONTAINER_CONTROL_IDS:
                # Only descend into containers that carry a name (real region)
                # or are at shallow depth, to avoid exploding node counts.
                try:
                    cname = (child.CurrentName or "").strip()
                except Exception:
                    cname = ""
                if cname or depth < 3:
                    self._walk_uia_tree(uia, child, elements, depth + 1, max_depth, cap)
            else:
                # Unknown / generic: descend to reach wrapped media controls.
                self._walk_uia_tree(uia, child, elements, depth + 1, max_depth, cap)

            try:
                child = walker.GetNextSiblingElement(child)
            except Exception:
                return

    def find_element(
        self, target_name: Optional[str] = None, element_index: Optional[int] = None
    ) -> Optional[UIElement]:
        """Find matching UI element by index or target name."""
        if not self._last_elements:
            self.inspect_active_window()

        # 1. Match by index
        if element_index is not None:
            for elem in self._last_elements:
                if elem.index == element_index:
                    return elem

        # 2. Match by target_name
        if target_name:
            q = target_name.lower().strip()
            # Exact match
            for elem in self._last_elements:
                if elem.name.lower().strip() == q:
                    return elem
            # Partial match. Deliberately one-directional: the reverse test
            # (`elem.name in q`) meant a control named "x" matched any query
            # containing an x, so lookups for names that do not exist returned
            # an arbitrary control instead of failing. Short names are also
            # excluded from substring matching for the same reason.
            for elem in self._last_elements:
                name = elem.name.lower().strip()
                if len(name) >= 3 and q in name:
                    return elem

        return None

    def find_relative_element(
        self,
        target_name: Optional[str] = None,
        relative_to: Optional[str] = None,
        direction: str = "left",
    ) -> Optional[UIElement]:
        """Find control positioned relative to an anchor text label."""
        if not self._last_elements:
            self.inspect_active_window()

        if not relative_to:
            return self.find_element(target_name=target_name)

        anchor_elem = self.find_element(target_name=relative_to)
        if not anchor_elem:
            return self.find_element(target_name=target_name)

        ref_cx, ref_cy = anchor_elem.center

        candidates = []
        for elem in self._last_elements:
            if elem == anchor_elem:
                continue
            ex, ey = elem.center
            if abs(ey - ref_cy) <= 50:
                dist = abs(ex - ref_cx)
                dir_ok = False
                if direction in ("left", "before") and ex < ref_cx:
                    dir_ok = True
                elif direction in ("right", "after") and ex > ref_cx:
                    dir_ok = True
                elif direction in ("beside", "near", "adjacent"):
                    dir_ok = True

                if dir_ok:
                    candidates.append((dist, elem))

        if candidates:
            candidates.sort(key=lambda c: c[0])
            return candidates[0][1]

        return self.find_element(target_name=target_name)

    def find_element_at_point(self, x: int, y: int, tolerance_px: int = 30) -> Optional[UIElement]:
        """Find interactive Win32 UIA control under or near point (x, y) for exact bounds snapping."""
        if not self._last_elements:
            self.inspect_active_window()

        best_elem: Optional[UIElement] = None
        best_dist = float("inf")

        for elem in self._last_elements:
            left, top, right, bottom = elem.bounds
            # Exact hit inside element bounding rectangle
            if left <= x <= right and top <= y <= bottom:
                return elem

            # Calculate distance to element center
            cx, cy = elem.center
            dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            if dist <= tolerance_px and dist < best_dist:
                best_dist = dist
                best_elem = elem

        return best_elem
