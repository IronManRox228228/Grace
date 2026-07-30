"""Builds and caches the element graph for the focused window.

Why this exists as a single owned object rather than "whoever needs elements
walks the tree": the agent used to observe the screen, put numbered elements in
the prompt, and then - when the model picked element 3 - re-walk the tree from a
*fresh* inspector inside the click handler and renumber everything from a screen
that had since changed. Element 3 at click time was not element 3 at observe
time.

Here the graph is built once per observation, cached against the foreground
window, and the same ids are used all the way through to the click.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from grace.perception.elements import (
    FRAME_PAGE,
    ElementNode,
    elements_to_prompt,
    find_at_point,
    find_by_id,
    find_by_name,
)
from grace.perception.uia_provider import UiaProvider, is_browser_window

logger = logging.getLogger("grace.perception.graph")

# How long a cached graph stays valid for the same window. Short: UIs change.
DEFAULT_TTL_SECONDS = 1.5


@dataclass
class WindowRef:
    hwnd: int = 0
    title: str = ""
    class_name: str = ""
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def is_browser(self) -> bool:
        return is_browser_window(self.title, self.class_name)


@dataclass
class ElementGraph:
    """An immutable snapshot of the interactive controls in one window."""

    elements: list[ElementNode] = field(default_factory=list)
    window: WindowRef = field(default_factory=WindowRef)
    built_at: float = field(default_factory=time.monotonic)
    sources: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.elements)

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.built_at

    def by_id(self, element_id: int) -> Optional[ElementNode]:
        return find_by_id(self.elements, element_id)

    def by_name(
        self,
        query: str,
        frame: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Optional[ElementNode]:
        return find_by_name(self.elements, query, frame=frame, role=role)

    def at_point(self, x: int, y: int, tolerance_px: int = 30) -> Optional[ElementNode]:
        return find_at_point(self.elements, x, y, tolerance_px=tolerance_px)

    def focused(self) -> Optional[ElementNode]:
        for element in self.elements:
            if element.focused:
                return element
        return None

    def resolve(
        self,
        element_id: Optional[int] = None,
        target_name: Optional[str] = None,
        frame: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Optional[ElementNode]:
        """Resolve a planner-supplied target to a concrete element.

        Tried in order of reliability: an explicit id, then a name scoped to the
        requested frame, then the same name anywhere. The frame-scoped attempt
        is what keeps "the search box" off the browser's address bar when the
        planner asked for page content.
        """
        if element_id is not None:
            found = self.by_id(element_id)
            if found is not None:
                return found

        if target_name:
            if frame:
                # An explicit frame is a constraint, not a hint: falling back to
                # an unscoped search here would hand back the address bar for a
                # request that deliberately said "page".
                return self.by_name(target_name, frame=frame, role=role)

            # No frame given. In a browser, page content is nearly always the
            # intent, so try it before anything in the surrounding chrome.
            if self.window.is_browser:
                found = self.by_name(target_name, frame=FRAME_PAGE, role=role)
                if found is not None:
                    return found
            return self.by_name(target_name, role=role)

        return None

    def to_prompt(self, limit: Optional[int] = 40) -> str:
        return elements_to_prompt(self.elements, limit=limit)


class ElementGraphBuilder:
    """Builds element graphs, caching per foreground window."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS, cdp_port: int = 0):
        self._uia = UiaProvider()
        self._ttl = ttl_seconds
        self._cdp_port = cdp_port
        self._lock = threading.Lock()
        self._cached: Optional[ElementGraph] = None

    # -- window discovery ------------------------------------------------

    @staticmethod
    def active_window() -> WindowRef:
        try:
            import win32gui

            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return WindowRef()
            return WindowRef(
                hwnd=hwnd,
                title=win32gui.GetWindowText(hwnd),
                class_name=win32gui.GetClassName(hwnd),
                rect=tuple(win32gui.GetWindowRect(hwnd)),
            )
        except Exception as e:
            logger.debug(f"Foreground window query failed: {e}")
            return WindowRef()

    # -- building --------------------------------------------------------

    def build(self, window: Optional[WindowRef] = None) -> ElementGraph:
        """Build a fresh graph for the given (or foreground) window."""
        window = window or self.active_window()
        if not window.hwnd:
            return ElementGraph(window=window)

        sources: list[str] = []
        elements: list[ElementNode] = []

        # DOM first when available: its rects and roles come straight from the
        # renderer. UIA then fills in browser chrome, which CDP cannot see.
        dom_elements: list[ElementNode] = []
        if self._cdp_port and window.is_browser:
            dom_elements = self._build_dom(window)
            if dom_elements:
                elements.extend(dom_elements)
                sources.append("dom")

        try:
            uia_elements = self._uia.snapshot(
                window.hwnd,
                title=window.title,
                class_name=window.class_name,
                start_id=len(elements) + 1,
            )
        except Exception as e:
            logger.debug(f"UIA snapshot failed: {e}")
            uia_elements = []

        if uia_elements:
            if dom_elements:
                uia_elements = _drop_overlapping(uia_elements, dom_elements)
            elements.extend(uia_elements)
            sources.append("uia")

        # Ids must be dense and stable within a graph; renumber after merging.
        for index, element in enumerate(elements, start=1):
            element.id = index

        graph = ElementGraph(elements=elements, window=window, sources=tuple(sources))
        with self._lock:
            self._cached = graph
        return graph

    def get(self, max_age: Optional[float] = None) -> ElementGraph:
        """Return a cached graph if it is fresh and for the same window."""
        ttl = self._ttl if max_age is None else max_age
        window = self.active_window()

        with self._lock:
            cached = self._cached

        if (
            cached is not None
            and cached.window.hwnd == window.hwnd
            and cached.age_seconds <= ttl
        ):
            logger.debug(f"Element graph cache hit ({len(cached)} elements, {cached.age_seconds:.2f}s old)")
            return cached

        return self.build(window)

    def invalidate(self) -> None:
        """Force the next get() to rebuild. Call after anything that acts on the UI."""
        with self._lock:
            self._cached = None

    def _build_dom(self, window: WindowRef) -> list[ElementNode]:
        try:
            from grace.perception.dom_provider import DomProvider

            return DomProvider(self._cdp_port).snapshot(window)
        except Exception as e:
            logger.debug(f"DOM provider unavailable ({e}); using UIA only")
            return []


def _drop_overlapping(
    uia_elements: list[ElementNode],
    dom_elements: list[ElementNode],
    tolerance_px: int = 6,
) -> list[ElementNode]:
    """Remove UIA nodes that duplicate a DOM node already collected.

    Chromium exposes page content through both channels, so without this the
    planner would see every page control twice under two different ids.
    """
    kept: list[ElementNode] = []
    for element in uia_elements:
        if element.frame != FRAME_PAGE:
            kept.append(element)
            continue
        duplicate = any(
            abs(element.center[0] - dom.center[0]) <= tolerance_px
            and abs(element.center[1] - dom.center[1]) <= tolerance_px
            for dom in dom_elements
        )
        if not duplicate:
            kept.append(element)
    return kept
