"""Optional DOM element extraction over the Chrome DevTools Protocol.

**Attach-only, and off by default.** Grace never launches a browser with a
debug flag and never creates or touches a browser profile. If ``GRACE_CDP_PORT``
names a port that is already listening, this attaches to it; otherwise the
element graph is built from the UIA/ARIA tree, which works in every browser, in
the user's own profile, with no flags.

That restriction is deliberate. Chrome and Edge 136+ refuse
``--remote-debugging-port`` when launched against the default user-data-dir, so
"CDP everywhere" would mean forcing users onto a throwaway profile. UIA already
exposes the same DOM accessibility tree, so CDP here only buys exact geometry
when someone has independently opened a debug port.
"""

import json
import logging
import socket
from typing import Any, Optional

from grace.perception.elements import FRAME_PAGE, SOURCE_DOM, ElementNode

logger = logging.getLogger("grace.perception.dom")

CONNECT_TIMEOUT = 0.35
CALL_TIMEOUT = 2.0
MAX_ELEMENTS = 200

# AX roles worth offering as targets, mapped onto our canonical role names.
AX_ROLE_MAP = {
    "button": "button", "link": "link", "textbox": "textbox",
    "searchbox": "searchbox", "combobox": "combobox", "checkbox": "checkbox",
    "radio": "radiobutton", "menuitem": "menuitem", "tab": "tab",
    "option": "option", "switch": "switch", "slider": "slider",
    "spinbutton": "spinbutton", "listitem": "listitem", "treeitem": "treeitem",
}


def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    """True if something is already listening. Never starts anything."""
    if not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT):
            return True
    except Exception:
        return False


class DomProvider:
    """Reads interactive elements from an already-open DevTools endpoint."""

    def __init__(self, port: int, host: str = "127.0.0.1"):
        self._port = port
        self._host = host

    def is_available(self) -> bool:
        return port_is_open(self._port, self._host)

    def snapshot(self, window) -> list[ElementNode]:
        """Return DOM elements in screen coordinates, or [] if unavailable."""
        if not self.is_available():
            return []

        try:
            target = self._pick_target(window)
            if not target:
                return []
            payload = self._query_page(target)
        except Exception as e:
            logger.debug(f"CDP query failed ({e}); falling back to UIA")
            return []

        return self._to_elements(payload, window)

    # -- CDP plumbing ----------------------------------------------------

    def _http_get(self, path: str) -> Any:
        import urllib.request

        url = f"http://{self._host}:{self._port}{path}"
        with urllib.request.urlopen(url, timeout=CALL_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))

    def _pick_target(self, window) -> Optional[dict]:
        """Choose the debuggable page that matches the focused window title."""
        targets = [t for t in self._http_get("/json") if t.get("type") == "page"]
        if not targets:
            return None

        title = (getattr(window, "title", "") or "").lower()
        for target in targets:
            target_title = (target.get("title") or "").lower()
            if target_title and target_title in title:
                return target
        return targets[0]

    def _query_page(self, target: dict) -> list[dict]:
        """Evaluate a collector in the page and return plain dicts.

        Uses Runtime.evaluate rather than Accessibility.getFullAXTree because it
        returns roles, accessible names, placeholders and viewport rects in one
        round trip instead of one per node.
        """
        from websocket import create_connection  # provided by websocket-client

        ws_url = target.get("webSocketDebuggerUrl")
        if not ws_url:
            return []

        expression = """
        (() => {
          const SEL = 'a[href],button,input,select,textarea,[role],[onclick],[tabindex]';
          const out = [];
          for (const el of document.querySelectorAll(SEL)) {
            const r = el.getBoundingClientRect();
            if (r.width <= 4 || r.height <= 4) continue;
            const style = getComputedStyle(el);
            if (style.visibility === 'hidden' || style.display === 'none') continue;
            const role = el.getAttribute('role') ||
              (el.tagName === 'A' ? 'link' :
               el.tagName === 'BUTTON' ? 'button' :
               el.tagName === 'SELECT' ? 'combobox' :
               el.tagName === 'TEXTAREA' ? 'textbox' :
               el.tagName === 'INPUT' ? (el.type === 'search' ? 'searchbox' :
                                         el.type === 'checkbox' ? 'checkbox' :
                                         el.type === 'radio' ? 'radio' : 'textbox')
               : 'generic');
            const name = (el.getAttribute('aria-label') || el.getAttribute('title') ||
                          el.getAttribute('alt') || (el.innerText || '').trim().slice(0, 120) || '');
            out.push({
              role, name,
              placeholder: el.getAttribute('placeholder') || '',
              value: (el.value !== undefined && el.value !== null) ? String(el.value).slice(0, 120) : '',
              id: el.id || '',
              x: r.left, y: r.top, w: r.width, h: r.height,
              focused: document.activeElement === el,
              disabled: !!el.disabled,
              container: (el.closest('[role="banner"],[role="navigation"],[role="main"],header,nav,main,form')
                          || {}).getAttribute?.('aria-label') || ''
            });
            if (out.length >= %d) break;
          }
          return JSON.stringify({elements: out, dpr: window.devicePixelRatio,
                                 ox: window.screenX, oy: window.screenY,
                                 iw: window.innerWidth, ih: window.innerHeight,
                                 ow: window.outerWidth, oh: window.outerHeight});
        })()
        """ % MAX_ELEMENTS

        connection = create_connection(ws_url, timeout=CALL_TIMEOUT)
        try:
            connection.send(json.dumps({
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {"expression": expression, "returnByValue": True},
            }))
            raw = json.loads(connection.recv())
        finally:
            try:
                connection.close()
            except Exception:
                pass

        value = raw.get("result", {}).get("result", {}).get("value")
        if not value:
            return []
        return json.loads(value)

    def _to_elements(self, payload, window) -> list[ElementNode]:
        if not isinstance(payload, dict):
            return []

        raw_elements = payload.get("elements") or []
        # Viewport -> screen. The page's own screenX/screenY plus the chrome
        # height (outer minus inner) gives the content origin.
        origin_x = payload.get("ox", 0)
        origin_y = payload.get("oy", 0)
        chrome_h = max(0, payload.get("oh", 0) - payload.get("ih", 0))
        chrome_w = max(0, payload.get("ow", 0) - payload.get("iw", 0))

        elements: list[ElementNode] = []
        for index, raw in enumerate(raw_elements, start=1):
            try:
                left = int(origin_x + chrome_w // 2 + raw["x"])
                top = int(origin_y + chrome_h + raw["y"])
                width, height = int(raw["w"]), int(raw["h"])
            except Exception:
                continue

            name = (raw.get("name") or "").strip()
            placeholder = (raw.get("placeholder") or "").strip()
            automation_id = (raw.get("id") or "").strip()
            if not name and not placeholder and not automation_id:
                continue

            role = AX_ROLE_MAP.get((raw.get("role") or "").lower(), (raw.get("role") or "generic").lower())
            elements.append(
                ElementNode(
                    id=index,
                    role=role,
                    name=name[:160],
                    rect=(left, top, left + width, top + height),
                    center=(left + width // 2, top + height // 2),
                    value=(raw.get("value") or "")[:120],
                    placeholder=placeholder[:120],
                    # Everything CDP sees is page content by definition.
                    frame=FRAME_PAGE,
                    container=(raw.get("container") or "")[:60],
                    focused=bool(raw.get("focused")),
                    focusable=True,
                    enabled=not raw.get("disabled"),
                    offscreen=False,
                    automation_id=automation_id[:60],
                    source=SOURCE_DOM,
                )
            )

        return elements
