"""The element node schema the agent reasons over.

The old observation format gave the model only ``name``, ``control_type`` and
``bounds``. That is not enough to tell a browser's address bar apart from a
search box inside the page - they are both edit controls with plausible names -
which is exactly where the agent used to go wrong.

The fields that resolve that ambiguity are:

``frame``
    ``chrome`` for browser UI (omnibox, tabs, toolbar), ``page`` for anything
    inside the rendered document, ``app`` for ordinary desktop windows. This is
    the single most useful discriminator: "type the query into the YouTube
    search box" means ``frame == "page"``, never the omnibox.
``placeholder`` / ``value``
    An empty omnibox and an empty site search box differ only by their
    placeholder text.
``container``
    The nearest named ancestor region, so "the search box in the masthead" is
    expressible.
``focused`` / ``focusable``
    Lets a typing step assert that the intended field actually has focus
    before sending keystrokes.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

FRAME_CHROME = "chrome"
FRAME_PAGE = "page"
FRAME_APP = "app"

SOURCE_UIA = "uia"
SOURCE_DOM = "dom"
SOURCE_WIN32 = "win32"

# Roles that are worth offering to the model as click/type targets.
INTERACTIVE_ROLES = {
    "button", "checkbox", "combobox", "edit", "textbox", "hyperlink", "link",
    "listitem", "menuitem", "option", "radiobutton", "searchbox", "slider",
    "spinbutton", "splitbutton", "tab", "tabitem", "toolbar", "treeitem",
    "switch", "menuitemcheckbox", "menuitemradio",
}


@dataclass
class ElementNode:
    """One interactive control, from either the UIA tree or the DOM."""

    id: int
    role: str
    name: str
    rect: tuple[int, int, int, int]  # left, top, right, bottom (screen px)
    center: tuple[int, int]
    value: str = ""
    placeholder: str = ""
    frame: str = FRAME_APP
    container: str = ""
    focused: bool = False
    focusable: bool = False
    enabled: bool = True
    offscreen: bool = False
    automation_id: str = ""
    source: str = SOURCE_UIA

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]

    @property
    def is_interactive(self) -> bool:
        return self.role.lower() in INTERACTIVE_ROLES

    def contains(self, x: int, y: int) -> bool:
        left, top, right, bottom = self.rect
        return left <= x <= right and top <= y <= bottom

    def distance_to(self, x: int, y: int) -> float:
        cx, cy = self.center
        return ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5

    def search_text(self) -> str:
        """Everything a name-based lookup should be allowed to match against."""
        return " ".join(
            p for p in (self.name, self.placeholder, self.value, self.automation_id) if p
        ).lower()

    def to_dict(self, compact: bool = False) -> dict[str, Any]:
        """Serialise. ``compact`` drops defaults to keep prompts small."""
        data: dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "name": self.name,
            "rect": list(self.rect),
            "center": list(self.center),
            "frame": self.frame,
        }
        if not compact:
            data.update(
                {
                    "value": self.value,
                    "placeholder": self.placeholder,
                    "container": self.container,
                    "focused": self.focused,
                    "focusable": self.focusable,
                    "enabled": self.enabled,
                    "offscreen": self.offscreen,
                    "automation_id": self.automation_id,
                    "source": self.source,
                }
            )
            return data

        # Compact: only carry what actually differs from the default.
        if self.value:
            data["value"] = self.value
        if self.placeholder:
            data["placeholder"] = self.placeholder
        if self.container:
            data["container"] = self.container
        if self.focused:
            data["focused"] = True
        if not self.enabled:
            data["enabled"] = False
        return data


def elements_to_json(
    elements: list[ElementNode],
    limit: Optional[int] = None,
    compact: bool = True,
) -> str:
    """Render elements as a JSON array for the planner prompt."""
    subset = elements[:limit] if limit else elements
    return json.dumps([e.to_dict(compact=compact) for e in subset], ensure_ascii=False)


def elements_to_prompt(
    elements: list[ElementNode],
    limit: Optional[int] = 40,
) -> str:
    """Render the element graph as a labelled JSON block for the prompt."""
    if not elements:
        return "### Interactive Elements: (none detected)"

    subset = elements[:limit] if limit else elements
    frames = {e.frame for e in subset}
    header = ["### Interactive Elements (click/type by `id`)"]
    if FRAME_CHROME in frames and FRAME_PAGE in frames:
        header.append(
            "Note: `frame` is \"chrome\" for browser UI (address bar, tabs) and "
            "\"page\" for content inside the web page. A site's own search box is "
            "always frame=\"page\"."
        )
    header.append(elements_to_json(subset, compact=True))
    if limit and len(elements) > limit:
        header.append(f"({len(elements) - limit} further elements not shown)")
    return "\n".join(header)


def find_by_id(elements: list[ElementNode], element_id: int) -> Optional[ElementNode]:
    for element in elements:
        if element.id == element_id:
            return element
    return None


def find_by_name(
    elements: list[ElementNode],
    query: str,
    frame: Optional[str] = None,
    role: Optional[str] = None,
) -> Optional[ElementNode]:
    """Resolve a target by name, preferring exact matches and visible controls.

    ``frame`` lets a caller say "the search box in the page, not the omnibox",
    which is the whole reason the field exists.
    """
    if not query:
        return None

    q = query.lower().strip()
    candidates = [e for e in elements if not e.offscreen and e.enabled]
    if frame:
        candidates = [e for e in candidates if e.frame == frame]
    if role:
        candidates = [e for e in candidates if e.role.lower() == role.lower()]
    if not candidates:
        return None

    # Exact name, then exact placeholder, then substring - in that order, so a
    # button literally called "Search" beats one merely containing the word.
    for element in candidates:
        if element.name.lower().strip() == q:
            return element
    for element in candidates:
        if element.placeholder.lower().strip() == q:
            return element
    for element in candidates:
        if q in element.search_text():
            return element
    return None


def find_at_point(
    elements: list[ElementNode],
    x: int,
    y: int,
    tolerance_px: int = 30,
    max_size: int = 200,
) -> Optional[ElementNode]:
    """Snap a predicted coordinate onto a nearby small control.

    Large containers are skipped: snapping a click onto the centre of a
    full-width panel is worse than leaving the original coordinate alone.
    """
    best: Optional[ElementNode] = None
    best_dist = float("inf")

    for element in elements:
        if element.offscreen:
            continue
        too_big = element.width > max_size or element.height > max_size
        if element.contains(x, y) and not too_big:
            return element
        if too_big:
            continue
        dist = element.distance_to(x, y)
        if dist <= tolerance_px and dist < best_dist:
            best_dist = dist
            best = element

    return best
