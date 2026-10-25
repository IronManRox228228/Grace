"""Stateful Sim Desktop for Tape Drives.

Replaces the frozen screen in the tape harness with a deterministic state machine
for each tape. Generates synthetic screenshots from the element tree using PIL,
updates UIA/DOM element trees and window state on valid dispatched actions,
and verifies that the sim has reached its goal state.
"""

from __future__ import annotations

import dataclasses
import os
from io import BytesIO
from typing import Any, Optional

from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.harness.tape import Tape
from grace.intent.parser import Intent
from grace.perception.element_graph import ElementGraph
from grace.perception.elements import ElementNode


def render_element_tree_image(
    window_title: str,
    elements: list[ElementNode],
    width: int = 1920,
    height: int = 1080,
) -> bytes:
    """Render a clean synthetic desktop image from the element tree using PIL."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Window title bar
    draw.rectangle([0, 0, width, 40], fill=(225, 225, 225), outline=(180, 180, 180), width=1)
    draw.text((15, 12), window_title or "Desktop", fill=(30, 30, 30))

    # Elements
    for el in elements:
        if not el.rect or len(el.rect) < 4:
            continue
        left, top, right, bottom = el.rect
        if right <= left or bottom <= top:
            continue

        if el.role in ("edit", "document"):
            fill_color = (255, 255, 255)
            border_color = (0, 120, 215) if el.focused else (160, 160, 160)
            border_width = 2 if el.focused else 1
        elif el.role in ("button", "menuitem"):
            fill_color = (230, 230, 230)
            border_color = (0, 120, 215) if el.focused else (120, 120, 120)
            border_width = 2 if el.focused else 1
        else:
            fill_color = (240, 240, 240)
            border_color = (180, 180, 180)
            border_width = 1

        draw.rectangle([left, top, right, bottom], fill=fill_color, outline=border_color, width=border_width)

        display_text = str(el.value if el.value else (el.placeholder if el.placeholder else (el.name or "")))
        if display_text:
            text_x = left + 8
            text_y = top + max(2, (bottom - top - 14) // 2)
            draw.text((text_x, text_y), display_text[:60], fill=(20, 20, 20))

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@dataclasses.dataclass
class SimState:
    name: str
    window_title: str
    elements: list[ElementNode]
    is_goal: bool = False
    class_name: str = "SimWindow"


class BaseTapeSim:
    """Base deterministic state machine for a simulated desktop."""

    def __init__(self, tape_name: str):
        self.tape_name = tape_name
        self.states: dict[str, SimState] = {}
        self.current_state_name: str = "initial"
        self.dispatched_actions: list[dict[str, Any]] = []
        self._snapshot_cache: dict[str, ScreenSnapshot] = {}

    @property
    def current_state(self) -> SimState:
        return self.states[self.current_state_name]

    def is_goal_state(self) -> bool:
        return self.current_state.is_goal

    def get_snapshot(self) -> ScreenSnapshot:
        state = self.current_state
        if state.name in self._snapshot_cache:
            return self._snapshot_cache[state.name]

        window = WindowInfo(
            hwnd=1234,
            title=state.window_title,
            class_name=state.class_name,
            rect=(0, 0, 1920, 1080),
        )
        graph = ElementGraph(
            elements=state.elements,
            window=window,
            sources=("uia",),
        )
        png_bytes = render_element_tree_image(
            window_title=state.window_title,
            elements=state.elements,
            width=1920,
            height=1080,
        )
        snapshot = ScreenSnapshot(
            active_window=window,
            ocr_lines=[],
            width=1920,
            height=1080,
            image_width=1920,
            image_height=1080,
            dpi_scale=1.0,
            graph=graph,
            png_bytes=png_bytes,
        )
        self._snapshot_cache[state.name] = snapshot
        return snapshot

    def dispatch(self, intent: Intent, confirmed: bool = False) -> dict[str, Any]:
        raise NotImplementedError


class TwoStepSim(BaseTapeSim):
    """Deterministic state machine for `agentic_two_step`.

    Goal: Click the search box, type hello.
    State 0 ('initial'): search box unfocused, value empty.
    State 1 ('focused'): search box focused (blue outline, keyboard focus).
    State 2 ('typed'): search box has value 'hello' (GOAL STATE).
    State 3 ('submitted'): results show on Enter (GOAL STATE).
    """

    def __init__(self):
        super().__init__("agentic_two_step")
        self._setup_states()

    def _setup_states(self) -> None:
        title = "Grace test page - Edge"

        # State 0: initial (unfocused)
        s0_elements = [
            ElementNode(id=0, role="edit", name="Address and search bar", rect=(120, 60, 1700, 96),
                        center=(910, 78), placeholder="Search or enter web address", container="Toolbar",
                        focused=False, focusable=True, enabled=True),
            ElementNode(id=1, role="button", name="Refresh", rect=(80, 60, 116, 96),
                        center=(98, 78), container="Toolbar", focused=False, focusable=True, enabled=True),
            ElementNode(id=2, role="edit", name="Search", rect=(700, 300, 1220, 348),
                        center=(960, 324), placeholder="Search this site", container="Main",
                        focused=False, focusable=True, enabled=True, value=""),
            ElementNode(id=3, role="button", name="Submit", rect=(1230, 300, 1330, 348),
                        center=(1280, 324), container="Main", focused=False, focusable=True, enabled=True),
        ]
        self.states["initial"] = SimState(
            name="initial", window_title=title, elements=s0_elements, is_goal=False,
            class_name="Chrome_WidgetWin_1",
        )

        # State 1: focused
        s1_elements = [
            ElementNode(id=0, role="edit", name="Address and search bar", rect=(120, 60, 1700, 96),
                        center=(910, 78), placeholder="Search or enter web address", container="Toolbar",
                        focused=False, focusable=True, enabled=True),
            ElementNode(id=1, role="button", name="Refresh", rect=(80, 60, 116, 96),
                        center=(98, 78), container="Toolbar", focused=False, focusable=True, enabled=True),
            ElementNode(id=2, role="edit", name="Search", rect=(700, 300, 1220, 348),
                        center=(960, 324), placeholder="Search this site", container="Main",
                        focused=True, focusable=True, enabled=True, value=""),
            ElementNode(id=3, role="button", name="Submit", rect=(1230, 300, 1330, 348),
                        center=(1280, 324), container="Main", focused=False, focusable=True, enabled=True),
        ]
        self.states["focused"] = SimState(
            name="focused", window_title=title, elements=s1_elements, is_goal=False,
            class_name="Chrome_WidgetWin_1",
        )

        # State 2: typed (GOAL STATE)
        s2_elements = [
            ElementNode(id=0, role="edit", name="Address and search bar", rect=(120, 60, 1700, 96),
                        center=(910, 78), placeholder="Search or enter web address", container="Toolbar",
                        focused=False, focusable=True, enabled=True),
            ElementNode(id=1, role="button", name="Refresh", rect=(80, 60, 116, 96),
                        center=(98, 78), container="Toolbar", focused=False, focusable=True, enabled=True),
            ElementNode(id=2, role="edit", name="Search", rect=(700, 300, 1220, 348),
                        center=(960, 324), placeholder="Search this site", container="Main",
                        focused=True, focusable=True, enabled=True, value="hello"),
            ElementNode(id=3, role="button", name="Submit", rect=(1230, 300, 1330, 348),
                        center=(1280, 324), container="Main", focused=False, focusable=True, enabled=True),
        ]
        self.states["typed"] = SimState(
            name="typed", window_title=title, elements=s2_elements, is_goal=True,
            class_name="Chrome_WidgetWin_1",
        )

        # State 3: submitted (GOAL STATE)
        s3_elements = [
            ElementNode(id=0, role="edit", name="Address and search bar", rect=(120, 60, 1700, 96),
                        center=(910, 78), placeholder="Search or enter web address", container="Toolbar",
                        focused=False, focusable=True, enabled=True),
            ElementNode(id=1, role="button", name="Refresh", rect=(80, 60, 116, 96),
                        center=(98, 78), container="Toolbar", focused=False, focusable=True, enabled=True),
            ElementNode(id=2, role="edit", name="Search", rect=(700, 300, 1220, 348),
                        center=(960, 324), placeholder="Search this site", container="Main",
                        focused=False, focusable=True, enabled=True, value="hello"),
            ElementNode(id=3, role="button", name="Submit", rect=(1230, 300, 1330, 348),
                        center=(1280, 324), container="Main", focused=False, focusable=True, enabled=True),
            ElementNode(id=4, role="text", name="Search Results for 'hello'", rect=(700, 380, 1220, 420),
                        center=(960, 400), container="Main", focused=False, focusable=False, enabled=True),
        ]
        self.states["submitted"] = SimState(
            name="submitted", window_title=title, elements=s3_elements, is_goal=True,
            class_name="Chrome_WidgetWin_1",
        )

    def dispatch(self, intent: Intent, confirmed: bool = False) -> dict[str, Any]:
        self.dispatched_actions.append({"tool": intent.tool, "params": intent.params})
        tool = intent.tool
        params = intent.params or {}

        # Click action
        if tool == "cua_click":
            target_id = params.get("element_id") or params.get("element_index")
            target_name = (params.get("target_name") or params.get("name") or "").lower()
            x = params.get("x")
            y = params.get("y")
            coord_match = (x is not None and y is not None and 700 <= x <= 1220 and 300 <= y <= 348)

            is_search_box = (
                target_id == 2
                or coord_match
                or ("search" in target_name and "address" not in target_name)
            )

            if is_search_box:
                if self.current_state_name in ("initial", "focused"):
                    self.current_state_name = "focused"
                    return {"status": "ok", "result": {"ok": True, "message": "clicked", "verified": True, "sent": True}}
            elif target_id == 3 or "submit" in target_name:
                if self.current_state_name == "typed":
                    self.current_state_name = "submitted"
                    return {"status": "ok", "result": {"ok": True, "message": "clicked", "verified": True, "sent": True}}
                return {"status": "ok", "result": {"ok": True, "message": "clicked", "verified": True, "sent": True}}
            elif target_id == 1 or "refresh" in target_name:
                self.current_state_name = "initial"
                return {"status": "ok", "result": {"ok": True, "message": "clicked", "verified": True, "sent": True}}
            return {"status": "ok", "result": {"ok": True, "message": "clicked", "verified": True, "sent": True}}

        # Type text action
        if tool == "cua_type_text":
            text = params.get("text", "")
            if self.current_state_name in ("focused", "typed"):
                if "hello" in text.lower():
                    self.current_state_name = "typed"
                    return {"status": "ok", "result": {"ok": True, "message": "typed", "verified": True, "sent": True}}
                return {"status": "ok", "result": {"ok": True, "message": "typed", "verified": True, "sent": True}}
            # If typed before focusing, state unchanged
            return {"status": "ok", "result": {"ok": False, "message": "no focused element", "verified": False, "sent": True}}

        # Press key action
        if tool == "cua_press_key":
            key = str(params.get("key", "")).lower()
            if key in ("enter", "return") and self.current_state_name == "typed":
                self.current_state_name = "submitted"
                return {"status": "ok", "result": {"ok": True, "message": "pressed Enter", "verified": True, "sent": True}}
            return {"status": "ok", "result": {"ok": True, "message": f"pressed {key}", "verified": True, "sent": True}}

        if tool == "converse":
            return {"status": "ok"}

        return {"status": "ok"}


class ReadPdfSim(BaseTapeSim):
    """Deterministic state machine for `agentic_read_pdf`.

    Goal: Read report PDF to user.
    State 0 ('initial'): Notepad open with empty editor.
    State 1 ('file_found'): Files searched; Documents\\report.pdf found.
    State 2 ('pdf_read'): PDF text extracted/read (GOAL STATE).
    State 3 ('pdf_opened'): PDF opened in viewer (GOAL STATE).
    """

    def __init__(self, documents_root: Optional[str] = None):
        super().__init__("agentic_read_pdf")
        if documents_root is None:
            documents_root = (
                os.environ.get("GRACE_DOCUMENTS_DIR")
                or os.environ.get("DOCUMENTS_ROOT")
                or os.environ.get("DOCUMENTS_DIR")
            )
        if documents_root is None:
            user_profile = os.environ.get("USERPROFILE", os.path.expanduser("~"))
            documents_root = os.path.join(user_profile, "Documents")
        self._documents_root = documents_root
        self._pdf_path = os.path.join(self._documents_root, "report.pdf")
        self._setup_states()

    def _setup_states(self) -> None:
        title = "Untitled - Notepad"
        elements = [
            ElementNode(id=0, role="document", name="Text Editor", rect=(8, 60, 1912, 1040),
                        center=(960, 550), focused=True, focusable=True, enabled=True, value=""),
            ElementNode(id=1, role="menuitem", name="File", rect=(8, 30, 60, 58),
                        center=(34, 44), focused=False, focusable=True, enabled=True),
            ElementNode(id=2, role="menuitem", name="Edit", rect=(60, 30, 112, 58),
                        center=(86, 44), focused=False, focusable=True, enabled=True),
        ]
        self.states["initial"] = SimState(
            name="initial", window_title=title, elements=elements, is_goal=False,
            class_name="Notepad",
        )
        self.states["file_found"] = SimState(
            name="file_found", window_title=title, elements=elements, is_goal=False,
            class_name="Notepad",
        )
        self.states["pdf_read"] = SimState(
            name="pdf_read", window_title=title, elements=elements, is_goal=True,
            class_name="Notepad",
        )

        opened_elements = [
            ElementNode(id=0, role="document", name="report.pdf", rect=(100, 100, 1800, 1000),
                        center=(950, 550), focused=True, focusable=True, enabled=True,
                        value="Third quarter results exceeded projections."),
        ]
        self.states["pdf_opened"] = SimState(
            name="pdf_opened", window_title="report.pdf - Edge", elements=opened_elements, is_goal=True,
            class_name="Chrome_WidgetWin_1",
        )

    def dispatch(self, intent: Intent, confirmed: bool = False) -> dict[str, Any]:
        self.dispatched_actions.append({"tool": intent.tool, "params": intent.params})
        tool = intent.tool
        params = intent.params or {}

        if tool == "search_files":
            query = params.get("query", "").lower()
            self.current_state_name = "file_found"
            return {
                "status": "ok",
                "text": f"Here are the matching files:\n  1. {self._pdf_path}",
                "files": [self._pdf_path],
            }

        if tool == "read_pdf":
            path = params.get("path", "")
            self.current_state_name = "pdf_read"
            return {
                "status": "ok",
                "action": "read_pdf",
                "text": "[Excerpt 1]: Third quarter results exceeded projections.",
            }

        if tool == "summarize_pdf":
            self.current_state_name = "pdf_read"
            return {
                "status": "ok",
                "action": "summarize_pdf",
                "text": "[Section 1]: Third quarter results exceeded projections.",
                "speakable_summary": "Third quarter results exceeded projections.",
            }

        if tool == "open_file":
            name = params.get("name", "")
            if "report.pdf" in name.lower():
                self.current_state_name = "pdf_opened"
                return {"status": "ok", "text": "I've opened report.pdf."}
            return {"status": "error", "error": f"File '{name}' not found."}

        if tool == "open_app":
            return {"status": "ok", "text": f"Opened {params.get('name', 'app')}."}

        if tool == "converse":
            return {"status": "ok"}

        return {"status": "ok"}


class GenericTapeSim(BaseTapeSim):
    """Fallback simulator reconstructing states from tape snapshots."""

    def __init__(self, tape: Tape):
        scenario_name = tape.meta.get("scenario") or os.path.basename(os.path.normpath(tape.dir)) or "generic_tape"
        super().__init__(scenario_name)
        from grace.harness.generate import blank_screen
        from grace.harness.replay import _rebuild_snapshot

        raw_snapshots = [r["snapshot"] for r in tape.snapshots] if tape.snapshots else [blank_screen()]
        self._snapshots = [_rebuild_snapshot(r) for r in raw_snapshots]
        self._index = 0
        self._raw_records = [
            {"tool": r.get("tool"), "result": r["result"]} for r in tape.dispatch
        ]
        self._last_served = None

        initial_snap = self._snapshots[0]
        elements = getattr(initial_snap.graph, "elements", []) if initial_snap.graph else []
        title = initial_snap.active_window.title if initial_snap.active_window else "Window"
        self.states["initial"] = SimState(name="initial", window_title=title, elements=elements, is_goal=False)

    def is_goal_state(self) -> bool:
        return self._index >= len(self._snapshots) - 1

    def get_snapshot(self) -> ScreenSnapshot:
        idx = min(self._index, len(self._snapshots) - 1)
        snap = self._snapshots[idx]
        if not snap.png_bytes:
            elements = getattr(snap.graph, "elements", []) if snap.graph else []
            title = snap.active_window.title if snap.active_window else "Window"
            snap.png_bytes = render_element_tree_image(title, elements, snap.width, snap.height)
        return snap

    def dispatch(self, intent: Intent, confirmed: bool = False) -> dict[str, Any]:
        self.dispatched_actions.append({"tool": intent.tool, "params": intent.params})
        matched_result = None

        for i, record in enumerate(self._raw_records):
            if record["tool"] == intent.tool:
                matched_result = self._raw_records.pop(i)["result"]
                self._index = min(self._index + 1, len(self._snapshots) - 1)
                break

        if matched_result is None and self._raw_records:
            matched_result = self._raw_records.pop(0)["result"]

        if matched_result is not None:
            self._last_served = matched_result
            return matched_result
        if self._last_served is not None:
            return self._last_served
        return {"status": "ok"}


class YouTubeMusicSim(BaseTapeSim):
    """Deterministic state machine for `agentic_youtube_music`.

    Goal: Open YouTube Music, search for "Signal" by Home, and play it.
    State 0 ('initial' / 'browser_open'): Browser open on music.youtube.com.
    State 1 ('search_focused'): Search box has keyboard focus.
    State 2 ('query_typed'): "signal by home" typed into search box.
    State 3 ('results_shown'): Results shown including "Signal" and "Home".
    State 4 ('song_playing'): Player bar shows "Signal · Home" playing (GOAL STATE).
    """

    def __init__(self):
        super().__init__("agentic_youtube_music")
        self._query = ""
        self._setup_states()

    def _setup_states(self) -> None:
        title = "YouTube Music - Edge"

        # State 0: browser_open
        s0_elements = [
            ElementNode(id=0, role="edit", name="Address and search bar", rect=(120, 60, 1700, 96),
                        center=(910, 78), placeholder="Search or enter web address", container="Toolbar",
                        focused=False, focusable=True, enabled=True, value="music.youtube.com"),
            ElementNode(id=1, role="button", name="Search", rect=(600, 60, 640, 100),
                        center=(620, 80), container="Header", focused=False, focusable=True, enabled=True),
            ElementNode(id=2, role="edit", name="Search songs, albums, artists", rect=(650, 60, 1200, 100),
                        center=(925, 80), placeholder="Search songs, albums, artists", container="Header",
                        focused=False, focusable=True, enabled=True, value=""),
            ElementNode(id=3, role="text", name="Home", rect=(100, 200, 300, 240),
                        center=(200, 220), container="Navigation", focused=False, focusable=False, enabled=True),
            ElementNode(id=4, role="text", name="Explore", rect=(100, 260, 300, 300),
                        center=(200, 280), container="Navigation", focused=False, focusable=False, enabled=True),
            ElementNode(id=5, role="text", name="Library", rect=(100, 320, 300, 360),
                        center=(200, 340), container="Navigation", focused=False, focusable=False, enabled=True),
        ]
        self.states["initial"] = SimState(
            name="initial", window_title=title, elements=s0_elements, is_goal=False,
            class_name="Chrome_WidgetWin_1",
        )
        self.states["browser_open"] = self.states["initial"]

        # State 1: search_focused
        s1_elements = [
            ElementNode(id=0, role="edit", name="Address and search bar", rect=(120, 60, 1700, 96),
                        center=(910, 78), placeholder="Search or enter web address", container="Toolbar",
                        focused=False, focusable=True, enabled=True, value="music.youtube.com"),
            ElementNode(id=1, role="button", name="Search", rect=(600, 60, 640, 100),
                        center=(620, 80), container="Header", focused=False, focusable=True, enabled=True),
            ElementNode(id=2, role="edit", name="Search songs, albums, artists", rect=(650, 60, 1200, 100),
                        center=(925, 80), placeholder="Search songs, albums, artists", container="Header",
                        focused=True, focusable=True, enabled=True, value=self._query),
            ElementNode(id=3, role="text", name="Home", rect=(100, 200, 300, 240),
                        center=(200, 220), container="Navigation", focused=False, focusable=False, enabled=True),
            ElementNode(id=4, role="text", name="Explore", rect=(100, 260, 300, 300),
                        center=(200, 280), container="Navigation", focused=False, focusable=False, enabled=True),
            ElementNode(id=5, role="text", name="Library", rect=(100, 320, 300, 360),
                        center=(200, 340), container="Navigation", focused=False, focusable=False, enabled=True),
        ]
        self.states["search_focused"] = SimState(
            name="search_focused", window_title=title, elements=s1_elements, is_goal=False,
            class_name="Chrome_WidgetWin_1",
        )

        # State 2: query_typed
        s2_elements = [
            ElementNode(id=0, role="edit", name="Address and search bar", rect=(120, 60, 1700, 96),
                        center=(910, 78), placeholder="Search or enter web address", container="Toolbar",
                        focused=False, focusable=True, enabled=True, value="music.youtube.com"),
            ElementNode(id=1, role="button", name="Search", rect=(600, 60, 640, 100),
                        center=(620, 80), container="Header", focused=False, focusable=True, enabled=True),
            ElementNode(id=2, role="edit", name="Search songs, albums, artists", rect=(650, 60, 1200, 100),
                        center=(925, 80), placeholder="Search songs, albums, artists", container="Header",
                        focused=True, focusable=True, enabled=True, value="signal by home"),
            ElementNode(id=3, role="text", name="Home", rect=(100, 200, 300, 240),
                        center=(200, 220), container="Navigation", focused=False, focusable=False, enabled=True),
        ]
        self.states["query_typed"] = SimState(
            name="query_typed", window_title=title, elements=s2_elements, is_goal=False,
            class_name="Chrome_WidgetWin_1",
        )

        # State 3: results_shown
        s3_title = 'Search results for "signal by home" - YouTube Music - Edge'
        s3_elements = [
            ElementNode(id=0, role="edit", name="Address and search bar", rect=(120, 60, 1700, 96),
                        center=(910, 78), placeholder="Search or enter web address", container="Toolbar",
                        focused=False, focusable=True, enabled=True, value="music.youtube.com/search?q=signal+by+home"),
            ElementNode(id=2, role="edit", name="Search songs, albums, artists", rect=(650, 60, 1200, 100),
                        center=(925, 80), placeholder="Search songs, albums, artists", container="Header",
                        focused=False, focusable=True, enabled=True, value="signal by home"),
            ElementNode(id=10, role="button", name="Signal - Home (Play song)", rect=(300, 200, 1200, 260),
                        center=(750, 230), container="Results", focused=False, focusable=True, enabled=True),
            ElementNode(id=11, role="button", name="Play", rect=(1100, 210, 1180, 250),
                        center=(1140, 230), container="Results", focused=False, focusable=True, enabled=True),
            ElementNode(id=12, role="text", name="Signal · Home", rect=(300, 260, 1200, 300),
                        center=(750, 280), container="Results", focused=False, focusable=False, enabled=True),
        ]
        self.states["results_shown"] = SimState(
            name="results_shown", window_title=s3_title, elements=s3_elements, is_goal=False,
            class_name="Chrome_WidgetWin_1",
        )

        # State 4: song_playing (GOAL STATE)
        s4_title = "Signal · Home - YouTube Music - Edge"
        s4_elements = [
            ElementNode(id=0, role="edit", name="Address and search bar", rect=(120, 60, 1700, 96),
                        center=(910, 78), placeholder="Search or enter web address", container="Toolbar",
                        focused=False, focusable=True, enabled=True, value="music.youtube.com/watch?v=signal"),
            ElementNode(id=10, role="button", name="Signal - Home (Play song)", rect=(300, 200, 1200, 260),
                        center=(750, 230), container="Results", focused=False, focusable=True, enabled=True),
            ElementNode(id=20, role="text", name="Signal · Home", rect=(200, 980, 600, 1040),
                        center=(400, 1010), container="Player bar", focused=False, focusable=False, enabled=True,
                        value="Signal · Home"),
            ElementNode(id=21, role="button", name="Pause", rect=(930, 980, 990, 1040),
                        center=(960, 1010), container="Player bar", focused=False, focusable=True, enabled=True),
            ElementNode(id=22, role="text", name="Playing", rect=(1010, 980, 1100, 1040),
                        center=(1055, 1010), container="Player bar", focused=False, focusable=False, enabled=True,
                        value="Playing"),
        ]
        self.states["song_playing"] = SimState(
            name="song_playing", window_title=s4_title, elements=s4_elements, is_goal=True,
            class_name="Chrome_WidgetWin_1",
        )

    def is_goal_state(self) -> bool:
        state = self.states.get(self.current_state_name)
        if not state:
            return False
        has_player_bar = any(
            e.container == "Player bar" and "signal · home" in (e.name or e.value or "").lower()
            for e in state.elements
        )
        return state.is_goal or has_player_bar

    def dispatch(self, intent: Intent, confirmed: bool = False) -> dict[str, Any]:
        self._snapshot_cache.clear()
        self.dispatched_actions.append({"tool": intent.tool, "params": intent.params})
        tool = intent.tool
        params = intent.params or {}

        if tool in ("open_app", "open_url"):
            if self.current_state_name == "initial":
                self.current_state_name = "browser_open"
            return {"status": "ok", "text": "Opened YouTube Music."}

        if tool == "cua_click":
            target_id = params.get("element_id") or params.get("element_index")
            target_name = (params.get("target_name") or params.get("name") or "").lower()
            x = params.get("x")
            y = params.get("y")

            is_play_song = (
                target_id in (10, 11, 12)
                or any(k in target_name for k in ("signal", "play", "home"))
                or (x is not None and y is not None and 300 <= x <= 1200 and 200 <= y <= 300)
            )
            if is_play_song and self.current_state_name in ("results_shown", "query_typed", "search_focused"):
                self.current_state_name = "song_playing"
                return {"status": "ok", "result": {"ok": True, "message": "playing Signal · Home", "verified": True, "sent": True}}

            # Clicking the search button (element 1) submits the search if query is typed
            is_search_btn = (
                target_id == 1
                or target_name == "search"
                or (x is not None and y is not None and 600 <= x <= 645 and 60 <= y <= 100)
            )
            if is_search_btn and self._query:
                self.current_state_name = "results_shown"
                return {"status": "ok", "result": {"ok": True, "message": "submitted search", "verified": True, "sent": True}}

            is_search = (
                target_id in (1, 2)
                or "search" in target_name
                or (x is not None and y is not None and 600 <= x <= 1200 and 60 <= y <= 100)
            )
            if is_search:
                if self._query:
                    self.current_state_name = "query_typed"
                else:
                    self.current_state_name = "search_focused"
                return {"status": "ok", "result": {"ok": True, "message": "clicked search", "verified": True, "sent": True}}

            return {"status": "ok", "result": {"ok": True, "message": "clicked", "verified": True, "sent": True}}

        if tool == "cua_type_text":
            text = (params.get("text") or "").strip()
            self._query = text
            # Update element 2 in states
            for sname in ("search_focused", "query_typed", "results_shown"):
                if sname in self.states:
                    for el in self.states[sname].elements:
                        if el.id == 2:
                            el.value = text
            if "\n" in text or "\r" in text:
                self.current_state_name = "results_shown"
                return {"status": "ok", "result": {"ok": True, "message": "typed and submitted query", "verified": True, "sent": True}}
            if "signal" in text.lower() or "home" in text.lower():
                self.current_state_name = "query_typed"
                return {"status": "ok", "result": {"ok": True, "message": "typed query", "verified": True, "sent": True}}
            return {"status": "ok", "result": {"ok": True, "message": "typed", "verified": True, "sent": True}}

        if tool == "cua_press_key":
            key = str(params.get("key", "")).lower().strip()
            if any(k in key for k in ("enter", "return")) and self.current_state_name in ("query_typed", "search_focused", "results_shown"):
                self.current_state_name = "results_shown"
                return {"status": "ok", "result": {"ok": True, "message": "submitted search", "verified": True, "sent": True}}
            return {"status": "ok", "result": {"ok": True, "message": f"pressed {key}", "verified": True, "sent": True}}

        if tool == "converse":
            return {"status": "ok"}

        return {"status": "ok"}


class PdfToNotesSim(BaseTapeSim):
    """Deterministic state machine for `agentic_pdf_to_notes`.

    Goal: Find the deadline in report PDF and put it in a new note.
    States:
      - PDF found and read (text contains "Final submission deadline: 14 November")
      - Notepad open
      - The text "14 November" typed
    Goal check: the Notepad buffer contains "14 November".
    """

    def __init__(self, documents_root: Optional[str] = None):
        super().__init__("agentic_pdf_to_notes")
        if documents_root is None:
            documents_root = (
                os.environ.get("GRACE_DOCUMENTS_DIR")
                or os.environ.get("DOCUMENTS_ROOT")
                or os.environ.get("DOCUMENTS_DIR")
            )
        if documents_root is None:
            user_profile = os.environ.get("USERPROFILE", os.path.expanduser("~"))
            documents_root = os.path.join(user_profile, "Documents")
        self._documents_root = documents_root
        self._pdf_path = os.path.join(self._documents_root, "report.pdf")
        self._notepad_open = False
        self._pdf_read = False
        self._buffer = ""
        self._setup_states()

    def _setup_states(self) -> None:
        desktop_elements = [
            ElementNode(id=0, role="listitem", name="report.pdf", rect=(40, 40, 120, 120),
                        center=(80, 80), focused=False, focusable=True, enabled=True),
            ElementNode(id=1, role="button", name="Start", rect=(0, 1040, 60, 1080),
                        center=(30, 1060), focused=False, focusable=True, enabled=True),
        ]
        self.states["initial"] = SimState(
            name="initial", window_title="Desktop", elements=desktop_elements, is_goal=False,
            class_name="Progman",
        )
        self.states["file_found"] = SimState(
            name="file_found", window_title="Desktop", elements=desktop_elements, is_goal=False,
            class_name="Progman",
        )
        self.states["pdf_read"] = SimState(
            name="pdf_read", window_title="Desktop", elements=desktop_elements, is_goal=False,
            class_name="Progman",
        )

        opened_elements = [
            ElementNode(id=0, role="document", name="report.pdf", rect=(100, 100, 1800, 1000),
                        center=(950, 550), focused=True, focusable=True, enabled=True,
                        value="Final submission deadline: 14 November"),
        ]
        self.states["pdf_opened"] = SimState(
            name="pdf_opened", window_title="report.pdf - Edge", elements=opened_elements, is_goal=False,
            class_name="Chrome_WidgetWin_1",
        )

        notepad_elements = [
            ElementNode(id=0, role="document", name="Text Editor", rect=(8, 60, 1912, 1040),
                        center=(960, 550), focused=True, focusable=True, enabled=True, value=""),
            ElementNode(id=1, role="menuitem", name="File", rect=(8, 30, 60, 58),
                        center=(34, 44), focused=False, focusable=True, enabled=True),
            ElementNode(id=2, role="menuitem", name="Edit", rect=(60, 30, 112, 58),
                        center=(86, 44), focused=False, focusable=True, enabled=True),
        ]
        self.states["notepad_open"] = SimState(
            name="notepad_open", window_title="Untitled - Notepad", elements=notepad_elements, is_goal=False,
            class_name="Notepad",
        )

        typed_elements = [
            ElementNode(id=0, role="document", name="Text Editor", rect=(8, 60, 1912, 1040),
                        center=(960, 550), focused=True, focusable=True, enabled=True, value="14 November"),
            ElementNode(id=1, role="menuitem", name="File", rect=(8, 30, 60, 58),
                        center=(34, 44), focused=False, focusable=True, enabled=True),
            ElementNode(id=2, role="menuitem", name="Edit", rect=(60, 30, 112, 58),
                        center=(86, 44), focused=False, focusable=True, enabled=True),
        ]
        self.states["note_typed"] = SimState(
            name="note_typed", window_title="*Untitled - Notepad", elements=typed_elements, is_goal=True,
            class_name="Notepad",
        )

    def is_goal_state(self) -> bool:
        state = self.states.get(self.current_state_name)
        if not state:
            return False
        for el in state.elements:
            if el.role == "document" and "14 november" in (el.value or "").lower():
                return True
        return False

    def dispatch(self, intent: Intent, confirmed: bool = False) -> dict[str, Any]:
        self.dispatched_actions.append({"tool": intent.tool, "params": intent.params})
        tool = intent.tool
        params = intent.params or {}

        if tool == "search_files":
            if self.current_state_name == "initial":
                self.current_state_name = "file_found"
            return {
                "status": "ok",
                "text": f"Here are the matching files:\n  1. {self._pdf_path}",
                "files": [self._pdf_path],
            }

        if tool in ("read_pdf", "summarize_pdf"):
            self._pdf_read = True
            if self.current_state_name in ("initial", "file_found"):
                self.current_state_name = "pdf_read"
            return {
                "status": "ok",
                "action": "read_pdf",
                "text": "Project Report\nFinal submission deadline: 14 November\nPlease submit deliverables on time.",
            }

        if tool == "open_file" or ((tool in ("cua_click", "cua_press_key")) and (
            "report" in (params.get("target_name") or params.get("name") or "").lower()
            or "pdf" in (params.get("target_name") or params.get("name") or "").lower()
            or ((params.get("element_id") or params.get("element_index")) == 0 and self.current_state_name in ("initial", "file_found", "pdf_read"))
            or "d" in str(params.get("key", "")).lower()
            or "enter" in str(params.get("key", "")).lower()
        )):
            self._pdf_read = True
            self.current_state_name = "pdf_opened"
            return {
                "status": "ok",
                "result": {
                    "ok": True,
                    "message": "opened report.pdf",
                    "verified": True,
                    "sent": True,
                    "text": "Final submission deadline: 14 November",
                },
                "text": "Final submission deadline: 14 November",
            }

        if tool in ("cua_click", "cua_press_key"):
            target_name = (params.get("target_name") or params.get("name") or "").lower()
            if "notepad" in target_name or "note" in target_name:
                self._notepad_open = True
                self.current_state_name = "notepad_open"
                return {"status": "ok", "result": {"ok": True, "message": "clicked", "verified": True, "sent": True}}

            return {"status": "ok", "result": {"ok": True, "message": "action executed", "verified": True, "sent": True}}

        if tool in ("open_app", "cua_activate", "cua_list_windows"):
            name = (params.get("name") or params.get("window_name") or params.get("title") or "").lower()
            if "notepad" in name or "note" in name or tool in ("cua_activate", "cua_list_windows"):
                self._notepad_open = True
                if "14 november" in self._buffer.lower():
                    self.current_state_name = "note_typed"
                else:
                    self.current_state_name = "notepad_open"
                return {"status": "ok", "text": "I've opened Notepad."}
            return {"status": "ok", "text": f"Opened {name}."}

        if tool == "cua_type_text":
            text = params.get("text", "")
            if self._notepad_open or self.current_state_name in ("notepad_open", "note_typed", "pdf_opened"):
                self._buffer = text
                if "14 november" in text.lower():
                    self.current_state_name = "note_typed"
                    for el in self.states["note_typed"].elements:
                        if el.role == "document":
                            el.value = text
                    return {"status": "ok", "result": {"ok": True, "message": "typed", "verified": True, "sent": True}}
                else:
                    for el in self.states["notepad_open"].elements:
                        if el.role == "document":
                            el.value = text
                    return {"status": "ok", "result": {"ok": True, "message": "typed", "verified": True, "sent": True}}
            return {"status": "ok", "result": {"ok": False, "message": "no text editor open", "verified": False, "sent": True}}

        if tool == "converse":
            return {"status": "ok"}

        return {"status": "ok"}


def create_sim_for_tape(tape: Any, doc_root: Optional[str] = None) -> BaseTapeSim:
    scenario = (tape.meta.get("scenario") or "") if hasattr(tape, "meta") and isinstance(tape.meta, dict) else ""
    tape_dir = getattr(tape, "dir", "") or getattr(tape, "directory", "") or ""
    name = f"{scenario} {tape_dir}".lower()

    if "agentic_two_step" in name:
        return TwoStepSim()
    if "agentic_read_pdf" in name:
        root = (
            doc_root
            or os.environ.get("GRACE_DOCUMENTS_DIR")
            or os.environ.get("DOCUMENTS_ROOT")
            or os.environ.get("DOCUMENTS_DIR")
        )
        return ReadPdfSim(documents_root=root)
    if "agentic_youtube_music" in name:
        return YouTubeMusicSim()
    if "agentic_pdf_to_notes" in name:
        root = (
            doc_root
            or os.environ.get("GRACE_DOCUMENTS_DIR")
            or os.environ.get("DOCUMENTS_ROOT")
            or os.environ.get("DOCUMENTS_DIR")
        )
        return PdfToNotesSim(documents_root=root)
    return GenericTapeSim(tape)


class SimPerception:
    """Perception provider backed by the desktop simulator."""

    def __init__(self, sim: BaseTapeSim):
        self._sim = sim

    def capture_snapshot(self) -> ScreenSnapshot:
        return self._sim.get_snapshot()

    async def capture_snapshot_async(self) -> ScreenSnapshot:
        return self._sim.get_snapshot()


class SimDispatcher:
    """Dispatcher stand-in executing against the desktop simulator."""

    def __init__(self, sim: BaseTapeSim, inner_dispatcher: Any):
        self._sim = sim
        self._inner = inner_dispatcher
        self.calls: list[dict[str, Any]] = []

    async def execute(self, intent: Intent, confirmed: bool = False) -> dict[str, Any]:
        from grace.tools.dispatcher import Dispatcher

        refusal = Dispatcher.confirmation_required(intent, confirmed)
        if refusal is not None:
            return refusal

        self.calls.append({"tool": intent.tool, "params": intent.params})

        # Emit tool started/finished for event contract parity
        if not intent.tool.startswith("cua_"):
            await self._inner._emit_tool_started(intent.tool, intent.params)
        else:
            await self._inner._emit_cua_started(intent.tool[4:])
        await self._inner._emit_tool_finished()

        self._sim._snapshot_cache.clear()
        return self._sim.dispatch(intent, confirmed=confirmed)
