"""Desktop Perception Engine for Grace Agentic Loop.

Provides hardware-accelerated OCR (Windows Media OCR / PyTesseract)
and Win32 UI control hierarchy context to feed Gemma's reasoning loop.
"""

import asyncio
import hashlib
import io
import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional, List

from grace.harness import get_recorder

logger = logging.getLogger("grace.agent.perception")


def _snapshot_tape(snapshot: "ScreenSnapshot") -> dict:
    """Serialise a snapshot for a session tape.

    The full element list is recorded in order and uncompacted: this is the
    parity surface for the Rust UIA walker, and ``compact=True`` drops exactly
    the fields (source, container, focus, offscreen) most likely to reveal a
    tree-walk difference. The PNG is reduced to a digest and its dimensions -
    replay never needs the pixels, only the guarantee that the same image was
    produced, and full screenshots would make tapes unstoreable.
    """
    graph = snapshot.graph
    window = snapshot.active_window

    return {
        "window": (
            None if window is None
            else {
                # hwnd is a per-run handle value and would defeat tape diffing;
                # the identity that matters is title + class.
                "title": window.title,
                "class_name": window.class_name,
                "rect": list(window.rect),
            }
        ),
        "width": snapshot.width,
        "height": snapshot.height,
        "image_width": snapshot.image_width,
        "image_height": snapshot.image_height,
        "dpi_scale": snapshot.dpi_scale,
        "png_digest": (
            hashlib.sha256(snapshot.png_bytes).hexdigest()[:16]
            if snapshot.png_bytes else None
        ),
        "png_bytes": len(snapshot.png_bytes) if snapshot.png_bytes else 0,
        "graph_sources": list(graph.sources) if graph is not None else [],
        "elements": (
            [element.to_dict() for element in graph.elements]
            if graph is not None else []
        ),
        "ocr_lines": [
            {"text": line.text, "bounding_box": list(line.bounding_box)}
            for line in (snapshot.ocr_lines or [])
        ],
        "legacy_ui_elements": len(snapshot.ui_elements or []),
    }


async def _await(awaitable):
    """Wrap a WinRT awaitable so ``asyncio.run`` will accept it.

    WinRT projections return ``IAsyncOperation``, which implements
    ``__await__`` but is not a coroutine - and ``asyncio.run`` checks for a
    coroutine specifically, raising "a coroutine was expected, got
    DataWriterStoreOperation". Every WinRT OCR call therefore failed on its
    first await, was swallowed by the broad ``except`` in ``_perform_ocr``, and
    fell through to Tesseract.
    """
    return await awaitable


def _run_async(awaitable):
    """Resolve a WinRT awaitable from either a sync or an async context.

    This used to call nest_asyncio.apply() and then re-enter the *running*
    loop. Monkey-patching the loop to be reentrant from library code is a
    process-wide change that breaks the invariants asyncio.to_thread and
    aiohttp rely on, and it can deadlock if the coroutine ever needs the loop
    it is nested inside. Here a nested call is pushed to a short-lived thread
    with its own loop instead, so the outer loop is never re-entered.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is None:
        return asyncio.run(_await(awaitable))

    result: dict = {}

    def _worker():
        try:
            result["value"] = asyncio.run(_await(awaitable))
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller's thread
            result["error"] = e

    thread = threading.Thread(target=_worker, name="grace-winrt-ocr", daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _word_rect(word) -> Optional[tuple[int, int, int, int]]:
    """Read a WinRT OcrWord's rect, tolerating either naming convention."""
    rect = getattr(word, "bounding_rect", None)
    if rect is None:
        rect = getattr(word, "boundingRect", None)
    if rect is None:
        return None
    try:
        return (int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    except Exception:
        return None


#: The two ways a window can present itself to the agent.
OBSERVABILITY_RICH = "rich"
OBSERVABILITY_BLIND = "blind"

#: Fewest actionable controls a window can report and still be treated as
#: readable. Overridable via OBSERVABILITY_MIN_ACTIONABLE; see
#: ``ScreenSnapshot.observability`` for why a count is enough and why this
#: number leans high.
#:
#: Calibrated against real captures rather than guessed. A Windows Terminal
#: window - whose contents UIA does not expose at all - reports exactly five:
#: one tab, three scrollbar arrows, and the System menu. Window furniture is
#: worth about five controls, so five is the wrong side of the line.
DEFAULT_MIN_ACTIONABLE = 8


#: Most OCR lines promoted to elements. The screenshot only has room for so
#: many legible badges, and MAX_MARKS bounds the drawing anyway.
MAX_OCR_ELEMENTS = 30

#: Shortest recognised string worth offering as a target. Single characters are
#: overwhelmingly OCR noise off window borders and icons.
MIN_OCR_TEXT_LEN = 2


def _ocr_elements(
    ocr_lines: list,
    screen_size: tuple,
    image_size: tuple,
    start_id: int,
) -> list:
    """Promote recognised text lines to element nodes.

    An app that does not implement UI Automation - WhatsApp is the case that
    forced this - reports its window frame and nothing else, so there is nothing
    to mark and nothing to click by id. Its content is still legible on screen,
    and OCR already runs for exactly this situation. Promoting those lines gives
    the rest of the stack something to work with without teaching it a second
    kind of target: they are ordinary elements with ``source="ocr"``, so marks,
    the legend, ``by_id`` and the click path all handle them unchanged.

    Boxes arrive in image space, because OCR runs on the downscaled screenshot.
    They are converted to screen space here, since that is what every element
    consumer assumes, and converted back down only for drawing.
    """
    from grace.perception.elements import ROLE_TEXT, SOURCE_OCR, ElementNode

    screen_w, screen_h = screen_size
    image_w, image_h = image_size
    if not image_w or not image_h:
        return []

    sx = screen_w / float(image_w)
    sy = screen_h / float(image_h)

    elements = []
    next_id = start_id
    for line in ocr_lines:
        text = (getattr(line, "text", "") or "").strip()
        if len(text) < MIN_OCR_TEXT_LEN:
            continue

        x, y, w, h = line.bounding_box
        if w <= 0 or h <= 0:
            continue

        left = int(round(x * sx))
        top = int(round(y * sy))
        right = int(round((x + w) * sx))
        bottom = int(round((y + h) * sy))

        elements.append(
            ElementNode(
                id=next_id,
                role=ROLE_TEXT,
                name=text,
                rect=(left, top, right, bottom),
                center=((left + right) // 2, (top + bottom) // 2),
                source=SOURCE_OCR,
            )
        )
        next_id += 1
        if len(elements) >= MAX_OCR_ELEMENTS:
            break

    return elements


def _min_actionable() -> int:
    try:
        from grace.config import Config

        return int(Config().observability_min_actionable)
    except Exception:
        return DEFAULT_MIN_ACTIONABLE


@dataclass
class OcrLine:
    """Extracted text line with bounding rectangle."""

    text: str
    bounding_box: tuple[int, int, int, int]  # x, y, width, height


@dataclass
class WindowInfo:
    """Active window metadata."""

    hwnd: int
    title: str
    class_name: str
    rect: tuple[int, int, int, int]  # left, top, right, bottom


@dataclass
class ScreenSnapshot:
    """Complete perception snapshot at a single point in time."""

    active_window: Optional[WindowInfo]
    ocr_lines: List[OcrLine]
    width: int
    height: int
    ui_elements: List[Any] = None
    png_bytes: Optional[bytes] = None
    # Fix Bug #2: DPI scale factor (logical -> physical) so the LLM's relative
    # (x, y) predictions map 1:1 to the physical screen pixel grid.
    dpi_scale: float = 1.0
    # The structured element graph. Carried on the snapshot so the click path
    # uses the exact same ids the model was shown, instead of re-walking the
    # tree and renumbering from a screen that has since changed.
    graph: Any = None
    # Dimensions of the image actually sent to the vision model, which is
    # downscaled. Grounding coordinates come back in *this* space.
    image_width: int = 0
    image_height: int = 0

    @property
    def actionable_count(self) -> int:
        """Controls on screen a step could actually aim at."""
        graph = self.graph
        if graph is not None:
            count = getattr(graph, "actionable_count", None)
            if count is not None:
                return count
            return len(graph)
        # The legacy flat list carries no enabled/offscreen flags, so every
        # entry counts. It is only reached when the graph failed to build.
        return len(self.ui_elements or [])

    @property
    def observability(self) -> str:
        """``rich`` or ``blind`` - whether the agent can see this window.

        This is the one judgement the rest of the loop keys off, instead of each
        caller re-deriving "does this look like enough elements". It decides
        which observation the planner is given and which kinds of target are
        legal, so it has to be a property of the snapshot rather than an opinion
        formed at each use site.

        The threshold is calibrated, not principled. No purely structural rule
        separates four window-frame buttons from four real ones - a scrollbar
        arrow and a Send button are both an enabled, onscreen ``button`` - so
        this counts, and the number comes from measuring real windows rather
        than from taste (see DEFAULT_MIN_ACTIONABLE).

        What makes a crude count acceptable is that its two errors do not cost
        the same. Calling a rich window blind wastes a screenshot and some
        vision tokens, and the goal still completes. Calling a blind window rich
        is what produced a 130-step run: the planner is handed the window frame,
        told it is the screen, and invents coordinates for everything it cannot
        find. So the threshold sits high enough to catch the second case and is
        allowed to make the first.
        """
        return OBSERVABILITY_RICH if self.actionable_count >= _min_actionable() else OBSERVABILITY_BLIND

    @property
    def is_blind(self) -> bool:
        return self.observability == OBSERVABILITY_BLIND

    def header_markdown(self) -> list[str]:
        """Window identity and geometry. Shared by both observation modes."""
        lines = []
        if self.active_window:
            w = self.active_window
            lines.append(f"### Focused Window: '{w.title}' (Class: {w.class_name})")
            lines.append(f"Bounds: [left: {w.rect[0]}, top: {w.rect[1]}, right: {w.rect[2]}, bottom: {w.rect[3]}]")
        else:
            lines.append("### Focused Window: None / Unknown")

        lines.append(f"Screen Dimensions: {self.width}x{self.height}")
        # Fix Bug #2: surface DPI scale so coordinates predict physical pixels.
        lines.append(f"DPI Scale: {self.dpi_scale:.2f} (logical -> physical)")
        return lines

    def to_markdown(self) -> str:
        """Format the screen state into clean markdown for Gemma."""
        lines = self.header_markdown()
        window_title_hint = self.active_window.title if self.active_window else ""

        # Prefer the structured graph: it carries role, value, placeholder and
        # the chrome/page distinction, none of which the flat list had.
        if self.graph is not None and len(self.graph):
            lines.append("")
            lines.append(self.graph.to_prompt(limit=40))
        elif self.ui_elements:
            lines.append("\n### Active Interactive Accessibility Controls (Use element_index to click):")
            for elem in self.ui_elements[:35]:
                cl, ct, cr, cb = elem.bounds
                # FIX BUG #2: Include window parameter hint in prompt
                window_str = f', window="{window_title_hint}"' if window_title_hint else ""
                lines.append(f"[{elem.index}] {elem.control_type}: \"{elem.name}\" | Bounds: [{cl}, {ct}, {cr}, {cb}] -> `cua_click(element_index={elem.index}{window_str})`")

        if self.ocr_lines:
            lines.append("\n### Screen Text (Top OCR Output):")
            for i, line in enumerate(self.ocr_lines[:15], 1):  # Increased from 8 to 15
                x, y, w, h = line.bounding_box
                lines.append(f"{i}. \"{line.text}\" (x:{x}, y:{y}, w:{w}, h:{h})")
        else:
            lines.append("\n### Screen Text: (No readable text detected)")

        return "\n".join(lines)


@dataclass
class Observation:
    """What the planner is actually shown for one step.

    Carries the mode so the loop and the guards do not each re-derive it, and so
    a trace can say which one a step was planned in.
    """

    markdown: str
    image_b64: Optional[str] = None
    mode: str = OBSERVABILITY_RICH
    marks: int = 0

    @property
    def is_blind(self) -> bool:
        return self.mode == OBSERVABILITY_BLIND


def observe_for_planner(snapshot: "ScreenSnapshot") -> Observation:
    """Choose an observation that matches what this window will support.

    A textual element list and a screenshot are not interchangeable, and mixing
    them is where the agent went wrong: it was given a four-entry element list
    for an app it could not read, and answered with invented coordinates.

    So the two modes are kept whole rather than blended:

    ``rich``
        The element JSON, as before. No image - a model reasoning over exact
        ids does not need pixels, and sending both costs tokens and invites the
        model to trust the picture over the list.
    ``blind``
        A screenshot with numbered badges drawn on every target, plus a legend
        naming them. The badge numbers are element ids, so the answer resolves
        through the same path a rich-mode answer does.

    Falls back to the rich rendering if the image cannot be marked. That is a
    worse observation, but it is the one that already existed, and a goal that
    proceeds badly beats one that cannot proceed.
    """
    from grace.automation import som_overlay

    if not snapshot.is_blind:
        return Observation(markdown=snapshot.to_markdown(), mode=OBSERVABILITY_RICH)

    graph = snapshot.graph
    elements = list(getattr(graph, "elements", []) or [])
    image_size = (snapshot.image_width, snapshot.image_height)

    if snapshot.png_bytes and elements and all(image_size):
        marks = som_overlay.marks_for(
            elements, (snapshot.width, snapshot.height), image_size
        )
        marked = som_overlay.render_or_none(snapshot.png_bytes, marks) if marks else None
        if marked:
            import base64

            lines = snapshot.header_markdown()
            lines.append("")
            lines.append(
                "This app does not report its contents to Windows, so there is no "
                "element list. The screenshot has every target marked."
            )
            lines.append("")
            lines.append(som_overlay.legend(marks))
            return Observation(
                markdown="\n".join(lines),
                image_b64=base64.b64encode(marked).decode("ascii"),
                mode=OBSERVABILITY_BLIND,
                marks=len(marks),
            )

    logger.debug("Blind window but no markable screenshot; using the text rendering")
    return Observation(markdown=snapshot.to_markdown(), mode=OBSERVABILITY_BLIND)


class PerceptionEngine:
    """Captures desktop state, performs OCR, and extracts Win32 control metadata."""

    _shared_inspector = None
    _shared_graph_builder = None

    def __init__(self, run_ocr: bool = True):
        self._winrt_ocr_available = False
        self._run_ocr = run_ocr
        self._check_ocr_support()
        if PerceptionEngine._shared_inspector is None:
            from grace.automation.ui_inspector import UIInspector
            PerceptionEngine._shared_inspector = UIInspector()

    @classmethod
    def get_shared_inspector(cls):
        if cls._shared_inspector is None:
            from grace.automation.ui_inspector import UIInspector
            cls._shared_inspector = UIInspector()
        return cls._shared_inspector

    @classmethod
    def get_graph_builder(cls):
        """One builder process-wide, so its cache is actually shared."""
        if cls._shared_graph_builder is None:
            from grace.config import Config
            from grace.perception.element_graph import ElementGraphBuilder

            cls._shared_graph_builder = ElementGraphBuilder(cdp_port=Config().cdp_port)
        return cls._shared_graph_builder

    def _check_ocr_support(self):
        """Check availability of Windows.Media.Ocr."""
        try:
            import winrt.windows.media.ocr as ocr
            import winrt.windows.globalization as glob
            self._winrt_ocr_available = True
            logger.info("Windows Media OCR available")
        except ImportError:
            logger.info("Windows Media OCR (winrt) not installed, fallback OCR mode active")

    def capture_snapshot(self) -> ScreenSnapshot:
        """Capture desktop screenshot, run OCR, and query active window info."""
        from grace.util.timing import stage

        active_window = self._get_active_window_info()
        with stage("screenshot"):
            width, height, img_bytes, image_width, image_height = self._take_screenshot()

        graph = None
        try:
            with stage("element_graph") as graph_stage:
                graph = self.get_graph_builder().build()
                graph_stage.detail(f"{len(graph)} elements via {'+'.join(graph.sources) or 'none'}")
        except Exception as e:
            logger.debug(f"Element graph build skipped: {e}")

        # OCR is only needed where the accessibility tree comes up empty, e.g.
        # canvas-rendered or remote-desktop UIs. Skipping it when the graph is
        # rich removes the most expensive part of an observation.
        #
        # The test is the actionable count, not len(graph): a window reporting
        # four disabled or offscreen entries is as unreadable as one reporting
        # none, and the raw length said otherwise.
        ocr_lines = []
        actionable = graph.actionable_count if graph is not None else 0
        should_ocr = self._run_ocr and img_bytes and actionable < _min_actionable()
        if should_ocr:
            with stage("ocr") as ocr_stage:
                ocr_lines = self._perform_ocr(img_bytes, width, height)
                ocr_stage.detail(f"{len(ocr_lines)} lines")

            # Promoted into the graph so the ids the model is shown are the ids
            # the click path resolves. Appended rather than merged: the UIA
            # entries keep their ids, so a graph that is only partly blind does
            # not renumber under the planner mid-goal.
            if graph is not None and ocr_lines and image_width and image_height:
                promoted = _ocr_elements(
                    ocr_lines,
                    (width, height),
                    (image_width, image_height),
                    start_id=max((e.id for e in graph.elements), default=0) + 1,
                )
                if promoted:
                    graph.elements.extend(promoted)
                    logger.debug(
                        f"Promoted {len(promoted)} OCR lines to elements "
                        f"({actionable} actionable UIA controls)"
                    )

        # The legacy flat inspector is now only a fallback: when the graph has
        # elements it supersedes this entirely, and running both walked the UIA
        # tree twice per observation for nothing.
        ui_elements = []
        if graph is None or not len(graph):
            try:
                with stage("uia_legacy") as uia_stage:
                    ui_elements = self.get_shared_inspector().inspect_active_window()
                    uia_stage.detail(f"{len(ui_elements)} elements")
            except Exception as e:
                logger.debug(f"UIInspector query in perception engine skipped: {e}")

        # Fix Bug #2: capture the active per-monitor DPI scale so the agent
        # prompt exposes the logical->physical coordinate mapping.
        dpi_scale = 1.0
        try:
            from grace.automation.dpi_helper import DPIHelper
            dpi_scale = DPIHelper.get_dpi_scale(
                active_window.hwnd if active_window else None
            )
        except Exception as e:
            logger.debug(f"DPI scale detection skipped: {e}")

        snapshot = ScreenSnapshot(
            active_window=active_window,
            ocr_lines=ocr_lines,
            width=width,
            height=height,
            ui_elements=ui_elements,
            png_bytes=img_bytes,
            dpi_scale=dpi_scale,
            graph=graph,
            image_width=image_width,
            image_height=image_height,
        )

        # Element IDs are positional - assigned by index after overlap dedup -
        # and the planner prompt refers to them by number. If the Rust UIA
        # walker orders the tree even slightly differently, every ID shifts and
        # the agent confidently clicks the wrong control. That failure is
        # silent, so the whole ordered node list is taped, not a summary.
        recorder = get_recorder()
        if recorder is not None:
            recorder.record_snapshot(_snapshot_tape(snapshot))
        return snapshot

    async def capture_snapshot_async(self) -> ScreenSnapshot:
        """Asynchronously capture desktop perception snapshot without blocking the event loop."""
        return await asyncio.to_thread(self.capture_snapshot)

    def _get_active_window_info(self) -> Optional[WindowInfo]:
        """Query foreground window details via win32gui."""
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return None
            title = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            return WindowInfo(
                hwnd=hwnd,
                title=title,
                class_name=class_name,
                rect=rect,
            )
        except Exception as e:
            logger.debug(f"Failed to query active window info: {e}")
            return None

    def _take_screenshot(self) -> tuple[int, int, Optional[bytes], int, int]:
        """Screenshot the desktop.

        Returns (screen_w, screen_h, png_bytes, image_w, image_h). The image is
        downscaled before encoding: a full 4K PNG base64'd into every turn is a
        large share of the prompt, and grounding models are trained on far
        smaller inputs. image_w/h are returned so predicted coordinates can be
        scaled back to screen space.
        """
        try:
            import pyautogui
            from grace.config import Config

            screenshot = pyautogui.screenshot()
            screen_w, screen_h = screenshot.width, screenshot.height

            max_width = Config().screenshot_max_width
            image = screenshot
            if max_width and screen_w > max_width:
                ratio = max_width / float(screen_w)
                image = screenshot.resize(
                    (max_width, max(1, int(screen_h * ratio)))
                )

            buf = io.BytesIO()
            image.save(buf, format="PNG", compress_level=1)
            return screen_w, screen_h, buf.getvalue(), image.width, image.height
        except Exception as e:
            logger.debug(f"pyautogui screenshot failed: {e}, using fallback dimensions")
            return 1920, 1080, None, 0, 0

    def _perform_ocr(self, img_bytes: bytes, width: int, height: int) -> List[OcrLine]:
        """Perform OCR on screenshot bytes."""
        if not img_bytes:
            return []

        # Attempt Windows Media OCR if available
        if self._winrt_ocr_available:
            try:
                import winrt.windows.media.ocr as ocr
                import winrt.windows.graphics.imaging as imaging
                import winrt.windows.storage.streams as streams

                engine = ocr.OcrEngine.try_create_from_user_profile_languages()
                if engine:
                    stream = streams.InMemoryRandomAccessStream()
                    writer = streams.DataWriter(stream)
                    writer.write_bytes(img_bytes)
                    _run_async(writer.store_async())
                    stream.seek(0)

                    decoder = _run_async(imaging.BitmapDecoder.create_async(stream))
                    bitmap = _run_async(decoder.get_software_bitmap_async())
                    result = _run_async(engine.recognize_async(bitmap))

                    lines = []
                    for line in result.lines:
                        text = line.text
                        # Real bounding box from the word rectangles. The Python
                        # WinRT projection exposes snake_case `bounding_rect`;
                        # the old `boundingRect` raised AttributeError on every
                        # line, so this silently fell through to Tesseract on
                        # every single turn after paying for the WinRT decode.
                        rect = (0, 0, 0, 0)
                        try:
                            words = list(line.words or [])
                            if words:
                                boxes = [_word_rect(w) for w in words]
                                boxes = [b for b in boxes if b is not None]
                                if boxes:
                                    min_x = min(b[0] for b in boxes)
                                    min_y = min(b[1] for b in boxes)
                                    max_r = max(b[0] + b[2] for b in boxes)
                                    max_b = max(b[1] + b[3] for b in boxes)
                                    rect = (min_x, min_y, max_r - min_x, max_b - min_y)
                        except Exception as e:
                            logger.debug(f"OCR word rect extraction failed: {e}")
                        lines.append(OcrLine(text=text, bounding_box=rect))
                    return lines
            except Exception as e:
                logger.debug(f"WinRT OCR execution failed: {e}, falling back to PIL/Tesseract")

        # Fallback to PyTesseract
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(io.BytesIO(img_bytes))
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            lines = []
            n_boxes = len(data["text"])
            current_line_words = []
            line_bbox = [width, height, 0, 0]

            for i in range(n_boxes):
                word = data["text"][i].strip()
                if not word:
                    continue
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                current_line_words.append(word)
                line_bbox[0] = min(line_bbox[0], x)
                line_bbox[1] = min(line_bbox[1], y)
                line_bbox[2] = max(line_bbox[2], x + w)
                line_bbox[3] = max(line_bbox[3], y + h)

                if i == n_boxes - 1 or data["line_num"][i] != data["line_num"][min(i + 1, n_boxes - 1)]:
                    if current_line_words:
                        text = " ".join(current_line_words)
                        bw = line_bbox[2] - line_bbox[0]
                        bh = line_bbox[3] - line_bbox[1]
                        lines.append(OcrLine(text=text, bounding_box=(line_bbox[0], line_bbox[1], bw, bh)))
                        current_line_words = []
                        line_bbox = [width, height, 0, 0]
            return lines
        except Exception:
            return []
