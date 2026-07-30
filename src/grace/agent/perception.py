"""Desktop Perception Engine for Grace Agentic Loop.

Provides hardware-accelerated OCR (Windows Media OCR / PyTesseract)
and Win32 UI control hierarchy context to feed Gemma's reasoning loop.
"""

import asyncio
import io
import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional, List

logger = logging.getLogger("grace.agent.perception")


def _run_async(coro):
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
        return asyncio.run(coro)

    result: dict = {}

    def _worker():
        try:
            result["value"] = asyncio.run(coro)
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

    def to_markdown(self) -> str:
        """Format the screen state into clean markdown for Gemma."""
        lines = []

        window_title_hint = ""
        if self.active_window:
            w = self.active_window
            window_title_hint = w.title
            lines.append(f"### Focused Window: '{w.title}' (Class: {w.class_name})")
            lines.append(f"Bounds: [left: {w.rect[0]}, top: {w.rect[1]}, right: {w.rect[2]}, bottom: {w.rect[3]}]")
        else:
            lines.append("### Focused Window: None / Unknown")

        lines.append(f"Screen Dimensions: {self.width}x{self.height}")
        # Fix Bug #2: surface DPI scale so coordinates predict physical pixels.
        lines.append(f"DPI Scale: {self.dpi_scale:.2f} (logical -> physical)")

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
        ocr_lines = []
        should_ocr = self._run_ocr and img_bytes and (graph is None or len(graph) < 5)
        if should_ocr:
            with stage("ocr") as ocr_stage:
                ocr_lines = self._perform_ocr(img_bytes, width, height)
                ocr_stage.detail(f"{len(ocr_lines)} lines")

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

        return ScreenSnapshot(
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
