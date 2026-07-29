"""Desktop Perception Engine for Grace Agentic Loop.

Provides hardware-accelerated OCR (Windows Media OCR / PyTesseract)
and Win32 UI control hierarchy context to feed Gemma's reasoning loop.
"""

import asyncio
import io
import logging
from dataclasses import dataclass
from typing import Any, Optional, List

logger = logging.getLogger("grace.agent.perception")


def _run_async(coro):
    """Safely run async coroutine in sync or async contexts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Running inside existing event loop
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)


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

        if self.ui_elements:
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

    def __init__(self):
        self._winrt_ocr_available = False
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
        active_window = self._get_active_window_info()
        width, height, img_bytes = self._take_screenshot()

        ocr_lines = []
        if img_bytes:
            ocr_lines = self._run_ocr(img_bytes, width, height)

        ui_elements = []
        try:
            ui_elements = self.get_shared_inspector().inspect_active_window()
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

    def _take_screenshot(self) -> tuple[int, int, Optional[bytes]]:
        """Take screenshot and return (width, height, PNG bytes)."""
        try:
            import pyautogui
            screenshot = pyautogui.screenshot()
            buf = io.BytesIO()
            screenshot.save(buf, format="PNG", compress_level=1)
            return screenshot.width, screenshot.height, buf.getvalue()
        except Exception as e:
            logger.debug(f"pyautogui screenshot failed: {e}, using fallback dimensions")
            return 1920, 1080, None

    def _run_ocr(self, img_bytes: bytes, width: int, height: int) -> List[OcrLine]:
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
                        # FIX BUG #1: Calculate real bounding box from word rectangles
                        if line.words and len(line.words) > 0:
                            min_x = min(int(w.boundingRect.x) for w in line.words)
                            min_y = min(int(w.boundingRect.y) for w in line.words)
                            max_r = max(int(w.boundingRect.x + w.boundingRect.width) for w in line.words)
                            max_b = max(int(w.boundingRect.y + w.boundingRect.height) for w in line.words)
                            rect = (min_x, min_y, max_r - min_x, max_b - min_y)
                        else:
                            rect = (0, 0, 0, 0)
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
