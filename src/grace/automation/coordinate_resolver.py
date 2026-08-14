"""Unified Coordinate Resolution Pipeline for Project Grace.

The cascade, as actually implemented by ``resolve()``:

1. **UIA tree** - by ``element_index``, ``target_name`` or ``relative_to``.
2. **UIA point-snap** - given a coordinate, snap it onto a small control within
   30px. Skipped for anything larger than 200px, since snapping to the centre
   of a full-width panel is worse than the original coordinate.
3. **OculiX template match**, then **OculiX OCR** - both only when a
   ``target_name`` is given *and* the JVM bridge is available. It is off by
   default (``USE_OCULIX=false``), so in the shipped configuration neither runs.
4. **Explicit coordinates**, clamped to the window.

This docstring used to describe a five-pass cascade including "Pure OpenCV
template matching" and "Multi-Engine OCR search (WinRT / PyTesseract)". Neither
exists: ``vision_engine.py`` implements both, but ``_get_vision_engine`` is
never called from anywhere, so ``self._vision`` stays None for the life of the
process. With OculiX off, the real cascade is UIA -> point-snap -> raw
coordinates. Documenting fallbacks that do not run made every failure here look
like bad luck rather than a missing stage.

``ResolvedTarget.method`` still advertises ``"opencv"`` and ``"ocr"`` for the
same reason; ``resolve()`` cannot emit either.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Any

from grace.automation.dpi_helper import DPIHelper
from grace.automation.oculix_bridge import OculixBridge, VisualMatch
from grace.automation.template_generator import TemplateGenerator
from grace.automation.ui_inspector import UIInspector, UIElement

logger = logging.getLogger("grace.automation.coordinate_resolver")


@dataclass
class ResolvedTarget:
    """Result of coordinate resolution."""
    x: int
    y: int
    confidence: float
    method: str       # "uia" | "oculix" | "opencv" | "ocr" | "explicit"
    element: Optional[UIElement] = None
    match: Optional[VisualMatch] = None


class CoordinateResolver:
    """Unified target coordinate resolver with 5-pass cascade."""

    def __init__(self, inspector: Optional[UIInspector] = None):
        self._inspector = inspector or UIInspector()
        self._vision = None

    def _get_vision_engine(self):
        if self._vision is None:
            from grace.automation.vision_engine import VisionEngine
            self._vision = VisionEngine()
        return self._vision

    @staticmethod
    def _clamp_center_to_window(
        center: Tuple[int, int],
        window_bounds: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[int, int]:
        """Clamp a resolved center point inside the target window bounds.

        Fix Bug #5: UIA/OCR centers can drift outside the window under DPI
        scaling; clamp so SendInput never clicks the desktop chrome.
        """
        if not window_bounds or len(window_bounds) < 4:
            return center
        cx, cy = center
        left, top, right, bottom = window_bounds
        cx = max(left, min(right, cx))
        cy = max(top, min(bottom, cy))
        return cx, cy

    def resolve(
        self,
        element_index: Optional[int] = None,
        target_name: Optional[str] = None,
        relative_to: Optional[str] = None,
        direction: str = "left",
        x: Optional[int] = None,
        y: Optional[int] = None,
        window_bounds: Optional[Tuple[int, int, int, int]] = None,
        window_title: Optional[str] = None,
        screenshot_bytes: Optional[bytes] = None,
    ) -> Optional[ResolvedTarget]:
        """Resolve click target coordinates using Win32 UIA bounds snapping + OculiX fallback."""

        DPIHelper.ensure_dpi_aware()

        # PASS 1a: Win32 UIA Tree Search (by index, target_name, or relative anchor)
        if element_index is not None or target_name or relative_to:
            uia_target = self._resolve_uia(element_index, target_name, relative_to, direction)
            if uia_target:
                uia_target.x, uia_target.y = self._clamp_center_to_window(
                    (uia_target.x, uia_target.y), window_bounds
                )
                logger.info(f"CoordinateResolver: Pass 1a (Win32 UIA Tree) hit at ({uia_target.x}, {uia_target.y})")
                return uia_target

        # PASS 1b: Win32 UIA Point-to-Element Bounds Center Snapping for (x, y)
        if x is not None and y is not None:
            norm_x, norm_y = self.normalize_coords(x, y, window_bounds)
            elem = self._inspector.find_element_at_point(norm_x, norm_y, tolerance_px=30)
            if elem:
                elem_w = abs(elem.bounds[2] - elem.bounds[0])
                elem_h = abs(elem.bounds[3] - elem.bounds[1])
                # Only snap to small, distinct controls (buttons, inputs, icons) - not large container panels
                MAX_SNAP_SIZE = 200
                if elem_w <= MAX_SNAP_SIZE and elem_h <= MAX_SNAP_SIZE:
                    cx, cy = elem.center
                    cx, cy = self._clamp_center_to_window((cx, cy), window_bounds)
                    logger.info(f"CoordinateResolver: Pass 1b (Win32 UIA Point Snap) snapped ({x}, {y}) -> '{elem.name}' center ({cx}, {cy})")
                    return ResolvedTarget(
                        x=cx, y=cy, confidence=1.0, method="uia_snap", element=elem
                    )
                else:
                    logger.info(f"CoordinateResolver: Pass 1b skipped snap for large container element '{elem.name}' ({elem_w}x{elem_h})")

        # PASS 2: OculiX Java Bridge Pattern & Text Matching Fallback
        if target_name and target_name.strip() and OculixBridge.is_available():
            template_paths = TemplateGenerator.render_text_templates(target_name.strip())
            oculix_target = self._resolve_oculix(template_paths, window_bounds)
            if oculix_target:
                logger.info(f"CoordinateResolver: Pass 2a (OculiX Pattern) hit at ({oculix_target.x}, {oculix_target.y}) conf={oculix_target.confidence:.2f}")
                return oculix_target

            oculix_ocr = self._resolve_oculix_text(target_name.strip(), window_bounds)
            if oculix_ocr:
                logger.info(f"CoordinateResolver: Pass 2b (OculiX OCR) hit '{target_name}' at ({oculix_ocr.x}, {oculix_ocr.y})")
                return oculix_ocr

        # PASS 3: Direct Coords (Scaled 1000x1000 -> Screen Pixels, clamped to window)
        if x is not None and y is not None:
            final_x, final_y = self.normalize_coords(x, y, window_bounds)
            final_x, final_y = self._clamp_center_to_window((final_x, final_y), window_bounds)
            logger.info(f"CoordinateResolver: Pass 3 (Direct Coords) resolved ({x}, {y}) -> ({final_x}, {final_y})")
            return ResolvedTarget(
                x=final_x, y=final_y, confidence=1.0, method="explicit"
            )

        logger.warning(f"CoordinateResolver: Resolution failed for target_name='{target_name}', element_index={element_index}, x={x}, y={y}")
        return None

    def normalize_coords(
        self,
        x: int,
        y: int,
        window_bounds: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[int, int]:
        """Convert relative window coordinates to screen absolute coordinates correctly."""
        if not window_bounds or len(window_bounds) < 4:
            return x, y

        left, top, right, bottom = window_bounds
        w = right - left
        h = bottom - top

        if 0 <= x <= w and 0 <= y <= h:
            return left + x, top + y

        return x, y

    def _resolve_uia(self, element_index, target_name, relative_to, direction):
        try:
            if relative_to:
                elem = self._inspector.find_relative_element(
                    target_name=target_name, relative_to=relative_to, direction=direction
                )
            else:
                elem = self._inspector.find_element(
                    target_name=target_name, element_index=element_index
                )
            if elem:
                return ResolvedTarget(
                    x=elem.center[0], y=elem.center[1],
                    confidence=1.0, method="uia", element=elem
                )
        except Exception as e:
            logger.debug(f"UIA resolution exception: {e}")
        return None

    def _resolve_oculix(self, template_paths, window_bounds: Optional[Tuple[int, int, int, int]]):
        """Try each template variant via OculiX; return the first confident match."""
        for template_path in template_paths:
            match = OculixBridge.find(template_path, region=window_bounds, similarity=0.7)
            if match:
                return ResolvedTarget(
                    x=match.x, y=match.y, confidence=match.confidence,
                    method="oculix", match=match
                )
        return None

    def _resolve_oculix_text(self, text: str, window_bounds: Optional[Tuple[int, int, int, int]]):
        match = OculixBridge.find_text(text, region=window_bounds)
        if match:
            return ResolvedTarget(
                x=match.x, y=match.y, confidence=match.confidence,
                method="oculix_ocr", match=match
            )
        return None
