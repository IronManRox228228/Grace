"""Pure Python OpenCV Vision Engine (Fallback when OculiX/JVM unavailable).

Provides multi-scale template matching (cv2.matchTemplate) and fallback OCR.
"""

import io
import logging
from typing import Optional, List, Tuple
import numpy as np

logger = logging.getLogger("grace.automation.vision_engine")

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

from grace.automation.oculix_bridge import VisualMatch


class VisionEngine:
    """Pure-Python OpenCV visual template matching and OCR fallback engine."""

    def __init__(self):
        if not HAS_OPENCV:
            logger.warning("OpenCV (cv2) is not installed. VisionEngine disabled.")

    def find_on_screen(
        self,
        screenshot_bytes: bytes,
        template_bytes_or_path,
        threshold: float = 0.7,
        scale_range: Tuple[float, float, int] = (0.8, 1.2, 5),
    ) -> Optional[VisualMatch]:
        """Perform multi-scale template matching using OpenCV.

        Args:
            screenshot_bytes: Raw PNG bytes of target screen image.
            template_bytes_or_path: Bytes or filepath of template PNG image.
            threshold: Min match confidence score (0.0 - 1.0).
            scale_range: (min_scale, max_scale, num_steps) scale space search.
        """
        if not HAS_OPENCV or not screenshot_bytes:
            return None

        try:
            # Decode screenshot
            screen_arr = cv2.imdecode(
                np.frombuffer(screenshot_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if screen_arr is None:
                return None

            # Decode template
            if isinstance(template_bytes_or_path, str):
                template_arr = cv2.imread(template_bytes_or_path, cv2.IMREAD_COLOR)
                label = template_bytes_or_path
            else:
                template_arr = cv2.imdecode(
                    np.frombuffer(template_bytes_or_path, np.uint8), cv2.IMREAD_COLOR
                )
                label = "template_bytes"

            if template_arr is None:
                return None

            screen_gray = cv2.cvtColor(screen_arr, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template_arr, cv2.COLOR_BGR2GRAY)
            t_h, t_w = template_gray.shape[:2]

            best_match = None
            max_val_global = -1.0

            # Scale space loop
            scales = np.linspace(scale_range[0], scale_range[1], scale_range[2])
            for scale in scales:
                sw = int(t_w * scale)
                sh = int(t_h * scale)
                if sw <= 0 or sh <= 0 or sw > screen_gray.shape[1] or sh > screen_gray.shape[0]:
                    continue

                scaled_temp = cv2.resize(template_gray, (sw, sh), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(screen_gray, scaled_temp, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > max_val_global:
                    max_val_global = max_val
                    cx = max_loc[0] + sw // 2
                    cy = max_loc[1] + sh // 2
                    best_match = VisualMatch(
                        x=cx, y=cy, width=sw, height=sh,
                        confidence=float(max_val), method="opencv",
                        label=f"{label} (scale={scale:.2f})"
                    )

            if best_match and best_match.confidence >= threshold:
                logger.info(
                    f"OpenCV matched '{label}' at ({best_match.x}, {best_match.y}) "
                    f"confidence={best_match.confidence:.2f}"
                )
                return best_match

        except Exception as e:
            logger.error(f"OpenCV find_on_screen failed: {e}")
        return None

    def find_text_on_screen(
        self,
        screenshot_bytes: bytes,
        target_text: str,
    ) -> Optional[VisualMatch]:
        """Find text on screen using PyTesseract OCR with real bounding box output."""
        if not screenshot_bytes or not target_text:
            return None

        try:
            import pytesseract

            img = Image.open(io.BytesIO(screenshot_bytes))
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            n_boxes = len(data["text"])
            target_clean = target_text.lower().strip()

            for i in range(n_boxes):
                word = data["text"][i].strip()
                if not word:
                    continue

                if target_clean in word.lower() or word.lower() in target_clean:
                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]
                    cx = x + w // 2
                    cy = y + h // 2
                    conf = float(data["conf"][i]) / 100.0 if data["conf"][i] != "-1" else 0.8
                    logger.info(f"OpenCV/PyTesseract OCR found text '{word}' at ({cx}, {cy})")
                    return VisualMatch(
                        x=cx, y=cy, width=w, height=h,
                        confidence=conf, method="ocr", label=word
                    )
        except Exception as e:
            logger.debug(f"PyTesseract OCR search failed for '{target_text}': {e}")
        return None
