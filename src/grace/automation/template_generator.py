"""Dynamic Template Generator for OculiX and OpenCV matching.

Generates PNG template files on the fly via:
1. High-contrast dynamic text rendering (Segoe UI / Arial)
2. Desktop screenshot region cropping from UIA bounding boxes
"""

import io
import logging
import os
import tempfile
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("grace.automation.template_generator")


class TemplateGenerator:
    """Generates PNG template image files for visual template matching."""

    _temp_dir = None

    @classmethod
    def get_temp_dir(cls) -> str:
        """Get or create directory for temporary template PNG files."""
        if cls._temp_dir is None or not os.path.exists(cls._temp_dir):
            cls._temp_dir = tempfile.mkdtemp(prefix="grace_templates_")
        return cls._temp_dir

    @classmethod
    def render_text_template(
        cls,
        text: str,
        font_size: int = 16,
        padding: int = 8,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        fg_color: Tuple[int, int, int] = (0, 0, 0),
        font_name: Optional[str] = None,
        bold: bool = False,
    ) -> Optional[str]:
        """Render text onto a high-contrast canvas image and save as a temporary PNG.

        Args:
            text: The text to render (e.g. "Play", "File", "Settings").
            font_size: Font size in points.
            padding: Margin around text in pixels.
            bg_color: RGB background tuple.
            fg_color: RGB text color tuple.
            font_name: Optional specific font file (e.g. "segoeuib.ttf"). If None,
                       the default Segoe UI / Arial / Calibri cascade is used.
            bold: If True, prefer bold font variants.

        Returns:
            Absolute path to generated PNG template file.
        """
        if not text or not text.strip():
            return None

        try:
            font = None
            if font_name:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                except IOError:
                    font = None

            if font is None:
                # Try Windows UI fonts in order (bold variants first when requested)
                font_candidates = []
                if bold:
                    font_candidates += ["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"]
                font_candidates += ["segoeui.ttf", "arial.ttf", "calibri.ttf", "tahoma.ttf"]
                for f_name in font_candidates:
                    try:
                        font = ImageFont.truetype(f_name, font_size)
                        break
                    except IOError:
                        continue

            if font is None:
                font = ImageFont.load_default()

            # Measure text size
            dummy_img = Image.new("RGB", (1, 1))
            draw = ImageDraw.Draw(dummy_img)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            width = text_w + (padding * 2)
            height = text_h + (padding * 2)

            img = Image.new("RGB", (width, height), color=bg_color)
            draw = ImageDraw.Draw(img)
            draw.text((padding - bbox[0], padding - bbox[1]), text, fill=fg_color, font=font)

            suffix = "_bold" if bold else ""
            temp_path = os.path.join(
                cls.get_temp_dir(), f"text_{hash(text) & 0xFFFFFFFF}{suffix}.png"
            )
            img.save(temp_path, format="PNG")
            logger.debug(f"Rendered text template '{text}' -> {temp_path} ({width}x{height})")
            return temp_path

        except Exception as e:
            logger.error(f"Failed to render text template '{text}': {e}")
            return None

    @classmethod
    def render_text_templates(cls, text: str, font_size: int = 16) -> List[str]:
        """Render a single text label in multiple font variants.

        Fix Bug #5: web apps render custom web fonts (YouTube Sans, Roboto) that
        pure GDI bitmap templates (Segoe UI) cannot match above the 0.7 threshold.
        Producing several template variants lets the OpenCV matcher keep the best
        score across the font space.

        Returns:
            List of absolute paths to generated PNG template files (may be empty
            on failure or blank input).
        """
        if not text or not text.strip():
            return []

        paths: List[str] = []
        # Variant 1: default (Segoe UI regular) — kept for backward compatibility
        p = cls.render_text_template(text, font_size=font_size)
        if p:
            paths.append(p)
        # Variant 2: bold
        p = cls.render_text_template(text, font_size=font_size, bold=True)
        if p:
            paths.append(p)
        # Variant 3: Arial (close to Roboto / system sans web fonts)
        p = cls.render_text_template(text, font_size=font_size, font_name="arial.ttf")
        if p and p not in paths:
            paths.append(p)
        return paths

    @classmethod
    def crop_uia_template(
        cls,
        screenshot_bytes: bytes,
        bounds: Tuple[int, int, int, int],  # left, top, right, bottom
        margin: int = 2,
    ) -> Optional[str]:
        """Crop a UIA element region from a screenshot and save as PNG.

        Args:
            screenshot_bytes: Raw screenshot PNG/BMP bytes.
            bounds: (left, top, right, bottom) element box.
            margin: Extra padding around bounds.

        Returns:
            Absolute path to cropped template PNG file.
        """
        if not screenshot_bytes or not bounds:
            return None

        try:
            left, top, right, bottom = bounds
            w = right - left
            h = bottom - top
            if w <= 0 or h <= 0:
                return None

            img = Image.open(io.BytesIO(screenshot_bytes))
            img_w, img_h = img.size

            crop_l = max(0, left - margin)
            crop_t = max(0, top - margin)
            crop_r = min(img_w, right + margin)
            crop_b = min(img_h, bottom + margin)

            cropped = img.crop((crop_l, crop_t, crop_r, crop_b))

            temp_path = os.path.join(
                cls.get_temp_dir(), f"crop_{left}_{top}_{w}_{h}.png"
            )
            cropped.save(temp_path, format="PNG")
            logger.debug(f"Cropped UIA template region [{bounds}] -> {temp_path}")
            return temp_path

        except Exception as e:
            logger.error(f"Failed to crop UIA template: {e}")
            return None
