"""Set-of-Marks (SoM) Visual Overlay Generator for Grace.

Draws numbered, semi-transparent bounding box tags ([1], [2], [3]) directly over
interactive UI controls on screenshot images before sending to vision LLMs.
This allows multimodal vision models (Gemini / Gemma) to lock onto exact target tags
without relying on raw pixel coordinate guessing.
"""

import io
import logging
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

from grace.automation.ui_inspector import UIElement

logger = logging.getLogger("grace.automation.som_overlay")


class SoMOverlayGenerator:
    """Generates Set-of-Marks visual overlay images for multimodal LLM vision prompts."""

    @classmethod
    def apply_overlay(cls, image_bytes: bytes, elements: List[UIElement]) -> bytes:
        """Draw bright numbered tags over UI controls and return tagged PNG bytes."""
        if not image_bytes or not elements:
            return image_bytes

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            # High-visibility colors
            box_outline = (255, 30, 30, 220)  # Bright Red
            tag_bg = (255, 30, 30, 240)      # Solid Red badge
            text_color = (255, 255, 255, 255) # White text

            font = None
            for font_name in ["arial.ttf", "segoeui.ttf", "calibri.ttf"]:
                try:
                    font = ImageFont.truetype(font_name, 16)
                    break
                except IOError:
                    continue
            if font is None:
                font = ImageFont.load_default()

            for elem in elements[:35]:
                cl, ct, cr, cb = elem.bounds
                if cr <= cl or cb <= ct:
                    continue

                # Draw bounding box rectangle
                draw.rectangle([cl, ct, cr, cb], outline=box_outline, width=2)

                # Draw numbered index tag badge at top-left of control
                tag_str = f" [{elem.index}] "
                bbox = draw.textbbox((cl, max(0, ct - 16)), tag_str, font=font)
                draw.rectangle(bbox, fill=tag_bg)
                draw.text((cl, max(0, ct - 16)), tag_str, fill=text_color, font=font)

            combined = Image.alpha_composite(image, overlay).convert("RGB")
            buf = io.BytesIO()
            combined.save(buf, format="PNG", compress_level=1)
            return buf.getvalue()
        except Exception as e:
            logger.debug(f"SoM overlay generation failed: {e}")
            return image_bytes
