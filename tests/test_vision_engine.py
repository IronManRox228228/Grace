"""Unit tests for pure Python VisionEngine template matching."""

import io
import unittest
from PIL import Image, ImageDraw, ImageFont
from grace.automation.template_generator import TemplateGenerator
from grace.automation.vision_engine import VisionEngine, HAS_OPENCV


class TestVisionEngine(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 400x300 canvas image
        img = Image.new("RGB", (400, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Render template file
        self.template_path = TemplateGenerator.render_text_template("Play")
        temp_img = Image.open(self.template_path)

        # Paste template image onto canvas at (150, 120)
        img.paste(temp_img, (150, 120))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.screenshot_bytes = buf.getvalue()

    @unittest.skipUnless(HAS_OPENCV, "OpenCV (cv2) not installed")
    def test_find_on_screen(self):
        engine = VisionEngine()
        match = engine.find_on_screen(self.screenshot_bytes, self.template_path, threshold=0.7)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(match.confidence, 0.7)
        # Target pasted at (150, 120) -> center should be at 150 + w/2, 120 + h/2
        self.assertGreater(match.x, 150)
        self.assertGreater(match.y, 120)


if __name__ == "__main__":
    unittest.main()
