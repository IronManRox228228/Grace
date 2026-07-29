"""Unit tests for DPIHelper, SoMOverlayGenerator, and Spatial Relative Target Resolution."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.automation.dpi_helper import DPIHelper
from grace.automation.som_overlay import SoMOverlayGenerator
from grace.automation.ui_inspector import UIInspector, UIElement


class TestDPISOM:
    """Test suite for DPIHelper and SoMOverlayGenerator."""

    def test_dpi_helper_resolution(self):
        DPIHelper.ensure_dpi_aware()
        w, h = DPIHelper.get_screen_resolution()
        assert w > 0 and h > 0

    def test_dpi_scale_coords(self):
        sx, sy = DPIHelper.scale_coords(500, 400, 1000, 800, 2000, 1600)
        assert sx == 1000 and sy == 800

    def test_find_relative_element_left(self):
        inspector = UIInspector()
        song_elem = UIElement(index=1, name="Bohemian Rhapsody", control_type="Text", bounds=(400, 300, 600, 320), center=(500, 310))
        play_elem = UIElement(index=2, name="Play", control_type="Button", bounds=(350, 300, 380, 320), center=(365, 310))
        other_elem = UIElement(index=3, name="Play", control_type="Button", bounds=(100, 800, 130, 820), center=(115, 810))

        inspector._last_elements = [song_elem, play_elem, other_elem]

        found = inspector.find_relative_element(target_name="Play", relative_to="Bohemian Rhapsody", direction="left")
        assert found is not None
        assert found.center == (365, 310)

    def test_som_overlay_generation(self):
        from PIL import Image
        import io

        img = Image.new("RGB", (400, 400), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        elem = UIElement(index=1, name="Play", control_type="Button", bounds=(50, 50, 100, 100), center=(75, 75))
        overlay_bytes = SoMOverlayGenerator.apply_overlay(raw_bytes, [elem])
        assert isinstance(overlay_bytes, bytes)
        assert len(overlay_bytes) > 0

    def test_dpi_scale_defaults_and_markdown(self):
        """Bug #2 fix: ScreenSnapshot exposes dpi_scale in its markdown."""
        from grace.agent.perception import ScreenSnapshot, WindowInfo

        snap = ScreenSnapshot(
            active_window=WindowInfo(hwnd=1, title="Edge", class_name="Chrome_WidgetWin_1", rect=(0, 0, 1920, 1080)),
            ocr_lines=[],
            width=1920,
            height=1080,
            dpi_scale=1.5,
        )
        assert snap.dpi_scale == 1.5
        md = snap.to_markdown()
        assert "DPI Scale: 1.50 (logical -> physical)" in md

    def test_get_dpi_scale_returns_positive(self):
        scale = DPIHelper.get_dpi_scale()
        assert scale > 0
