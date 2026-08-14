"""A mark must land on the control it names.

Three components share one coordinate contract and none of them states it:
``PerceptionEngine`` downscales the screenshot to 1280px wide and records both
sizes on the snapshot; ``ElementGraph`` holds rects in *screen* pixels; and the
overlay draws on the *image*. A badge drawn without applying the ratio between
them is displaced further the further right its control sits - and every
individual component is behaving correctly.

Nothing caught it, because the only existing test asserted that
``apply_overlay`` returned bytes, and the old implementation returned its input
unmodified on any exception. An overlay that drew nothing passed.

These tests look at pixels. The claim being made is geometric, so checking a
byte count or a return type would be checking something else.
"""

import io

import pytest
from PIL import Image

from grace.automation import som_overlay
from grace.perception.elements import ElementNode

pytestmark = pytest.mark.integration


SCREEN = (2560, 1440)
IMAGE = (1280, 720)  # what _take_screenshot produces from that screen


def element(element_id: int, rect: tuple) -> ElementNode:
    left, top, right, bottom = rect
    return ElementNode(
        id=element_id,
        role="button",
        name=f"control-{element_id}",
        rect=rect,
        center=((left + right) // 2, (top + bottom) // 2),
    )


def blank_image(size=IMAGE) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def painted_pixels(image: Image.Image, box: tuple) -> int:
    """How many pixels inside `box` are not the background."""
    left, top, right, bottom = box
    crop = image.crop((max(0, left), max(0, top), right, bottom)).convert("RGB")
    raw = crop.tobytes()
    return sum(
        1 for i in range(0, len(raw), 3) if raw[i:i + 3] != b"\xff\xff\xff"
    )


class TestScaling:
    def test_a_screen_rect_is_converted_to_image_space(self):
        # The defect, in one assertion. Unscaled, this rect would be drawn at
        # x=2000 on a 1280-wide image: off the right edge entirely.
        scaled = som_overlay.scale_rect((2000, 1000, 2200, 1100), SCREEN, IMAGE)
        assert scaled == (1000, 500, 1100, 550)

    def test_marks_are_positioned_in_image_space(self):
        marks = som_overlay.marks_for([element(1, (2000, 1000, 2200, 1100))], SCREEN, IMAGE)
        assert len(marks) == 1
        assert marks[0].rect == (1000, 500, 1100, 550)

    def test_a_mark_off_the_image_is_dropped_not_drawn(self):
        # Offscreen controls have rects outside the captured area. Drawing them
        # clamped to the edge would put a badge on an unrelated control.
        marks = som_overlay.marks_for([element(1, (4000, 3000, 4200, 3100))], SCREEN, IMAGE)
        assert marks == []

    def test_a_degenerate_rect_is_dropped(self):
        marks = som_overlay.marks_for([element(1, (100, 100, 100, 100))], SCREEN, IMAGE)
        assert marks == []


class TestRendering:
    def test_the_badge_lands_on_its_control(self):
        rect = (1000, 600, 1400, 700)  # screen space -> (500, 300, 700, 350)
        marks = som_overlay.marks_for([element(7, rect)], SCREEN, IMAGE)
        rendered = Image.open(io.BytesIO(som_overlay.render(blank_image(), marks)))

        expected = marks[0].rect
        # Generous vertical margin: the badge sits just above the box.
        near = (expected[0] - 5, expected[1] - 25, expected[2] + 5, expected[3] + 5)
        assert painted_pixels(rendered, near) > 0, "nothing was drawn near the control"

    def test_nothing_is_drawn_where_the_unscaled_rect_would_have_gone(self):
        # The regression itself: with the scaling dropped, this control's badge
        # would appear around x=1000 in image space rather than x=500.
        rect = (1000, 600, 1400, 700)
        marks = som_overlay.marks_for([element(7, rect)], SCREEN, IMAGE)
        rendered = Image.open(io.BytesIO(som_overlay.render(blank_image(), marks)))

        wrong = (rect[0] - 5, rect[1] - 25, min(IMAGE[0], rect[2] + 5), rect[3] + 5)
        assert painted_pixels(rendered, wrong) == 0, (
            "pixels were painted at the unscaled screen coordinates - the rect "
            "is being drawn in the wrong coordinate space"
        )

    def test_a_control_at_the_top_edge_keeps_its_badge(self):
        # The badge is drawn above the box; at y=0 that is off the image, and
        # clipping it would leave a marked control with no readable number.
        marks = som_overlay.marks_for([element(3, (200, 0, 600, 60))], SCREEN, IMAGE)
        rendered = Image.open(io.BytesIO(som_overlay.render(blank_image(), marks)))

        left, top, right, bottom = marks[0].rect
        assert painted_pixels(rendered, (left, top, right, bottom + 5)) > 0

    def test_rendering_without_marks_raises(self):
        # The behaviour the old vacuous test depended on. Returning the input
        # image here tells the caller the screen was marked when it was not,
        # and the planner is then asked to answer with numbers it cannot see.
        with pytest.raises(ValueError):
            som_overlay.render(blank_image(), [])

    def test_render_or_none_reports_failure_rather_than_faking_success(self):
        assert som_overlay.render_or_none(b"not a png", [som_overlay.Mark(1, "button", "x", (0, 0, 10, 10))]) is None


class TestLegend:
    def test_every_drawn_mark_appears_in_the_legend(self):
        elements = [element(i, (100 * i, 100, 100 * i + 80, 160)) for i in range(1, 6)]
        marks = som_overlay.marks_for(elements, SCREEN, IMAGE)
        text = som_overlay.legend(marks)

        for mark in marks:
            assert mark.label in text, f"{mark.label} was drawn but is not in the legend"

    def test_mark_numbers_are_element_ids_not_a_display_counter(self):
        # The other half of the original defect: the overlay tagged a different
        # numbering from the one the planner was shown, so an answer of "3"
        # meant one control to the model and another to the click path.
        elements = [element(11, (100, 100, 200, 200)), element(47, (300, 100, 400, 200))]
        marks = som_overlay.marks_for(elements, SCREEN, IMAGE)

        assert [m.id for m in marks] == [11, 47]
        assert "[47]" in som_overlay.legend(marks)
