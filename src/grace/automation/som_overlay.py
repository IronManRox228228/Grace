"""Set-of-Mark rendering: numbered tags drawn over targetable regions.

A vision model asked to click something in a screenshot has to name a position,
and models are poor at that. Set-of-Mark prompting removes the problem instead
of improving the estimate: every candidate is drawn with a numbered badge, and
the model answers with a number. Grace already resolves ids to screen
coordinates - that is what the element graph does - so a mark id is simply an
element id the model can *see*.

This module renders. It does not decide what deserves a mark; that is the
element graph's job, and it matters that the two agree, because a badge whose
number does not resolve is worse than no badge at all.

Two defects are fixed here relative to the previous version, both of which
produced a plausible-looking image that was wrong:

* it tagged ``UIElement.index`` from the legacy flat inspector, a different
  numbering from the ``ElementNode.id`` the planner is shown, so the model would
  have answered with a number meaning something else;
* it drew element rects, which are in screen pixels, onto the screenshot handed
  to models, which has been downscaled to 1280px wide - so every badge landed
  displaced by the scale factor, further from its control the further right it
  sat.

Failures raise. The previous version returned the unmodified image on any
exception, which is indistinguishable from success to the caller and made its
only test vacuous: an overlay that drew nothing still returned bytes.
"""

import io
import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("grace.automation.som_overlay")

#: Most marks to draw. Beyond this the image stops being readable - badges
#: overlap each other and the model does worse than with no marks at all.
MAX_MARKS = 35

_BOX_OUTLINE = (255, 30, 30, 220)
_TAG_BG = (255, 30, 30, 240)
_TEXT = (255, 255, 255, 255)
_FONT_SIZE = 15
_FONT_CANDIDATES = ("segoeui.ttf", "arial.ttf", "calibri.ttf", "tahoma.ttf")


@dataclass(frozen=True)
class Mark:
    """One numbered target, positioned in *image* space.

    ``id`` is the element graph id. It is not a display counter: the model
    answers with this number and the click path resolves it through
    ``ElementGraph.by_id``, so renumbering marks for tidiness would break the
    only thing they are for.
    """

    id: int
    role: str
    name: str
    rect: tuple[int, int, int, int]  # left, top, right, bottom, in image px

    @property
    def label(self) -> str:
        return f"[{self.id}]"


def scale_rect(
    rect: tuple[int, int, int, int],
    screen_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Screen-pixel rect -> image-pixel rect.

    ``ScreenSnapshot`` carries both sizes precisely so this conversion has a
    single definition. Everything that draws on the screenshot must go through
    it; the old overlay skipped it entirely.
    """
    screen_w, screen_h = screen_size
    image_w, image_h = image_size
    if not screen_w or not screen_h or not image_w or not image_h:
        return rect

    sx = image_w / float(screen_w)
    sy = image_h / float(screen_h)
    left, top, right, bottom = rect
    return (
        int(round(left * sx)),
        int(round(top * sy)),
        int(round(right * sx)),
        int(round(bottom * sy)),
    )


def marks_for(
    elements: Iterable,
    screen_size: tuple[int, int],
    image_size: tuple[int, int],
    limit: int = MAX_MARKS,
) -> list[Mark]:
    """Build the mark set for the elements a snapshot offers as targets.

    Degenerate and off-image rects are dropped rather than drawn: a zero-width
    badge is a number the model can read and nothing it can aim at.
    """
    image_w, image_h = image_size
    marks: list[Mark] = []

    for element in elements:
        if len(marks) >= limit:
            break

        left, top, right, bottom = scale_rect(
            tuple(element.rect), screen_size, image_size
        )
        if right <= left or bottom <= top:
            continue
        if right < 0 or bottom < 0 or left > image_w or top > image_h:
            continue

        marks.append(
            Mark(
                id=element.id,
                role=getattr(element, "role", ""),
                name=getattr(element, "name", "") or getattr(element, "placeholder", ""),
                rect=(
                    max(0, left),
                    max(0, top),
                    min(image_w, right),
                    min(image_h, bottom),
                ),
            )
        )

    return marks


def legend(marks: Sequence[Mark]) -> str:
    """The text half of the observation: what each drawn number is.

    Sent alongside the image rather than instead of it. The badge gives the
    model the position; this gives it the name, which is what lets it choose
    between two visually similar rows.
    """
    if not marks:
        return "### Marks: (none - nothing on screen could be marked)"

    lines = [
        "### Marks on the screenshot",
        "Each red badge is a target. Answer with its number as `element_id`.",
    ]
    for mark in marks:
        described = mark.name.strip() or "(unlabelled)"
        lines.append(f"{mark.label} {mark.role or 'region'}: {described}")
    return "\n".join(lines)


def _load_font() -> ImageFont.ImageFont:
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, _FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def render(image_bytes: bytes, marks: Sequence[Mark]) -> bytes:
    """Draw the marks onto the screenshot and return PNG bytes.

    Raises on failure. A caller that cannot mark an image needs to know, because
    the alternative is a planner told to answer with numbers it cannot see.
    """
    if not image_bytes:
        raise ValueError("no image to mark")
    if not marks:
        raise ValueError("no marks to draw")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font()

    for mark in marks:
        left, top, right, bottom = mark.rect
        draw.rectangle([left, top, right, bottom], outline=_BOX_OUTLINE, width=2)

        # Badge above the box where there is room, inside it at the top of the
        # screen where there is not - so a mark on a control at y=0 is still
        # legible instead of being clipped off the image.
        tag = f" {mark.label} "
        tag_y = top - _FONT_SIZE - 3
        if tag_y < 0:
            tag_y = top + 1
        box = draw.textbbox((left, tag_y), tag, font=font)
        draw.rectangle(box, fill=_TAG_BG)
        draw.text((left, tag_y), tag, fill=_TEXT, font=font)

    combined = Image.alpha_composite(image, overlay).convert("RGB")
    buf = io.BytesIO()
    combined.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def render_or_none(image_bytes: bytes, marks: Sequence[Mark]) -> Optional[bytes]:
    """``render``, but returning None instead of raising.

    For the loop, which has a real fallback - plan from the element list - and
    should take it rather than abort the goal.
    """
    try:
        return render(image_bytes, marks)
    except Exception as exc:
        logger.warning(f"SoM overlay failed, falling back to element-only prompt: {exc}")
        return None
