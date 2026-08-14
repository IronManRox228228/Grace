"""OCR has to actually produce marks, or blind mode marks only window furniture.

The Set-of-Mark path is only worth having if there is something to mark. An app
that does not report its controls to UI Automation reports its *window frame* -
a tab, three scrollbar arrows, a system menu - and marking those is marking
nothing the user asked for. The content is legible on screen, so OCR is where
the real targets come from.

That made a silent, long-standing failure load-bearing. Three faults stacked:

1. the WinRT OCR packages were never declared in requirements.txt, so the
   preferred backend was not installed and logged "not installed, fallback OCR
   mode active" at INFO on every start;
2. ``_run_async`` called ``asyncio.run`` on an ``IAsyncOperation``. WinRT
   projections return an object with ``__await__`` but ``asyncio.run`` requires
   a coroutine specifically, so every WinRT OCR call raised
   "a coroutine was expected, got DataWriterStoreOperation" on its first await;
3. that exception was swallowed by the broad ``except`` in ``_perform_ocr``,
   which fell through to Tesseract - whose binary no setup step installs.

Net effect: OCR returned zero lines on every capture, and nothing said so.
``should_ocr`` was computed, paid for and discarded.
"""

import asyncio

import pytest

from grace.agent.perception import (
    MIN_OCR_TEXT_LEN,
    OcrLine,
    _ocr_elements,
    _run_async,
)
from grace.perception.elements import ROLE_TEXT, SOURCE_OCR

pytestmark = pytest.mark.integration


class FakeAsyncOperation:
    """Awaitable but not a coroutine - the shape WinRT actually returns."""

    def __init__(self, value):
        self._value = value

    def __await__(self):
        async def _resolve():
            return self._value

        return _resolve().__await__()


class TestRunAsync:
    def test_accepts_a_non_coroutine_awaitable(self):
        # asyncio.run(op) raises ValueError here. This is the whole bug.
        assert _run_async(FakeAsyncOperation("ocr result")) == "ocr result"

    def test_accepts_a_plain_coroutine_too(self):
        async def work():
            return 42

        assert _run_async(work()) == 42

    def test_works_from_inside_a_running_loop(self):
        # The real call site is inside the agent loop. The nested case goes to
        # a short-lived thread rather than re-entering the running loop.
        async def outer():
            return _run_async(FakeAsyncOperation("nested"))

        assert asyncio.run(outer()) == "nested"

    def test_an_error_reaches_the_caller(self):
        class Boom:
            def __await__(self):
                async def _fail():
                    raise RuntimeError("winrt said no")

                return _fail().__await__()

        with pytest.raises(RuntimeError, match="winrt said no"):
            _run_async(Boom())


class TestOcrPromotion:
    """Recognised lines become ordinary elements, in screen coordinates."""

    SCREEN = (2560, 1440)
    IMAGE = (1280, 720)

    def line(self, text, box):
        return OcrLine(text=text, bounding_box=box)

    def test_boxes_are_converted_from_image_space_to_screen_space(self):
        # OCR runs on the downscaled screenshot; every other element consumer
        # assumes screen pixels. Skipping this puts the click at half the
        # distance across the screen.
        [element] = _ocr_elements(
            [self.line("Chemistry", (100, 200, 300, 40))],
            self.SCREEN, self.IMAGE, start_id=1,
        )
        assert element.rect == (200, 400, 800, 480)
        assert element.center == (500, 440)

    def test_promoted_lines_are_marked_as_ocr_derived(self):
        [element] = _ocr_elements(
            [self.line("Chemistry", (10, 20, 100, 30))], self.SCREEN, self.IMAGE, start_id=7,
        )
        assert element.id == 7
        assert element.source == SOURCE_OCR
        assert element.role == ROLE_TEXT

    def test_promoted_lines_are_not_counted_as_actionable(self):
        # They are targets, not evidence the app reports its controls. If they
        # counted, a blind window with lots of text would look readable and the
        # agent would be back to inventing coordinates.
        [element] = _ocr_elements(
            [self.line("Chemistry", (10, 20, 100, 30))], self.SCREEN, self.IMAGE, start_id=1,
        )
        assert element.is_actionable is False

    def test_ids_continue_from_the_existing_elements(self):
        elements = _ocr_elements(
            [self.line(f"line {i}", (0, i * 10, 100, 10)) for i in range(3)],
            self.SCREEN, self.IMAGE, start_id=6,
        )
        assert [e.id for e in elements] == [6, 7, 8]

    def test_noise_is_dropped(self):
        promoted = _ocr_elements(
            [
                self.line("", (0, 0, 10, 10)),
                self.line("x" * (MIN_OCR_TEXT_LEN - 1), (0, 20, 10, 10)),
                self.line("Chemistry", (0, 40, 100, 20)),
                self.line("zero size", (0, 60, 0, 0)),
            ],
            self.SCREEN, self.IMAGE, start_id=1,
        )
        assert [e.name for e in promoted] == ["Chemistry"]

    def test_promotion_is_bounded(self):
        many = [self.line(f"line {i}", (0, i * 5, 100, 4)) for i in range(200)]
        promoted = _ocr_elements(many, self.SCREEN, self.IMAGE, start_id=1)
        assert len(promoted) <= 30

    def test_nothing_is_promoted_without_image_dimensions(self):
        # A failed screenshot reports 0x0. Dividing by that is how a coordinate
        # becomes a ZeroDivisionError deep inside a click.
        assert _ocr_elements(
            [self.line("Chemistry", (10, 20, 100, 30))], self.SCREEN, (0, 0), start_id=1,
        ) == []
