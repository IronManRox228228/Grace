"""The observation and the legal targets must describe the same screen.

The agent's worst failure was not a broken component. Every part worked: the UIA
walker correctly reported that WhatsApp exposes four window-frame controls,
``to_markdown`` correctly rendered those four, and the planner correctly
produced a step. The defect was that the four were presented as *the screen*,
so the model - having no id for the chat row it wanted - invented a coordinate.
130 steps and twelve minutes later it was still inventing them.

So the contract is between three modules that each look right alone:

``agent/perception.py``   decides whether this window can be read at all
``agent/planner.py``      is shown one observation and told which targets exist
``automation/``           has to be able to resolve whatever comes back

These tests pin the pairing rather than any component: a readable window gets an
element list and id targets, an unreadable one gets a marked screenshot and mark
targets, and neither is ever offered a raw coordinate.
"""

import pytest

from grace.agent.perception import (
    OBSERVABILITY_BLIND,
    OBSERVABILITY_RICH,
    ScreenSnapshot,
    WindowInfo,
    observe_for_planner,
)
from grace.intent.tools import ALL_TOOLS
from grace.perception.element_graph import ElementGraph, WindowRef
from grace.perception.elements import ROLE_TEXT, SOURCE_OCR, ElementNode

pytestmark = pytest.mark.integration


SCREEN = (2560, 1440)
IMAGE = (1280, 720)


def control(element_id: int, name: str = "", role: str = "button", **kw) -> ElementNode:
    top = 100 + element_id * 40
    return ElementNode(
        id=element_id,
        role=role,
        name=name or f"control-{element_id}",
        rect=(100, top, 300, top + 30),
        center=(200, top + 15),
        **kw,
    )


def snapshot(elements, *, png=True) -> ScreenSnapshot:
    import io

    from PIL import Image

    image_bytes = None
    if png:
        buf = io.BytesIO()
        Image.new("RGB", IMAGE, (255, 255, 255)).save(buf, format="PNG")
        image_bytes = buf.getvalue()

    return ScreenSnapshot(
        active_window=WindowInfo(hwnd=1, title="WhatsApp", class_name="Chrome_WidgetWin_1", rect=(0, 0, 2560, 1440)),
        ocr_lines=[],
        width=SCREEN[0],
        height=SCREEN[1],
        graph=ElementGraph(elements=list(elements), window=WindowRef(hwnd=1, title="WhatsApp")),
        png_bytes=image_bytes,
        image_width=IMAGE[0] if png else 0,
        image_height=IMAGE[1] if png else 0,
    )


class TestObservabilityClassification:
    def test_a_window_reporting_only_its_frame_is_blind(self):
        # WhatsApp, as recorded: four title-bar controls and nothing else.
        frame = [control(i, n) for i, n in enumerate(["Minimise", "Maximise", "Close", "Menu"], 1)]
        assert snapshot(frame).observability == OBSERVABILITY_BLIND

    def test_a_terminal_reporting_only_furniture_is_blind(self):
        # Captured from a real Windows Terminal window, whose contents UIA does
        # not expose at all. It reports exactly these five, and an earlier
        # threshold of five classified it as readable - which is how this test
        # came to exist. The names are what UIA actually returned.
        furniture = [
            control(1, "", role="tab"),
            control(2, "Vertical Small Decrease"),
            control(3, "Vertical Large Increase"),
            control(4, "Vertical Small Increase"),
            control(5, "System", role="menuitem"),
        ]
        assert snapshot(furniture).observability == OBSERVABILITY_BLIND

    def test_a_window_reporting_its_contents_is_rich(self):
        assert snapshot([control(i) for i in range(1, 21)]).observability == OBSERVABILITY_RICH

    def test_disabled_and_offscreen_controls_do_not_count(self):
        # The count has to mean "things a step could aim at". Twenty entries
        # that are all unreachable is the same blindness as none.
        unusable = [control(i, enabled=False) for i in range(1, 11)]
        unusable += [control(i, offscreen=True) for i in range(11, 21)]
        assert snapshot(unusable).observability == OBSERVABILITY_BLIND

    def test_ocr_text_does_not_make_a_window_look_readable(self):
        # OCR-derived elements are targets, not evidence that the app reports
        # its controls. Counting them would hide the blindness they work around.
        frame = [control(i) for i in range(1, 4)]
        text = [
            ElementNode(
                id=100 + i, role=ROLE_TEXT, name=f"line {i}",
                rect=(0, i * 30, 400, i * 30 + 20), center=(200, i * 30 + 10),
                source=SOURCE_OCR,
            )
            for i in range(10)
        ]
        assert snapshot(frame + text).observability == OBSERVABILITY_BLIND


class TestModePairing:
    def test_rich_gives_an_element_list_and_no_image(self):
        view = observe_for_planner(snapshot([control(i) for i in range(1, 21)]))
        assert view.mode == OBSERVABILITY_RICH
        assert view.image_b64 is None
        assert "Interactive Elements" in view.markdown

    def test_blind_gives_a_marked_screenshot_and_a_legend(self):
        view = observe_for_planner(snapshot([control(i) for i in range(1, 4)]))
        assert view.mode == OBSERVABILITY_BLIND
        assert view.image_b64, "a blind window must be shown, not described"
        assert "Marks on the screenshot" in view.markdown

    def test_blind_does_not_also_send_the_element_json(self):
        # One observation per mode. Sending both invites the model to trust
        # whichever contradicts the other, and the element list is the one
        # already known to be wrong here.
        view = observe_for_planner(snapshot([control(i) for i in range(1, 4)]))
        assert "Interactive Elements" not in view.markdown

    def test_every_legend_entry_is_a_resolvable_id(self):
        # The defect that made the previous overlay unusable: it numbered marks
        # from a different sequence than the one the click path resolves.
        elements = [control(i) for i in range(1, 4)]
        snap = snapshot(elements)
        view = observe_for_planner(snap)

        for element in elements:
            assert f"[{element.id}]" in view.markdown
            assert snap.graph.by_id(element.id) is not None

    def test_a_blind_window_with_no_screenshot_still_produces_something(self):
        # Capture can fail. Degrading to the text rendering is worse, but a
        # goal that proceeds badly beats one that cannot proceed.
        view = observe_for_planner(snapshot([control(i) for i in range(1, 4)], png=False))
        assert view.mode == OBSERVABILITY_BLIND
        assert view.image_b64 is None
        assert view.markdown


class TestCoordinatesAreNotOffered:
    """No tool may advertise a raw screen position to the model.

    Every coordinate in the three recorded failures was invented. The planner
    prompt now forbids them; this asserts the schema agrees, because a prompt
    rule contradicted by the tool list is a rule the model will break.
    """

    POSITIONAL = {"x", "y", "from_x", "from_y", "to_x", "to_y"}

    def test_no_tool_declares_a_coordinate_parameter(self):
        offenders = {
            tool.name: sorted({p.name for p in tool.params} & self.POSITIONAL)
            for tool in ALL_TOOLS
            if {p.name for p in tool.params} & self.POSITIONAL
        }
        assert offenders == {}, (
            f"these tools still offer the model a position to invent: {offenders}"
        )

    def test_click_offers_an_id_and_a_name_instead(self):
        click = next(t for t in ALL_TOOLS if t.name == "cua_click")
        params = {p.name for p in click.params}
        assert "element_id" in params
        assert "target_name" in params

    def test_drag_takes_two_elements(self):
        drag = next(t for t in ALL_TOOLS if t.name == "cua_drag")
        params = {p.name for p in drag.params}
        assert {"from_element_id", "to_element_id"} <= params
