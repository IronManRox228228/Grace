import json
from collections import namedtuple
import pytest
from grace.perception.elements import ElementNode, elements_to_prompt
from grace.perception.element_graph import ElementGraph
from grace.agent.perception import ScreenSnapshot, OcrLine, observe_for_planner


def test_single_mode_coordinates_and_no_rect():
    node = ElementNode(
        id=8,
        role="button",
        name="Initiate search",
        rect=(400, 60, 418, 84),
        center=(409, 72),
        frame="chrome",
    )
    graph = ElementGraph(elements=[node])

    # Single mode prompt with 1920x1080 screen
    prompt_single = graph.to_prompt(is_single=True, screen_size=(1920, 1080))
    # Extract JSON part from prompt
    json_start = prompt_single.find("[")
    json_end = prompt_single.rfind("]") + 1
    items_single = json.loads(prompt_single[json_start:json_end])
    assert len(items_single) == 1
    item = items_single[0]

    # center scaled: 409 * 1000 / 1920 = 213.02 -> 213; 72 * 1000 / 1080 = 66.67 -> 67
    assert item["center"] == [213, 67]
    assert "rect" not in item
    assert item["id"] == 8
    assert item["name"] == "Initiate search"

    # Fleet mode retains rect and pixel center
    prompt_fleet = graph.to_prompt(is_single=False)
    items_fleet = json.loads(prompt_fleet[prompt_fleet.find("["):prompt_fleet.rfind("]") + 1])
    assert items_fleet[0]["rect"] == [400, 60, 418, 84]
    assert items_fleet[0]["center"] == [409, 72]

    # Element resolution still returns real screen coordinates
    resolved = graph.resolve(element_id=8)
    assert resolved is not None
    assert resolved.center == (409, 72)
    assert resolved.rect == (400, 60, 418, 84)


def test_single_mode_skips_ocr_and_no_double_untrusted():
    ocr_lines = [
        OcrLine(text="Search or type URL", bounding_box=(100, 50, 300, 70)),
    ]
    snap = ScreenSnapshot(
        active_window=None,
        ocr_lines=ocr_lines,
        width=1920,
        height=1080,
    )

    # In single mode, OCR is omitted from markdown
    single_md = snap.to_markdown(is_single=True)
    assert "Search or type URL" not in single_md
    assert "Screen Text (OCR Output)" not in single_md

    # In fleet mode, OCR is included, but WITHOUT untrusted data wrapper
    # (because the outer planner prompt wraps the whole section)
    fleet_md = snap.to_markdown(is_single=False)
    assert "Search or type URL" in fleet_md
    assert "<<<UNTRUSTED_DATA_START>>>" not in fleet_md
