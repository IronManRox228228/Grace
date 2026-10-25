"""Tests for tracking rendered element IDs and guarding against hallucinated element IDs.

From real-desktop-4:
- An element_id is only valid if that id was shown to the planner in this step's prompt.
- Track the set of ids rendered into the prompt per step. If ids weren't shown,
  or this id wasn't among them, drop it with an INFO log and resolve by target_name,
  or by explicit x/y if present.
- Even for a shown id, apply the name-mismatch guard when target_name is given,
  and refuse on mismatch, as today.
- Tests built from step 4's exact params:
  - In target-guidance mode, element_id=20 is ignored and the click resolves by target_name.
  - A shown id with a matching name clicks.
  - A shown id with a mismatched name is refused.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from grace.agent.planner import (
    Planner,
    extract_rendered_element_ids,
)
from grace.automation.computer_use import ComputerUse
from grace.perception.element_graph import ElementGraph, WindowRef
from grace.perception.elements import ElementNode, FRAME_APP, FRAME_PAGE


class TestExtractRenderedElementIds:
    def test_extract_from_json_element_list(self):
        text = '[{"id": 0, "role": "button", "name": "Play"}, {"id": 16, "role": "search"}]'
        assert extract_rendered_element_ids(text) == {0, 16}

    def test_extract_from_som_legend(self):
        text = "### Marks on the screenshot\n[20] link: Planetary Warfare Play\n[35] link: Signal"
        assert extract_rendered_element_ids(text) == {20, 35}

    def test_extract_from_ui_element_prompt(self):
        text = '[9] Combobox: "Search songs" -> `cua_click(element_index=9)`'
        assert extract_rendered_element_ids(text) == {9}

    def test_empty_or_none_returns_empty_set(self):
        assert extract_rendered_element_ids("") == set()
        assert extract_rendered_element_ids(None) == set()
        assert extract_rendered_element_ids("No elements here") == set()


class TestPlannerRenderedElementIds:
    @pytest.mark.asyncio
    async def test_target_guidance_mode_drops_hallucinated_element_id(self):
        mock_llm = MagicMock()
        mock_llm.generate_text = AsyncMock(return_value=(
            '{"thought": "Find Signal by Home", "action": "cua_click", '
            '"params": {"target_name": "Song \'Signal\' by Home in the search results list", "element_id": 20}}'
        ))
        planner = Planner(llm=mock_llm)

        step = await planner.plan(
            goal="search and play signal by home",
            elements_prompt="",
            hide_tree=True,
        )

        assert step is not None
        assert "element_id" not in step.params
        assert step.params["target_name"] == "Song 'Signal' by Home in the search results list"
        assert step.rendered_element_ids == set()
        assert step.target_guidance is True

    @pytest.mark.asyncio
    async def test_single_mode_retains_shown_element_id(self):
        mock_llm = MagicMock()
        mock_llm.generate_text = AsyncMock(return_value=(
            '{"thought": "Click search", "action": "cua_click", '
            '"params": {"target_name": "Search", "element_id": 16}}'
        ))
        planner = Planner(llm=mock_llm)

        prompt_tree = '[{"id": 16, "role": "search", "name": "Search"}]'
        step = await planner.plan(
            goal="search for song",
            elements_prompt=prompt_tree,
            hide_tree=False,
        )

        assert step is not None
        assert step.params["element_id"] == 16
        assert step.rendered_element_ids == {16}

    @pytest.mark.asyncio
    async def test_single_mode_drops_unshown_element_id(self):
        mock_llm = MagicMock()
        mock_llm.generate_text = AsyncMock(return_value=(
            '{"thought": "Click 99", "action": "cua_click", '
            '"params": {"target_name": "Search", "element_id": 99}}'
        ))
        planner = Planner(llm=mock_llm)

        prompt_tree = '[{"id": 16, "role": "search", "name": "Search"}]'
        step = await planner.plan(
            goal="search for song",
            elements_prompt=prompt_tree,
            hide_tree=False,
        )

        assert step is not None
        assert "element_id" not in step.params
        assert step.params["target_name"] == "Search"


class TestComputerUseStep4Scenarios:
    """Step 4 scenarios:
    Exact params: {"target_name": "Song 'Signal' by Home in the search results list", "element_id": 20}
    Graph contains:
    - id=20: "Planetary Warfare Play" (sidebar playlist)
    - id=35: "Song 'Signal' by Home" (search result)
    """

    @pytest.fixture
    def setup_graph(self, monkeypatch):
        el20 = ElementNode(
            id=20,
            role="link",
            name="Planetary Warfare Play",
            rect=(100, 500, 224, 598),
            center=(162, 549),
            frame=FRAME_PAGE,
        )
        el35 = ElementNode(
            id=35,
            role="link",
            name="Song 'Signal' by Home in the search results list",
            rect=(400, 250, 500, 350),
            center=(450, 300),
            frame=FRAME_PAGE,
        )
        graph = ElementGraph(
            elements=[el20, el35],
            window=WindowRef(hwnd=1705348, title="YouTube Music", class_name="Chrome_WidgetWin_1"),
        )

        clicked_at = []

        cu = ComputerUse()
        cu.start()

        monkeypatch.setattr(
            "grace.automation.computer_use._foreground_title",
            lambda: "YouTube Music",
        )
        monkeypatch.setattr(
            cu,
            "_click_at",
            lambda x, y, count, method, name, **kw: {
                "ok": True,
                "sent": True,
                "verified": True,
                "action": "click",
                "x": x,
                "y": y,
                "method": method,
                "target": name,
            },
        )

        return cu, graph

    def test_target_guidance_mode_ignores_element_id_20_and_resolves_by_target_name(self, setup_graph):
        cu, graph = setup_graph
        step4_params = {
            "target_name": "Song 'Signal' by Home in the search results list",
            "element_id": 20,
            "target_guidance": True,
            "_graph": graph,
        }

        res = cu.perform("click", step4_params)

        assert res["ok"] is True
        assert res["x"] == 450
        assert res["y"] == 300
        assert res["target"] == "Song 'Signal' by Home in the search results list"

    def test_shown_id_with_matching_name_clicks(self, setup_graph):
        cu, graph = setup_graph
        params = {
            "target_name": "Song 'Signal' by Home in the search results list",
            "element_id": 35,
            "_rendered_element_ids": {35},
            "_graph": graph,
        }

        res = cu.perform("click", params)

        assert res["ok"] is True
        assert res["x"] == 450
        assert res["y"] == 300
        assert res["target"] == "Song 'Signal' by Home in the search results list"

    def test_shown_id_with_mismatched_name_is_refused(self, setup_graph):
        cu, graph = setup_graph
        params = {
            "target_name": "Song 'Signal' by Home in the search results list",
            "element_id": 20,
            "_rendered_element_ids": {20},
            "_graph": graph,
        }

        res = cu.perform("click", params)

        assert res["ok"] is False
        assert res["status"] == "element_mismatch"
        assert "Planetary Warfare Play" in res["error"]
        assert "nothing was clicked" in res["evidence"]
