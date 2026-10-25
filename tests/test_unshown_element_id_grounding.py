"""Tests for unshown element_id grounding and executor not-found retry (retest 5 item 1).

1. Unshown element_id:
   - Holo sends cua_click(element_id=36, target_name='Play Signal - Home link') without coordinates.
   - When no ids were rendered in the prompt, needs_grounding is True and grounder is called.
2. Shown id:
   - When element_id was rendered in the prompt, needs_grounding is False and grounding is skipped.
3. Executor not-found retry:
   - When executor returns "not found" for a named target, and this step wasn't already grounded,
     ground it once and retry the click in the same step.
   - If executor still returns "not found", never ground or retry a second time.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from grace.agent.loop import AgentLoop
from grace.agent.planner import PlannedStep
from grace.agent.perception import ScreenSnapshot
from grace.agent.grounder import GroundedPoint
from grace.fleet import build_gpu_single_preset


class TestUnshownElementIdNeedsGrounding:
    def test_exact_step_with_no_ids_rendered_needs_grounding(self):
        """Step with no ids rendered and element_id=36: needs_grounding is True."""
        step = PlannedStep(
            action="cua_click",
            params={"element_id": 36, "target_name": "Play Signal - Home link"},
            rendered_element_ids=set(),
        )
        assert step.needs_grounding is True

    def test_shown_id_skips_grounding(self):
        """A shown id still skips grounding."""
        step = PlannedStep(
            action="cua_click",
            params={"element_id": 36, "target_name": "Play Signal - Home link"},
            rendered_element_ids={36},
        )
        assert step.needs_grounding is False

    def test_unknown_rendered_ids_treats_element_id_as_absent(self):
        """When rendered_element_ids is None (unknown), element_id is treated as absent."""
        step = PlannedStep(
            action="cua_click",
            params={"element_id": 36, "target_name": "Play Signal - Home link"},
            rendered_element_ids=None,
        )
        assert step.needs_grounding is True

    def test_element_index_unshown_needs_grounding(self):
        step = PlannedStep(
            action="cua_click",
            params={"element_index": 5, "target_name": "Play"},
            rendered_element_ids=set(),
        )
        assert step.needs_grounding is True

    def test_element_index_shown_skips_grounding(self):
        step = PlannedStep(
            action="cua_click",
            params={"element_index": 5, "target_name": "Play"},
            rendered_element_ids={5},
        )
        assert step.needs_grounding is False


class TestLoopGroundingExecution:
    @pytest.mark.asyncio
    async def test_exact_step_unshown_id_calls_grounder_in_loop(self):
        """In AgentLoop, an unshown element_id causes grounder to be invoked."""
        gemma = AsyncMock()
        dispatcher = AsyncMock()
        perception = MagicMock()
        grounder = AsyncMock()

        # Step 1: Holo cua_click with unshown id 36 and target_name
        # Step 2: Completion converse step
        gemma.generate_text.side_effect = [
            '```json\n{"thought": "Click play", "action": "cua_click", "params": {"element_id": 36, "target_name": "Play Signal - Home link"}, "is_completed": false, "user_update": "Clicking Play"}\n```',
            '```json\n{"thought": "Done", "action": "converse", "params": {"response": "Playing music"}, "is_completed": true, "final_response": "Playing music"}\n```',
        ]

        snapshot = ScreenSnapshot(
            active_window=None,
            ocr_lines=[],
            width=1920,
            height=1080,
            png_bytes=b"fake-png-bytes",
        )
        perception.capture_snapshot.return_value = snapshot

        # Mock grounder locating the target
        grounder.locate.return_value = GroundedPoint(x=733, y=78)

        # Dispatcher succeeds for click and converse
        dispatcher.execute.return_value = {
            "status": "ok",
            "result": {"ok": True, "sent": True, "verified": True},
        }

        loop = AgentLoop(
            gemma=gemma,
            dispatcher=dispatcher,
            perception=perception,
            grounder=grounder,
        )

        res = await loop.run(user_goal="play signal by home")

        assert res["status"] == "ok"
        # Verify grounder was called for step 1
        assert grounder.locate.call_count == 1
        ground_call_kwargs = grounder.locate.call_args[1]
        assert "Play Signal - Home link" in ground_call_kwargs["description"]

        # Verify dispatcher received grounded coordinates (733, 78)
        first_dispatch = dispatcher.execute.call_args_list[0][0][0]
        assert first_dispatch.tool == "cua_click"
        assert first_dispatch.params["x"] == 733
        assert first_dispatch.params["y"] == 78

    @pytest.mark.asyncio
    async def test_shown_id_skips_grounding_in_loop(self):
        """When an id was rendered in the prompt, grounder is NOT called."""
        gemma = AsyncMock()
        dispatcher = AsyncMock()
        perception = MagicMock()
        grounder = AsyncMock()

        # Planner outputs shown id 16
        gemma.generate_text.side_effect = [
            '```json\n{"thought": "Click search", "action": "cua_click", "params": {"element_id": 16, "target_name": "Search"}, "is_completed": false, "user_update": "Clicking search"}\n```',
            '```json\n{"thought": "Done", "action": "converse", "params": {}, "is_completed": true, "final_response": "Done"}\n```',
        ]

        # Elements prompt contains id 16
        snapshot = ScreenSnapshot(
            active_window=None,
            ocr_lines=[],
            width=1920,
            height=1080,
            png_bytes=b"fake-png-bytes",
        )
        perception.capture_snapshot.return_value = snapshot

        dispatcher.execute.return_value = {
            "status": "ok",
            "result": {"ok": True, "sent": True, "verified": True},
        }

        loop = AgentLoop(
            gemma=gemma,
            dispatcher=dispatcher,
            perception=perception,
            grounder=grounder,
        )
        loop._fleet_config = build_gpu_single_preset()

        with patch(
            "grace.agent.loop.observe_for_planner",
            return_value=MagicMock(markdown='[{"id": 16, "role": "search", "name": "Search"}]', image_b64=None),
        ):
            res = await loop.run(user_goal="search")

        assert res["status"] == "ok"
        assert grounder.locate.call_count == 0

    @pytest.mark.asyncio
    async def test_executor_not_found_leads_to_one_grounding_retry_never_more(self):
        """When executor returns 'not found' for a named target, ground once and retry."""
        gemma = AsyncMock()
        dispatcher = AsyncMock()
        perception = MagicMock()
        grounder = AsyncMock()

        # Step 1: Planned step with a shown id (so it initially skips grounding)
        gemma.generate_text.side_effect = [
            '```json\n{"thought": "Click play", "action": "cua_click", "params": {"element_id": 16, "target_name": "Play Signal"}, "is_completed": false, "user_update": "Clicking play"}\n```',
            '```json\n{"thought": "Done", "action": "converse", "params": {}, "is_completed": true, "final_response": "Done"}\n```',
        ]

        snapshot = ScreenSnapshot(
            active_window=None,
            ocr_lines=[],
            width=1920,
            height=1080,
            png_bytes=b"fake-png-bytes",
        )
        perception.capture_snapshot.return_value = snapshot

        # Grounder locates at (450, 300)
        grounder.locate.return_value = GroundedPoint(x=450, y=300)

        # First dispatch returns not-found error; retry dispatch succeeds!
        not_found_result = {
            "status": "error",
            "error": "Target element 'Play Signal' not found via UIA, OculiX, OpenCV, or OCR",
            "text": "Action error: Target element 'Play Signal' not found via UIA, OculiX, OpenCV, or OCR",
            "result": {
                "ok": False,
                "sent": False,
                "verified": False,
                "action": "click",
                "status": "element_not_found",
                "error": "Target element 'Play Signal' not found via UIA, OculiX, OpenCV, or OCR",
            },
        }
        success_result = {
            "status": "ok",
            "result": {"ok": True, "sent": True, "verified": True, "action": "click", "x": 450, "y": 300},
        }

        # Sequence:
        # 1: initial dispatch of cua_click -> not_found_result
        # 2: retry dispatch of cua_click with grounded coords -> success_result
        # 3: dispatch of converse -> ok
        dispatcher.execute.side_effect = [
            not_found_result,
            success_result,
            {"status": "ok"},
        ]

        loop = AgentLoop(
            gemma=gemma,
            dispatcher=dispatcher,
            perception=perception,
            grounder=grounder,
        )
        loop._fleet_config = build_gpu_single_preset()

        with patch(
            "grace.agent.loop.observe_for_planner",
            return_value=MagicMock(markdown='[{"id": 16, "role": "button", "name": "Play Signal"}]', image_b64=None),
        ):
            res = await loop.run(user_goal="play signal")

        assert res["status"] == "ok"
        # Grounder was called exactly ONCE as retry
        assert grounder.locate.call_count == 1
        # Click dispatch was called twice: initial + 1 retry
        assert dispatcher.execute.call_count == 2
        first_click = dispatcher.execute.call_args_list[0][0][0]
        second_click = dispatcher.execute.call_args_list[1][0][0]
        assert first_click.tool == "cua_click"
        assert second_click.tool == "cua_click"
        assert second_click.params["x"] == 450
        assert second_click.params["y"] == 300

    @pytest.mark.asyncio
    async def test_executor_not_found_still_failing_never_retries_a_second_time(self):
        """If retry still returns not found, loop must never ground or retry a second time."""
        gemma = AsyncMock()
        dispatcher = AsyncMock()
        perception = MagicMock()
        grounder = AsyncMock()

        gemma.generate_text.side_effect = [
            '```json\n{"thought": "Click play", "action": "cua_click", "params": {"element_id": 16, "target_name": "Play Signal"}, "is_completed": false, "user_update": "Clicking play"}\n```',
            '```json\n{"thought": "Done", "action": "converse", "params": {}, "is_completed": true, "final_response": "Done"}\n```',
        ]

        snapshot = ScreenSnapshot(
            active_window=None,
            ocr_lines=[],
            width=1920,
            height=1080,
            png_bytes=b"fake-png-bytes",
        )
        perception.capture_snapshot.return_value = snapshot

        grounder.locate.return_value = GroundedPoint(x=450, y=300)

        not_found_result = {
            "status": "error",
            "error": "Target element 'Play Signal' not found via UIA, OculiX, OpenCV, or OCR",
            "result": {
                "ok": False,
                "sent": False,
                "verified": False,
                "status": "element_not_found",
            },
        }

        # Dispatcher returns not found on initial and on retry!
        dispatcher.execute.side_effect = [
            not_found_result,
            not_found_result,
            {"status": "ok"},
        ]

        loop = AgentLoop(
            gemma=gemma,
            dispatcher=dispatcher,
            perception=perception,
            grounder=grounder,
        )
        loop._fleet_config = build_gpu_single_preset()

        with patch(
            "grace.agent.loop.observe_for_planner",
            return_value=MagicMock(markdown='[{"id": 16, "role": "button", "name": "Play Signal"}]', image_b64=None),
        ):
            await loop.run(user_goal="play signal")

        # Grounder called exactly once, never twice
        assert grounder.locate.call_count == 1
        # Click dispatch was called exactly twice (1 initial + 1 retry, never 3 clicks)
        cua_dispatches = [
            call[0][0] for call in dispatcher.execute.call_args_list
            if call[0][0].tool == "cua_click"
        ]
        assert len(cua_dispatches) == 2
