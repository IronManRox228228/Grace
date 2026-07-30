"""Tests for the planner / grounder split.

The split exists because one model call was doing goal decomposition, tool
selection, and pixel grounding at once - and the visible symptom was Grace
typing a YouTube search into the browser's address bar. The planner now works
from the element graph's `frame` field and only hands off to UI-TARS when a
target genuinely cannot be named.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.agent.grounder import Grounder, describe_target, scale_to_screen
from grace.agent.planner import (
    Planner,
    PlannedStep,
    PlannerBudgetExceeded,
    get_planner_system_prompt,
    parse_planned_step,
)
from grace.agent.safety import SafetyGuard
from grace.llm.gemma_client import RateLimitError


class FakeLLM:
    """Returns canned responses and records the calls it received."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def generate_text(self, prompt, system_prompt, **kwargs):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, **kwargs})
        if not self._responses:
            return None
        value = self._responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


STEP_JSON = (
    '{"thought": "focus the site search box", "action": "cua_click", '
    '"params": {"element_id": 4}, "expect": "the YouTube search box is focused", '
    '"is_completed": false, "user_update": "Clicking the search box"}'
)


class TestParsePlannedStep:
    def test_parses_plain_json(self):
        step = parse_planned_step(STEP_JSON)
        assert step.action == "cua_click"
        assert step.params == {"element_id": 4}
        assert step.expect == "the YouTube search box is focused"
        assert step.is_completed is False

    def test_parses_fenced_json(self):
        step = parse_planned_step(f"```json\n{STEP_JSON}\n```")
        assert step is not None and step.action == "cua_click"

    def test_parses_json_surrounded_by_prose(self):
        step = parse_planned_step(f"Sure, here you go:\n{STEP_JSON}\nHope that helps.")
        assert step is not None and step.action == "cua_click"

    def test_rejects_json_with_no_action(self):
        assert parse_planned_step('{"thought": "hmm", "params": {}}') is None

    def test_rejects_blank_action(self):
        assert parse_planned_step('{"action": "   ", "params": {}}') is None

    def test_rejects_non_json(self):
        assert parse_planned_step("I think we should click the button.") is None

    def test_rejects_empty_and_none(self):
        assert parse_planned_step("") is None
        assert parse_planned_step(None) is None

    def test_non_dict_params_are_replaced(self):
        step = parse_planned_step('{"action": "cua_click", "params": "nope"}')
        assert step.params == {}

    def test_converse_response_becomes_final_response(self):
        step = parse_planned_step(
            '{"action": "converse", "params": {"response": "All done."}, "is_completed": true}'
        )
        assert step.final_response == "All done."
        assert step.is_completed is True

    def test_user_update_defaults_to_action(self):
        step = parse_planned_step('{"action": "cua_scroll", "params": {}}')
        assert "cua_scroll" in step.user_update


class TestNeedsGrounding:
    """UI-TARS should be invoked only when nothing else can resolve the target."""

    def test_element_id_needs_no_grounding(self):
        step = PlannedStep(action="cua_click", params={"element_id": 4, "target_name": "Search"})
        assert step.needs_grounding is False

    def test_coordinates_need_no_grounding(self):
        step = PlannedStep(action="cua_click", params={"x": 100, "y": 200, "target_name": "Search"})
        assert step.needs_grounding is False

    def test_bare_name_needs_grounding(self):
        step = PlannedStep(action="cua_click", params={"target_name": "Play"})
        assert step.needs_grounding is True

    def test_describe_needs_grounding(self):
        step = PlannedStep(action="cua_click", params={"describe": "the red play triangle"})
        assert step.needs_grounding is True

    def test_non_pointing_actions_never_ground(self):
        assert PlannedStep(action="cua_type_text", params={"target_name": "Search"}).needs_grounding is False
        assert PlannedStep(action="converse", params={}).needs_grounding is False

    def test_no_target_at_all_does_not_ground(self):
        assert PlannedStep(action="cua_click", params={}).needs_grounding is False


class TestPlannerCalls:
    def test_returns_a_step(self):
        llm = FakeLLM([STEP_JSON])
        step = asyncio.run(Planner(llm).plan(goal="play a video", elements_prompt="[]"))
        assert step.action == "cua_click"

    def test_sends_no_image(self):
        """The planner reasons over element JSON; a screenshot is the grounder's job."""
        llm = FakeLLM([STEP_JSON])
        asyncio.run(Planner(llm).plan(goal="g", elements_prompt="[]"))
        assert "image_b64" not in llm.calls[0]

    def test_prompt_carries_the_element_list_and_goal(self):
        llm = FakeLLM([STEP_JSON])
        asyncio.run(Planner(llm).plan(
            goal="play lo-fi on YouTube",
            elements_prompt='[{"id":4,"role":"searchbox","frame":"page"}]',
            window_title="YouTube - Edge",
        ))
        prompt = llm.calls[0]["prompt"]
        assert "play lo-fi on YouTube" in prompt
        assert '"frame":"page"' in prompt
        assert "YouTube - Edge" in prompt

    def test_expectation_note_goes_last(self):
        """The freshest correction signal should be closest to the answer."""
        llm = FakeLLM([STEP_JSON])
        asyncio.run(Planner(llm).plan(
            goal="g", elements_prompt="[]", history="did stuff",
            expectation_note="IT FAILED: wrong focus",
        ))
        prompt = llm.calls[0]["prompt"]
        assert prompt.index("IT FAILED") > prompt.index("did stuff")

    def test_empty_response_returns_none(self):
        assert asyncio.run(Planner(FakeLLM([""])).plan(goal="g", elements_prompt="[]")) is None

    def test_unparseable_response_returns_none(self):
        assert asyncio.run(Planner(FakeLLM(["nope"])).plan(goal="g", elements_prompt="[]")) is None

    def test_rate_limit_propagates(self):
        planner = Planner(FakeLLM([RateLimitError("429")]))
        with pytest.raises(RateLimitError):
            asyncio.run(planner.plan(goal="g", elements_prompt="[]"))


class TestPlannerBudget:
    """The cap is what actually bounds quota use, since the key rate-limits fast."""

    def test_cap_is_enforced(self):
        llm = FakeLLM([STEP_JSON] * 10)
        planner = Planner(llm, max_calls=3)

        async def drive():
            for _ in range(3):
                await planner.plan(goal="g", elements_prompt="[]")
            with pytest.raises(PlannerBudgetExceeded):
                await planner.plan(goal="g", elements_prompt="[]")

        asyncio.run(drive())
        assert len(llm.calls) == 3

    def test_failed_parses_still_consume_budget(self):
        # Otherwise a model stuck emitting prose loops until the API cuts us off.
        llm = FakeLLM(["garbage", "garbage"])
        planner = Planner(llm, max_calls=2)

        async def drive():
            await planner.plan(goal="g", elements_prompt="[]")
            await planner.plan(goal="g", elements_prompt="[]")
            with pytest.raises(PlannerBudgetExceeded):
                await planner.plan(goal="g", elements_prompt="[]")

        asyncio.run(drive())

    def test_reset_restores_budget(self):
        planner = Planner(FakeLLM([STEP_JSON] * 5), max_calls=2)
        asyncio.run(planner.plan(goal="g", elements_prompt="[]"))
        assert planner.calls_remaining == 1
        planner.reset()
        assert planner.calls_remaining == 2
        assert planner.calls_made == 0


class TestPlannerPrompt:
    def test_states_the_frame_rule(self):
        prompt = get_planner_system_prompt()
        assert 'frame: "page"' in prompt
        assert "address bar" in prompt.lower()

    def test_asks_for_json_and_never_for_thought_action(self):
        # The old single prompt said "Do NOT output JSON" while the user turn
        # demanded JSON-only. Each prompt must now be self-consistent.
        prompt = get_planner_system_prompt()
        assert "JSON" in prompt
        assert "Do NOT output JSON" not in prompt

    def test_requires_an_expectation(self):
        assert "expect" in get_planner_system_prompt()


class TestGrounderScaling:
    """UI-TARS answers in the space of the image it was sent, not the screen."""

    def test_downscaled_image_scales_up(self):
        assert scale_to_screen(640, 360, (1280, 720), (1920, 1080)) == (960, 540)

    def test_same_size_is_identity(self):
        assert scale_to_screen(100, 200, (1920, 1080), (1920, 1080)) == (100, 200)

    def test_normalised_thousand_space_is_detected(self):
        # A coordinate larger than the image it was given cannot be a pixel.
        assert scale_to_screen(500, 500, (400, 300), (1920, 1080)) == (960, 540)

    def test_zero_sizes_are_passed_through(self):
        assert scale_to_screen(10, 20, (0, 0), (1920, 1080)) == (10, 20)


class TestDescribeTarget:
    def test_explicit_description_wins(self):
        assert describe_target({"describe": "the red circle", "target_name": "X"}) == "the red circle"

    def test_frame_becomes_words(self):
        assert "web page content" in describe_target({"target_name": "Search", "frame": "page"})
        assert "browser's toolbar" in describe_target({"target_name": "Address", "frame": "chrome"})

    def test_role_and_name_combine(self):
        described = describe_target({"target_name": "Play", "role": "button"})
        assert "button" in described and "Play" in described

    def test_bare_name(self):
        assert describe_target({"target_name": "Submit"}) == "labelled 'Submit'"


class TestGrounderBackend:
    def test_backend_starts_lazily_and_only_once(self):
        starts = []

        async def start():
            starts.append(1)
            return True

        llm = FakeLLM(["Thought: found it\nAction: click(start_box='(100,200)')"] * 2)
        grounder = Grounder(llm, on_demand_start=start)

        async def drive():
            await grounder.locate("a button", b"png", (1920, 1080), (1920, 1080))
            await grounder.locate("another", b"png", (1920, 1080), (1920, 1080))

        asyncio.run(drive())
        assert len(starts) == 1

    def test_no_call_without_a_screenshot(self):
        llm = FakeLLM(["Thought: x\nAction: click(start_box='(1,2)')"])
        grounder = Grounder(llm)
        assert asyncio.run(grounder.locate("a button", b"", (100, 100), (100, 100))) is None
        assert llm.calls == []

    def test_locates_and_scales(self):
        llm = FakeLLM(["Thought: the play button\nAction: click(start_box='<|box_start|>(320,180)<|box_end|>')"])
        point = asyncio.run(Grounder(llm).locate("play button", b"png", (640, 360), (1920, 1080)))
        assert (point.x, point.y) == (960, 540)

    def test_wait_response_yields_no_point(self):
        llm = FakeLLM(["Thought: cannot see it\nAction: wait()"])
        assert asyncio.run(Grounder(llm).locate("ghost", b"png", (100, 100), (100, 100))) is None

    def test_unparseable_response_yields_no_point(self):
        llm = FakeLLM(["I have no idea where that is."])
        assert asyncio.run(Grounder(llm).locate("ghost", b"png", (100, 100), (100, 100))) is None

    def test_llm_error_is_swallowed(self):
        llm = FakeLLM([RuntimeError("backend down")])
        assert asyncio.run(Grounder(llm).locate("x", b"png", (100, 100), (100, 100))) is None

    def test_grounder_prompt_is_ui_tars_native_not_graces_tool_schema(self):
        llm = FakeLLM(["Thought: x\nAction: click(start_box='(1,2)')"])
        asyncio.run(Grounder(llm).locate("a button", b"png", (10, 10), (10, 10)))
        system = llm.calls[0]["system_prompt"]
        assert "start_box" in system
        assert "cua_click" not in system


class TestSafetyKeyNormalisation:
    """`Alt+F4` used to slip past a case-sensitive exact-string comparison."""

    @pytest.mark.parametrize("key", [
        "alt+f4", "Alt+F4", "ALT+F4", "alt_l+f4", "Control_L+w", "ctrl+w", "CTRL+W",
    ])
    def test_dangerous_hotkeys_are_caught(self, key):
        is_safe, prompt = SafetyGuard.evaluate("cua_press_key", {"key": key})
        assert is_safe is False and prompt

    @pytest.mark.parametrize("key", ["Return", "ctrl+c", "Tab", "ctrl+a", ""])
    def test_harmless_keys_pass(self, key):
        is_safe, _ = SafetyGuard.evaluate("cua_press_key", {"key": key})
        assert is_safe is True

    def test_modifier_order_does_not_matter(self):
        is_safe, _ = SafetyGuard.evaluate("cua_press_key", {"key": "shift+ctrl+w"})
        assert is_safe is False

    def test_non_string_key_is_safe(self):
        assert SafetyGuard.evaluate("cua_press_key", {"key": None})[0] is True
