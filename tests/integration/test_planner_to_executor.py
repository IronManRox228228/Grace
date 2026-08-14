"""Planner JSON must survive the whole way to the desktop, or be refused.

Between the model's answer and the mouse there are four hops - `parse_planned_step`,
`Intent`, `Dispatcher.execute`, `ComputerUse.perform` - and not one of them
validates the parameter dict. `perform` then dispatches by reflection
(`getattr(self, f"_{action}")`), so a tool the model was told about but which
has no handler fails for the first time in front of the user.

The corpus cannot catch any of this: `perform` is stubbed in both
`harness/generate.py` and `harness/replay.py`, so everything below the
dispatcher is dark to replay. These tests run the real chain over a fake input
driver instead.

The invariant that matters most: **an unresolvable target must be refused, not
approximated.** Falling through to "type into whatever has focus" is how a
search query ends up in a document, and this user cannot see it happen or undo
it with a keyboard.
"""

import asyncio
import sys
import types

import pytest

from grace.agent.planner import parse_planned_step
from grace.automation.computer_use import ComputerUse
from grace.intent.parser import Intent
from grace.perception.elements import ElementNode
from grace.tools.dispatcher import Dispatcher

pytestmark = pytest.mark.integration


class FakeAutoGui:
    """Records what would have reached the desktop. Nothing is ever sent."""

    class FailSafeException(Exception):
        pass

    def __init__(self):
        self.calls: list[tuple] = []

    def write(self, text, interval=None):
        self.calls.append(("write", text))

    def hotkey(self, *keys):
        self.calls.append(("hotkey", *keys))

    def press(self, key):
        self.calls.append(("press", key))

    def click(self, *a, **k):
        self.calls.append(("click", a, k))

    def scroll(self, *a, **k):
        self.calls.append(("scroll", a, k))

    def hscroll(self, *a, **k):
        self.calls.append(("hscroll", a, k))

    def moveTo(self, *a, **k):
        self.calls.append(("moveTo", a, k))

    def drag(self, *a, **k):
        self.calls.append(("drag", a, k))

    def rightClick(self, *a, **k):
        self.calls.append(("rightClick", a, k))

    def middleClick(self, *a, **k):
        self.calls.append(("middleClick", a, k))


class FakeGraph:
    """The smallest element graph the executor's lookups need."""

    def __init__(self, elements):
        self.elements = list(elements)
        self.window = types.SimpleNamespace(title="Test Window", hwnd=1)

    def __len__(self):
        return len(self.elements)

    @property
    def actionable_count(self):
        return sum(1 for e in self.elements if e.is_actionable)

    def by_id(self, element_id):
        return next((e for e in self.elements if e.id == element_id), None)

    def focused(self):
        return next((e for e in self.elements if e.focused), None)

    def resolve(self, element_id=None, target_name=None, frame=None, role=None):
        if element_id is not None:
            return self.by_id(element_id)
        if target_name:
            wanted = target_name.lower()
            return next(
                (e for e in self.elements if wanted in e.search_text()), None
            )
        return None


def button(element_id, name, focused=False):
    return ElementNode(
        id=element_id, role="button", name=name,
        rect=(10, 20, 110, 60), center=(60, 40), focused=focused,
    )


@pytest.fixture
def chain(monkeypatch):
    """A real Dispatcher over a real ComputerUse over a fake desktop.

    Returns a function taking planner JSON and returning the dispatch result,
    plus the list of inputs that would have been sent.
    """
    fake_gui = FakeAutoGui()
    monkeypatch.setitem(sys.modules, "pyautogui", fake_gui)

    graph = FakeGraph([button(1, "Send"), button(2, "Attach")])
    builder = types.SimpleNamespace(
        get=lambda *a, **k: graph,
        invalidate=lambda: None,
    )
    perception = types.SimpleNamespace(
        PerceptionEngine=types.SimpleNamespace(get_graph_builder=lambda: builder),
        _min_actionable=lambda: 8,
    )
    monkeypatch.setitem(sys.modules, "grace.agent.perception", perception)
    # No settle delay: the probes are exercised, the wait is not what is under
    # test and 0.25s per action would dominate the suite.
    monkeypatch.setattr("grace.automation.computer_use.SETTLE_SECONDS", 0)
    # `_wrong_window` reads the real foreground window otherwise, which makes
    # every assertion here depend on what is open on the machine running it.
    monkeypatch.setattr(
        "grace.automation.computer_use._foreground_title", lambda: "Test Window"
    )

    cua = ComputerUse()
    cua.start()
    dispatcher = Dispatcher(computer_use=cua)

    def run(planner_json: str) -> dict:
        step = parse_planned_step(planner_json)
        assert step is not None, f"planner JSON did not parse: {planner_json}"
        intent = Intent(tool=step.action, params=step.params)
        return asyncio.run(dispatcher.execute(intent))

    yield run, fake_gui, graph
    cua.stop()


def plan(action: str, **params) -> str:
    import json

    return json.dumps({
        "thought": "doing the thing",
        "action": action,
        "params": params,
        "user_update": "Working…",
        "expect": "it works",
    })


class TestUnresolvableTargetsAreRefused:
    """The whole reason this seam is tested. Nothing may fall through."""

    def test_click_on_a_missing_element_sends_nothing(self, chain):
        run, gui, _ = chain
        result = run(plan("cua_click", element_id=99, window={"title": "Test Window"}))

        inner = result.get("result", result)
        assert inner.get("status") == "element_not_found", result
        assert inner.get("sent") is False
        assert gui.calls == [], f"a click was sent at a target that does not exist: {gui.calls}"

    def test_set_value_on_a_missing_element_sends_nothing(self, chain):
        # The dangerous one: this handler's body is select-all, delete, type.
        run, gui, _ = chain
        result = run(plan("cua_set_value", element_id=99, value="hello",
                          window={"title": "Test Window"}))

        inner = result.get("result", result)
        assert inner.get("status") == "element_not_found", result
        assert gui.calls == []

    def test_set_value_with_no_target_at_all_sends_nothing(self, chain):
        run, gui, _ = chain
        result = run(plan("cua_set_value", value="hello", window={"title": "Test Window"}))

        inner = result.get("result", result)
        assert inner.get("status") == "no_target", result
        assert gui.calls == []

    def test_drag_with_one_unresolvable_end_sends_nothing(self, chain):
        run, gui, _ = chain
        result = run(plan("cua_drag", from_element_id=1, to_element_id=99,
                          window={"title": "Test Window"}))

        inner = result.get("result", result)
        assert inner.get("status") == "element_not_found", result
        assert gui.calls == [], (
            f"a drag was performed with an invented endpoint: {gui.calls}"
        )


class TestResolvableTargetsReachTheDesktop:
    """The other half: a valid plan must not be refused by the new guards."""

    def test_click_by_element_id_reaches_the_driver(self, chain):
        run, gui, graph = chain
        result = run(plan("cua_click", element_id=1, window={"title": "Test Window"}))

        inner = result.get("result", result)
        assert inner.get("ok") is True, result
        assert inner.get("sent") is True
        assert inner.get("target") == "Send"

    def test_click_by_name_reaches_the_driver(self, chain):
        run, gui, _ = chain
        result = run(plan("cua_click", target_name="attach", window={"title": "Test Window"}))
        inner = result.get("result", result)
        assert inner.get("target") == "Attach", result

    def test_drag_between_two_real_elements_is_performed(self, chain):
        run, gui, _ = chain
        result = run(plan("cua_drag", from_element_id=1, to_element_id=2,
                          window={"title": "Test Window"}))
        inner = result.get("result", result)
        assert inner.get("ok") is True, result
        assert any(call[0] == "drag" for call in gui.calls), gui.calls


class TestActionContract:
    """Every result carries `sent` and `verified`, and they mean two things."""

    @pytest.mark.parametrize("action,params", [
        ("cua_click", {"element_id": 1}),
        ("cua_type_text", {"text": "hello"}),
        ("cua_press_key", {"key": "Return"}),
        ("cua_scroll", {"scrollY": 300}),
        ("cua_list_windows", {}),
    ])
    def test_every_action_reports_both_claims(self, chain, action, params):
        run, _, _ = chain
        result = run(plan(action, window={"title": "Test Window"}, **params))
        inner = result.get("result", result)

        assert "sent" in inner, f"{action} does not say whether input was dispatched"
        assert "verified" in inner, f"{action} does not say whether it worked"
        assert inner["verified"] in (True, False, None)
        assert "evidence" in inner

    def test_unknown_is_not_reported_as_failure(self, chain):
        # An app that reports no contents cannot contradict anything. Grading
        # that as False is what stopped a run that was succeeding.
        run, _, graph = chain
        graph.elements = []
        result = run(plan("cua_press_key", key="Return", window={"title": "Test Window"}))
        inner = result.get("result", result)

        assert inner["verified"] is None, (
            f"a blind window produced a definite verdict: {inner!r}"
        )
        assert inner["sent"] is True

    def test_a_visible_change_is_reported_as_verified(self, chain):
        run, _, graph = chain

        original = graph.focused

        def focus_moves():
            # The screen the probe reads after the action differs from the one
            # it read before: focus has landed somewhere.
            graph.focused = original
            return graph.elements[1] if graph.elements else None

        graph.focused = focus_moves
        result = run(plan("cua_press_key", key="Tab", window={"title": "Test Window"}))
        inner = result.get("result", result)
        assert inner["verified"] is True, inner
        assert "focus" in inner["evidence"]


class TestPreconditions:
    def test_acting_on_a_window_that_is_not_in_front_is_refused(self, chain, monkeypatch):
        # The plan names WhatsApp; Notepad is in front. Sending the keystrokes
        # anyway types the user's message into their document.
        run, gui, _ = chain
        monkeypatch.setattr(
            "grace.automation.computer_use._foreground_title", lambda: "Untitled - Notepad"
        )
        result = run(plan("cua_type_text", text="see you at six",
                          window={"title": "WhatsApp"}))

        inner = result.get("result", result)
        assert inner.get("status") == "wrong_window", result
        assert inner.get("sent") is False
        assert gui.calls == []
        assert "Notepad" in inner.get("error", ""), (
            "the refusal must name what is actually in front, or the planner "
            "cannot correct it"
        )

    def test_a_loosely_named_window_still_matches(self, chain, monkeypatch):
        # The model writes 'Chrome'; the real title is
        # 'New Tab - Google Chrome'. Refusing that would break working plans.
        run, gui, _ = chain
        monkeypatch.setattr(
            "grace.automation.computer_use._foreground_title",
            lambda: "New Tab - Google Chrome",
        )
        result = run(plan("cua_type_text", text="hello", window={"title": "Chrome"}))
        inner = result.get("result", result)
        assert inner.get("status") != "wrong_window", result
        assert ("write", "hello") in gui.calls

    def test_a_plan_that_names_no_window_is_not_refused(self, chain, monkeypatch):
        run, gui, _ = chain
        monkeypatch.setattr(
            "grace.automation.computer_use._foreground_title", lambda: "Anything At All"
        )
        result = run(plan("cua_type_text", text="hello"))
        inner = result.get("result", result)
        assert inner.get("status") != "wrong_window", result
        assert ("write", "hello") in gui.calls
