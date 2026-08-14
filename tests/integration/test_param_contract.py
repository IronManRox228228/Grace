"""The planner and the executor must mean the same thing by a parameter name.

The planner's system prompt tells the model, in order of preference, to target
an element by ``element_id`` (``agent/planner.py``: "1. `element_id` - an `id`
from the elements list. Always prefer this."). The tool schema rendered into
*the same prompt* advertises ``element_index`` (``intent/tools.py``), and
``element_id`` appears in no ``ToolParam`` at all. The model therefore sees both
names and reasonably emits either.

Nothing between the two validates the dict. ``PlannedStep.params`` is handed to
``Intent``, then to ``Dispatcher._execute_cua``, then straight into
``ComputerUse.perform`` - four hops, no schema check. And ``perform`` is stubbed
in both the tape generator and replay, so the whole of ``automation/`` is dark
to the corpus.

The consequence is not a tidy error. ``_set_value`` guards its blind path with
``elif element_index is not None or target_name:`` - a plan carrying
``element_id`` satisfies neither, so an element that cannot be resolved falls
through to ``ctrl+a``, ``delete``, and typing into whatever happens to hold
keyboard focus.

No input is ever dispatched by these tests: pyautogui and the element graph are
both replaced.
"""

import sys
import types

import pytest

from grace.automation.computer_use import ComputerUse
from grace.intent.tools import ALL_TOOLS

pytestmark = pytest.mark.integration


class FakeAutoGui:
    """Records what would have been sent to the desktop."""

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

    def moveTo(self, *a, **k):
        self.calls.append(("moveTo", a, k))


@pytest.fixture
def desktop(monkeypatch):
    """A ComputerUse whose input goes nowhere and whose screen has no elements.

    An empty element graph is the case that matters: it is what every
    unresolvable target looks like, and it is the state WhatsApp presents.
    """
    fake = FakeAutoGui()
    monkeypatch.setitem(sys.modules, "pyautogui", fake)

    perception = types.SimpleNamespace(
        get_graph_builder=lambda: types.SimpleNamespace(get=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "grace.agent.perception", perception)

    cua = ComputerUse()
    cua.start()
    yield cua, fake
    cua.stop()


def params_for(tool_name: str) -> set[str]:
    for tool in ALL_TOOLS:
        if tool.name == tool_name:
            return {p.name for p in tool.params}
    raise AssertionError(f"{tool_name} is not a declared tool")


class TestElementTargetNaming:
    """`element_id` and `element_index` must not mean different things."""

    def test_the_schema_and_the_prompt_agree_on_one_name(self):
        # The planner prompt names `element_id`; the schema must offer it, or
        # the model is being told to emit a parameter no tool declares.
        targetable = ["cua_click", "cua_set_value", "cua_secondary_action"]
        disagreeing = [
            name for name in targetable
            if "element_id" not in params_for(name)
        ]
        assert disagreeing == [], (
            f"the planner prompt tells the model to use `element_id`, but these "
            f"tools declare only `element_index`: {disagreeing}"
        )

    @pytest.mark.parametrize("param", ["element_id", "element_index"])
    def test_set_value_refuses_an_unresolvable_target(self, desktop, param):
        # The dangerous one. Falling through here means select-all, delete and
        # type into an arbitrary focused window.
        cua, fake = desktop
        result = cua.perform("set_value", {param: 99, "value": "hello"})

        assert result.get("status") == "element_not_found", (
            f"with {param}=99 unresolvable, _set_value returned {result!r}"
        )
        assert fake.calls == [], (
            f"input was dispatched despite an unresolvable target: {fake.calls}"
        )

    @pytest.mark.parametrize("param", ["element_id", "element_index"])
    def test_secondary_action_refuses_an_unresolvable_target(self, desktop, param):
        cua, fake = desktop
        result = cua.perform(
            "secondary_action", {param: 99, "action": "right click"}
        )
        assert result.get("ok") is not True
        assert fake.calls == []


class TestTypeText:
    def test_replace_clears_the_field_first(self, desktop):
        # The WhatsApp failure: "Chemistry" was still in the search box, so
        # typing "Coordination Compounds" produced
        # "ChemistryCoordination Compounds" and no results.
        cua, fake = desktop
        cua.perform("type_text", {"text": "Coordination Compounds", "replace": True})

        assert ("hotkey", "ctrl", "a") in fake.calls, (
            f"replace=True did not clear the field first: {fake.calls}"
        )
        assert ("write", "Coordination Compounds") in fake.calls
        assert fake.calls.index(("hotkey", "ctrl", "a")) < fake.calls.index(
            ("write", "Coordination Compounds")
        ), "the field was cleared after typing, not before"

    def test_append_is_still_the_default(self, desktop):
        # Typing into an empty field must not select-all first; that would make
        # every ordinary keystroke destructive.
        cua, fake = desktop
        cua.perform("type_text", {"text": "hello"})
        assert ("hotkey", "ctrl", "a") not in fake.calls
        assert ("write", "hello") in fake.calls

    def test_replace_is_declared_in_the_schema(self):
        assert "replace" in params_for("cua_type_text"), (
            "the executor supports `replace` but the model is never told it exists"
        )
