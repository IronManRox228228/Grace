"""A destructive tool must be confirmed whichever path reaches it.

`SafetyGuard` was consulted in exactly one place: `AgentLoop._continue`, just
before it dispatched a planned step. That covers the agentic path and nothing
else.

But `delete_file`, `close_app` and `lock_computer` are all in
`CapabilityRouter.FAST_PATH_TOOLS`, and the fast path calls
`Dispatcher.execute` straight from `main.py`. So the guard was skipped
precisely for the requests simple enough to route directly - which, for
single-tool requests like "delete my tax return", is all of them. The more
clearly the user asked for something irreversible, the less likely it was to be
confirmed.

The fix is to ask at the dispatch boundary, which both paths cross. These tests
are parameterised over the guard's own rule set rather than a copied list, so a
rule added later is covered without editing this file.
"""

import asyncio

import pytest

from grace.agent.safety import SafetyGuard
from grace.intent.parser import Intent
from grace.tools.dispatcher import Dispatcher

pytestmark = pytest.mark.integration


class RecordingDispatcher(Dispatcher):
    """A dispatcher that records what it would have executed."""

    def __init__(self):
        super().__init__()
        self.executed: list[str] = []

    async def _execute(self, intent: Intent) -> dict:
        self.executed.append(intent.tool)
        return {"status": "ok", "text": "done"}


@pytest.fixture
def dispatcher():
    return RecordingDispatcher()


def params_for(tool: str) -> dict:
    """Plausible params, so nothing is refused for the wrong reason."""
    return {"name": "quarterly-report.pdf"} if tool != "lock_computer" else {}


@pytest.mark.parametrize("tool", sorted(SafetyGuard.CONFIRMATION_REQUIRED_TOOLS))
def test_destructive_tools_are_intercepted_at_the_dispatch_boundary(dispatcher, tool):
    result = asyncio.run(dispatcher.execute(Intent(tool=tool, params=params_for(tool))))

    assert result["status"] == "confirmation_required", (
        f"{tool} executed without confirmation: {result!r}"
    )
    assert dispatcher.executed == [], (
        f"{tool} reached its handler before the user answered: {dispatcher.executed}"
    )
    assert result.get("confirmation_prompt"), "refused without asking anything"


@pytest.mark.parametrize("key", sorted(SafetyGuard.CONFIRMATION_REQUIRED_KEYS))
def test_window_closing_hotkeys_are_intercepted_too(dispatcher, key):
    # The guard's second rule set. These reach the desktop through a generic
    # tool, so a check keyed only on the tool name would let them all past.
    result = asyncio.run(
        dispatcher.execute(Intent(tool="cua_press_key", params={"key": key}))
    )
    assert result["status"] == "confirmation_required"
    assert dispatcher.executed == []


def test_confirmed_actions_actually_run(dispatcher):
    # The other half of the guarantee, and the one that broke before: a parked
    # action that the user approved has to be able to get past the guard, or
    # "yes" silently does nothing.
    result = asyncio.run(
        dispatcher.execute(
            Intent(tool="delete_file", params={"name": "notes.txt"}), confirmed=True
        )
    )
    assert dispatcher.executed == ["delete_file"]
    assert result["status"] == "ok"


def test_ordinary_tools_are_not_delayed(dispatcher):
    # A guard that asks about everything gets answered by reflex, which is worse
    # than no guard for a user whose only input channel is speech.
    result = asyncio.run(
        dispatcher.execute(Intent(tool="open_app", params={"name": "Notepad"}))
    )
    assert dispatcher.executed == ["open_app"]
    assert result["status"] == "ok"


def test_a_refused_dispatch_is_not_taped_as_an_execution(dispatcher, monkeypatch):
    # The corpus records dispatches to prove the Rust port takes the same route.
    # A call that was refused is not a call that happened, and taping it as one
    # would make replay demand the port execute something it must not.
    taped: list[tuple] = []

    class FakeRecorder:
        def record_dispatch(self, tool, params, result):
            taped.append((tool, params, result))

    monkeypatch.setattr("grace.tools.dispatcher.get_recorder", lambda: FakeRecorder())

    asyncio.run(dispatcher.execute(Intent(tool="delete_file", params={"name": "x.txt"})))
    assert taped == []

    asyncio.run(dispatcher.execute(Intent(tool="open_app", params={"name": "Notepad"})))
    assert [t[0] for t in taped] == ["open_app"]


class TestKeyAliasNormalisation:
    """pyautogui's own key names, not just the X11 keysyms, must be recognised.

    The guard only aliased control_l/shift_l/alt_l/super_l style names. pyautogui
    itself reports altleft, ctrlleft, shiftleft and winleft/winright/super/win,
    so a press coming through that path skipped the guard entirely.
    """

    @pytest.mark.parametrize("key", [
        "altleft+f4", "ctrlleft+w", "ctrl+f4", "shift+delete", "win+l",
    ])
    def test_native_pyautogui_names_still_require_confirmation(self, dispatcher, key):
        result = asyncio.run(
            dispatcher.execute(Intent(tool="cua_press_key", params={"key": key}))
        )
        assert result["status"] == "confirmation_required", key
        assert dispatcher.executed == []

    def test_an_unguarded_hotkey_is_not_delayed(self, dispatcher):
        result = asyncio.run(
            dispatcher.execute(Intent(tool="cua_press_key", params={"key": "ctrl+c"}))
        )
        assert result["status"] == "ok"
        assert dispatcher.executed == ["cua_press_key"]


class TestGuardCoverage:
    """The rule sets must name tools that exist, or they guard nothing."""

    def test_every_guarded_tool_is_a_real_tool(self):
        from grace.intent.tools import ALL_TOOLS

        declared = {t.name for t in ALL_TOOLS}
        unknown = SafetyGuard.CONFIRMATION_REQUIRED_TOOLS - declared
        assert unknown == set(), (
            f"the guard protects tools that no longer exist: {unknown} - "
            f"a renamed tool silently loses its confirmation"
        )

    def test_the_fast_path_contains_guarded_tools(self):
        # If this ever becomes empty the boundary fix is still correct, but the
        # reason for it has gone - and it should be noticed rather than assumed.
        from grace.intent.router import CapabilityRouter

        fast = getattr(CapabilityRouter, "FAST_PATH_TOOLS", set())
        assert SafetyGuard.CONFIRMATION_REQUIRED_TOOLS & set(fast), (
            "no destructive tool is on the fast path any more; this suite's "
            "premise should be re-checked"
        )
