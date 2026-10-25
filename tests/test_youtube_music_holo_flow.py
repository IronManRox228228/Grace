"""End to end: the real-desktop demo task driven by sloppy Holo outputs.

Planner replies are modelled on the captures in runs/2026-10-25_21-02-50 and
the earlier runs: invented tool names, call syntax, bogus element ids,
coordinates in [0, 1000], "Return" as a key. The desktop is the stateful
YouTubeMusicSim; nothing touches the real screen, mic or input devices.
"""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.agent.grounder import GroundedPoint
from grace.agent.loop import AgentLoop
from grace.harness.sim_desktop import SimDispatcher, SimPerception, YouTubeMusicSim

GOAL = "open youtube music in my browser, search for the song signal by home, and play it"
WIN = {"title": "YouTube Music - Personal - Microsoft Edge Beta"}


class DescribingSim(YouTubeMusicSim):
    """Adds the read-only tools the base sim answers with a bare ok."""

    def dispatch(self, intent, confirmed=False):
        if intent.tool == "describe_screen":
            self.dispatched_actions.append({"tool": intent.tool, "params": intent.params})
            names = [e.name for e in self.states[self.current_state_name].elements if e.name]
            return {"status": "ok", "text": "Active window: x\n" + "\n".join(f"- '{n}'" for n in names)}
        return super().dispatch(intent, confirmed)


def _plan(action, params=None, **extra):
    return json.dumps({"thought": "t", "action": action, "params": params or {}, **extra})


HOLO = [
    # bogus element_id (the planner is blind in the fleet preset, so every id is unshown)
    _plan("cua_click", {"window": WIN, "element_id": 16,
                        "target_name": "Search bar with placeholder text 'Search songs, albums, artists, podcasts'",
                        "click_count": 1}),
    # call syntax in the action field
    _plan("cua_type_text(text='signal by home', replace=False)", {"window": WIN}),
    # no cua_ prefix, "Return" as the key name
    _plan("press_key", {"window": WIN, "key": "Return"}, expect="Search results displayed"),
    _plan("cua_screenshot", {"window": WIN}),
    # the exact invented name that ended run 6
    _plan("cua_describe_screen", {}),
    # Play by name, with a bogus id and coordinates in [0, 1000]
    _plan("click", {"window": WIN, "element_id": 99, "target_name": "Play", "x": 500, "y": 300}),
    _plan("describe_screen", {}),
    _plan("converse", {}, is_completed=True, final_response="Playing Signal by Home."),
]


@pytest.mark.asyncio
async def test_demo_task_completes_with_sloppy_holo_outputs(caplog):
    sim = DescribingSim()
    gemma = AsyncMock()
    gemma.generate_text.side_effect = list(HOLO)

    async def locate(description, **kw):
        # The grounder is accurate: search box vs the Play button.
        return GroundedPoint(x=657, y=78) if "search" in description.lower() else GroundedPoint(x=1140, y=230)

    grounder = MagicMock()
    grounder.locate = AsyncMock(side_effect=locate)
    inner = MagicMock()
    for name in ("_emit_tool_started", "_emit_tool_finished", "_emit_cua_started"):
        setattr(inner, name, AsyncMock())
    dispatcher = SimDispatcher(sim, inner)
    settler = MagicMock()
    settler.wait_for_settle = AsyncMock(return_value=SimpleNamespace(settled=True))

    loop = AgentLoop(gemma=gemma, dispatcher=dispatcher, perception=SimPerception(sim),
                     grounder=grounder, settler=settler)
    res = await loop.run(GOAL)

    assert res["status"] == "ok", res
    assert res["final_response"] == "Playing Signal by Home."
    assert sim.is_goal_state()
    assert loop._pending is None
    # The harness may insert cua_activate when the planned window title differs.
    calls = [c for c in dispatcher.calls if c["tool"] != "cua_activate"]
    tools = [c["tool"] for c in calls]
    assert tools == [
        "cua_click", "cua_type_text", "cua_press_key", "cua_screenshot",
        "describe_screen", "cua_click", "describe_screen",
    ]
    # Pixel clicks came from the grounder, not from the planner's [0, 1000] guess.
    play = calls[5]["params"]
    assert (play["x"], play["y"]) == (1140, 230)
    assert calls[1]["params"]["text"] == "signal by home"
