"""The scripted tape generator, and the property that makes it useful.

A generated tape is only worth anything if it replays to itself. That is the
same claim ``test_harness_replay`` makes for a recorded turn, and it has to
hold for every scenario in the catalogue - a scenario that generates cleanly
but cannot be reproduced is a tape the Rust port can never be graded against.

The other thing tested here is safety. This generator constructs a real
``GraceApp`` and runs it against the real ``Dispatcher`` on the developer's own
machine. If a script routes somewhere unintended, nothing may lock the
workstation or move a file to the recycle bin to prove it.
"""

import asyncio
import json
import os

import pytest

from grace.harness.generate import (
    SCENARIOS,
    SIDE_EFFECTING_HANDLERS,
    RateLimited,
    Scenario,
    ScriptExhausted,
    Turn,
    generate,
)
from grace.harness.recorder import reset_recorder_for_tests
from grace.harness.tape import Tape


@pytest.fixture(autouse=True)
def _no_leaked_recorder():
    reset_recorder_for_tests()
    yield
    reset_recorder_for_tests()


# -- safety ----------------------------------------------------------------


def test_every_side_effecting_tool_leaf_is_replaced():
    """A tool the generator forgets to stub is one it would really perform.

    Derived from the tool schema rather than from a hand-kept list, so adding a
    tool to Grace fails this test until the generator is taught to stub it.
    `converse` is excluded deliberately: it is pure, and running the real one
    is the point.
    """
    from grace.intent.tools import ALL_TOOLS

    dispatchable = {
        tool.name for tool in ALL_TOOLS if not tool.name.startswith("cua_")
    } - {"converse"}
    stubbed = {name.lstrip("_") for name in SIDE_EFFECTING_HANDLERS}

    assert dispatchable == stubbed, (
        "SIDE_EFFECTING_HANDLERS is out of step with the tool schema: "
        f"unstubbed={sorted(dispatchable - stubbed)}, "
        f"stale={sorted(stubbed - dispatchable)}"
    )


def test_an_unscripted_tool_is_refused_rather_than_performed(tmp_path):
    """Reaching an unscripted tool must be reported, and must not run it."""
    scenario = Scenario(
        name="unscripted_tool",
        covers="fixture",
        turns=[Turn("lock my computer", [
            '{"tool": "lock_computer", "params": {}}',
        ])],
        # No tool_results entry for lock_computer, on purpose.
    )
    with pytest.raises(ScriptExhausted, match="lock_computer"):
        generate(scenario, str(tmp_path))

    # The tape is still on disk, so the route it actually took can be read.
    tape = Tape.load(os.path.join(str(tmp_path), scenario.name))
    assert tape.dispatch[0]["result"]["status"] == "error"


def test_running_off_the_end_of_a_script_terminates_and_raises(tmp_path):
    """A short script must fail, and must not spin.

    It cannot fail at the point of over-run: generate_text catches everything
    but RateLimitError and returns None, the planner reads that as an empty
    response, and the agent loop retries an unparseable plan forever under the
    shipped AGENT_MAX_ITERATIONS=0. The generator caps the loop and reports
    afterwards - this test is what pins that down, because without it the
    failure mode is a hang rather than a red test.
    """
    scenario = Scenario(
        name="short_script",
        covers="fixture",
        turns=[Turn("click the submit button", [
            '{"tool": "cua_click", "params": {"element_id": 0}}',
        ])],
        # The intent routes agentic, so the planner is called and has nothing.
    )
    with pytest.raises(ScriptExhausted, match="model call"):
        generate(scenario, str(tmp_path))


# -- provenance ------------------------------------------------------------


def test_a_generated_tape_says_it_is_synthetic(tmp_path):
    scenario = SCENARIOS[0]
    directory = generate(scenario, str(tmp_path))

    with open(os.path.join(directory, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    assert meta["synthetic"] is True
    assert meta["scenario"] == scenario.name
    assert meta["covers"] == scenario.covers
    # The config still has to be there: replay reads the follow-up timeout and
    # the VAD silence window back out of it.
    assert meta["config"]["followup_timeout_seconds"] == scenario.followup_timeout


def test_regenerating_replaces_the_previous_tape(tmp_path):
    """Appending to an old run's JSONL would tape a session that never happened."""
    scenario = SCENARIOS[0]
    first = Tape.load(generate(scenario, str(tmp_path)))
    second = Tape.load(generate(scenario, str(tmp_path)))

    assert [e["type"] for e in first.event_stream] == [
        e["type"] for e in second.event_stream
    ]
    assert len(second.events) == len(first.events)


# -- the catalogue ---------------------------------------------------------


def test_scenario_names_are_unique_and_described():
    names = [s.name for s in SCENARIOS]
    assert len(names) == len(set(names)), "two scenarios would share a tape directory"
    for scenario in SCENARIOS:
        assert scenario.covers.strip(), f"{scenario.name} does not say what it covers"
        assert scenario.turns, f"{scenario.name} has no turns"


def test_multi_turn_scenarios_leave_the_followup_window_open():
    """A follow-up turn with a 0s window has no chance to be heard."""
    for scenario in SCENARIOS:
        if len(scenario.turns) > 1:
            assert scenario.followup_timeout > 0, (
                f"{scenario.name} scripts {len(scenario.turns)} turns but closes "
                "the follow-up window immediately"
            )


# The round trip is the expensive part - each scenario runs a paced listening
# window per turn, twice. This subset spans the distinct shapes: no tool, a
# fast-path dispatch, a multi-step agentic goal, a parked safety confirmation
# resumed across a follow-up, and a model failure. Set GRACE_FULL_CORPUS=1 to
# run all of them.
_REPRESENTATIVE = (
    "conversation_plain",
    "fastpath_open_app",
    "agentic_two_step",
    "safety_confirm_accepted",
    "rate_limited_intent",
)


def _round_trip_scenarios():
    if os.getenv("GRACE_FULL_CORPUS", "").lower() in ("1", "true", "yes"):
        return SCENARIOS
    return [s for s in SCENARIOS if s.name in _REPRESENTATIVE]


@pytest.mark.parametrize(
    "scenario", _round_trip_scenarios(), ids=lambda s: s.name
)
def test_a_generated_tape_replays_to_itself(scenario, tmp_path):
    """The property the whole corpus rests on.

    Generation and replay share no stubbing code beyond the Null* edges, so
    this is not a tautology: generation drives the backend from a script, and
    replay drives it from the resulting recording. Agreement means the tape
    captured everything the route depended on.
    """
    from grace.harness.replay import replay

    tape = Tape.load(generate(scenario, str(tmp_path)))
    result = asyncio.run(replay(tape))

    assert result.ok, "\n" + result.report()

    # INCONCLUSIVE notes are tolerated: they mean the harness could not feed
    # the wall-clock VAD on schedule on this machine, which is a statement
    # about CPU contention and not about the backend. Anything else is a real
    # finding and must fail.
    findings = [note for note in result.notes if not note.startswith("INCONCLUSIVE")]
    assert not findings, "clean replay should have nothing to report:\n" + "\n".join(
        findings
    )


# -- the paths the catalogue exists to pin ---------------------------------


def _tape_for(name, tmp_path):
    scenario = next(s for s in SCENARIOS if s.name == name)
    return Tape.load(generate(scenario, str(tmp_path)))


def test_a_declined_confirmation_dispatches_nothing(tmp_path):
    """The one that matters most: "no" must mean nothing was performed."""
    tape = _tape_for("safety_confirm_declined", tmp_path)

    assert tape.dispatch == [], "a declined confirmation performed a tool call"
    spoken = [e["text"] for e in tape.event_stream if e["type"] == "ResponseChunk"]
    assert any("won't do that" in text for text in spoken)


def test_an_accepted_confirmation_dispatches_the_parked_step(tmp_path):
    tape = _tape_for("safety_confirm_accepted", tmp_path)

    assert tape.dispatch_calls == [{"tool": "delete_file", "params": {"name": "draft.txt"}}]


def test_an_unrelated_answer_cancels_the_parked_step(tmp_path):
    """Answering a delete prompt with a new request must not delete anything."""
    tape = _tape_for("safety_confirm_hijacked", tmp_path)

    assert tape.dispatch == [], "an unrelated utterance was treated as consent"
    transcripts = [record["text"] for record in tape.stt]
    assert transcripts == ["get rid of the draft file", "what can you do"]


def test_a_rate_limited_turn_reaches_no_tool(tmp_path):
    tape = _tape_for("rate_limited_intent", tmp_path)

    assert tape.dispatch == []
    assert tape.llm[0]["error"].startswith("RateLimitError")
    spoken = " ".join(
        e["text"] for e in tape.event_stream if e["type"] == "ResponseChunk"
    )
    assert "request limit" in spoken


def test_the_followup_chain_handles_three_turns_without_a_wake_word(tmp_path):
    tape = _tape_for("followup_chain_three_turns", tmp_path)

    assert [record["text"] for record in tape.stt] == [
        "open notepad", "now turn the volume up", "thanks that's all",
    ]
    assert [e["type"] for e in tape.event_stream].count("WakeWordDetected") == 1
    assert [call["tool"] for call in tape.dispatch_calls] == ["open_app", "adjust_volume"]


def test_each_turn_of_a_session_produces_distinct_audio(tmp_path):
    """Identical PCM across turns collapses the tape's transcript index.

    Not a cosmetic concern: it is how a replay ends up serving the last turn's
    transcript for the first turn, which reads as a backend regression.
    """
    tape = _tape_for("followup_chain_three_turns", tmp_path)

    digests = [record["pcm_digest"] for record in tape.stt]
    assert len(set(digests)) == len(digests)


def test_the_agent_loop_and_the_dispatcher_label_tools_differently(tmp_path):
    """Both ToolExecutionStarted shapes appear, which the contract allows.

    The agent loop sends {label, tool, step}; the dispatcher underneath it
    sends {label} only. A port that unified them would be changing what the
    renderer receives.
    """
    tape = _tape_for("agentic_two_step", tmp_path)

    started = [e for e in tape.event_stream if e["type"] == "ToolExecutionStarted"]
    assert any("tool" in e and "step" in e for e in started)
    assert any(set(e) == {"type", "label"} for e in started)


def test_rate_limited_replies_are_recorded_as_errors_not_empty_responses(tmp_path):
    """A tape that records a failure as an empty success is unreplayable."""
    scenario = Scenario(
        name="rate_limit_shape",
        covers="fixture",
        turns=[Turn("hello there", [RateLimited("429 slow down")])],
    )
    tape = Tape.load(generate(scenario, str(tmp_path)))

    assert tape.llm[0]["error"] == "RateLimitError: 429 slow down"
    assert tape.llm[0]["chunks"] == []
