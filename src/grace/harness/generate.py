"""Generate session tapes headlessly from scripted scenarios.

Phase 0 needs a corpus, and the plan asked for it to be recorded by hand. Most
of it does not have to be. What a tape has to pin down is a *route*: which
tools were asked for, in what order, with what parameters, and which events the
frontend saw along the way. Everything that decides a route - the intent
parser, ``CapabilityRouter``, ``AgentLoop``, ``SafetyGuard``, ``Dispatcher``,
``ResponseGenerator`` - is deterministic given a transcript and a set of model
replies. Both of those can be scripted, so those tapes can be generated.

Only the *external edges* are faked, and they are faked at exactly the seams
``replay.py`` already fakes:

* the model, at ``GemmaClient._stream_gemini_response`` - so ``chat()``'s own
  recording tap still runs and the tape holds a real request record;
* Whisper's model object, not ``WhisperStreaming`` - so ``_transcribe_pcm``
  still tapes the PCM it was handed;
* each ``Dispatcher._<tool>`` leaf, not ``Dispatcher.execute`` - so the tool
  labels, the ToolExecutionStarted/Finished pair, and ``record_dispatch`` are
  all the production ones. Only the call that touches the OS is replaced;
* ``ComputerUse.perform``;
* the microphone, as a paced chunk sequence, so the real VAD ends each turn.

Everything between those edges is production code, driven through
``GraceApp._handle_activation`` - the same entry point a wake word uses.

**What a generated tape cannot tell you.** It is stamped ``synthetic: true`` in
``meta.json``, and the distinction is not cosmetic:

* Its element graphs are hand-written, not walked from a live UIA tree. It
  grades the planner's use of a graph, never the fidelity of the graph itself.
  Phase 4's parity gate needs snapshots recorded from real windows.
* Its audio is a square wave. It exercises the VAD's wall-clock accounting, and
  says nothing about wake-word behaviour or transcription in a real room.
* It has no screenshot, so a step that needs grounding silently skips the
  grounder. Grounding parity needs its own screenshot corpus.
* Barge-in is invisible: with TTS stubbed, the ``tts_player.stop()`` a follow-up
  performs produces no event to compare.

Those four are the corpus that genuinely has to be recorded by hand. This
module covers the rest.

Usage::

    python -m grace.harness.generate --out corpus/
    python -m grace.harness.generate --out corpus/ --only safety_confirm_accepted
    python -m grace.harness.generate --list
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import os
import shutil
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from unittest import mock

from .clock import install_audio_clock_vad

logger = logging.getLogger("grace.harness.generate")


class ScriptExhausted(RuntimeError):
    """The backend asked for a scripted reply the scenario does not contain.

    Never defaulted around. A scenario that runs off the end of its script did
    not do what it was written to do - usually the routing went somewhere else
    - and quietly feeding it an empty response would tape that wrong route as
    if it were intended.
    """


class Overruns:
    """Collects script over-runs, to be raised once the session has ended.

    Raising at the point of over-run does not work, and finding out why is a
    finding in its own right: ``GemmaClient.generate_text`` catches every
    exception except ``RateLimitError`` and returns None
    (``llm/gemma_client.py:413``). The planner reads that as an empty response,
    the agent loop books it as an unparseable plan and tries again - and since
    ``AGENT_MAX_ITERATIONS`` defaults to 0, meaning unlimited, it tries again
    forever. A persistently failing planner therefore spins the agent loop
    without bound today, burning quota with no way for a user who cannot reach
    a keyboard to stop it. That is risk #8 in the migration plan's register,
    reachable with no porting bug at all.

    So the generator caps the loop (see ``_config_for``) and reports over-runs
    afterwards instead of trying to throw through code that swallows throws.
    """

    def __init__(self):
        self.reasons: list[str] = []

    def note(self, reason: str) -> None:
        self.reasons.append(reason)

    def raise_if_any(self, scenario_name: str) -> None:
        if not self.reasons:
            return
        raise ScriptExhausted(
            f"scenario '{scenario_name}' ran off the end of its script:\n  "
            + "\n  ".join(dict.fromkeys(self.reasons))
        )


class RateLimited:
    """Script entry: this model call raises RateLimitError instead of replying."""

    def __init__(self, message: str = "429 Too Many Requests"):
        self.message = message


# One scripted model reply: the chunk sequence to stream back, or a rate limit.
# Chunk boundaries are deliberate - see recorder.py on why they are recorded
# individually rather than joined.
Reply = Union[list[str], str, RateLimited]


@dataclass
class Turn:
    """One utterance and the model replies it should provoke, in call order."""

    transcript: str
    replies: list[Reply] = field(default_factory=list)


@dataclass
class Scenario:
    """A scripted session. Produces exactly one tape."""

    name: str
    covers: str
    turns: list[Turn]

    # Element graphs served to the agent loop, in observation order. The last
    # one repeats if the loop observes more times than there are entries, which
    # is the normal case: a screen does not change just because it was looked at
    # again.
    snapshots: list[dict] = field(default_factory=list)

    # Results for the tool leaves this scenario reaches, keyed by tool name.
    # A list serves one result per call.
    tool_results: dict[str, Any] = field(default_factory=dict)

    # Results for ComputerUse.perform, in call order.
    cua_results: list[dict] = field(default_factory=list)

    # Seconds the follow-up window stays open. 0 closes it immediately, which
    # is what a single-turn scenario wants; a multi-turn one needs long enough
    # for the next utterance's chunks to arrive.
    followup_timeout: int = 0


# -- scripted edges --------------------------------------------------------


def _scripted_gemini(replies: list[Reply], overruns: Overruns):
    """Replacement for GemmaClient._stream_gemini_response, serving a queue.

    Patched here rather than at ``chat()`` so the recording tap, the request
    dict, and the digest are all built by production code. A tape generated
    this way is indistinguishable in shape from a recorded one.
    """
    queue = list(replies)
    served = [0]

    async def _stream(self, messages, temperature=0.2, max_tokens=1024):
        if not queue:
            preview = ""
            for message in reversed(messages):
                content = message.get("content")
                if isinstance(content, str) and content:
                    preview = content[-200:]
                    break
            overruns.note(
                f"model call #{served[0] + 1} was made but only {served[0]} are "
                f"scripted; end of prompt: {preview!r}"
            )
            # An empty stream, which is what production sees when a request
            # fails. Raising here would be swallowed - see Overruns.
            return
        served[0] += 1
        reply = queue.pop(0)
        if isinstance(reply, RateLimited):
            from ..llm.gemma_client import RateLimitError

            raise RateLimitError(reply.message)
        for chunk in ([reply] if isinstance(reply, str) else reply):
            yield chunk

    return _stream


class _Segment:
    def __init__(self, text: str):
        self.text = text


class _ScriptedWhisperModel:
    """Stands in for faster-whisper's model, one transcript per call.

    Installed on the real WhisperStreaming so ``_transcribe_pcm`` still runs -
    and still tapes the exact PCM bytes it was handed, which is what makes the
    generated tape's stt records line up with its audio.
    """

    def __init__(self, transcripts: list[str], overruns: Overruns):
        self._queue = list(transcripts)
        self._overruns = overruns

    def transcribe(self, audio, **kwargs):
        if not self._queue:
            # Empty, not an exception: a transcription failure ends the turn
            # through a different branch (main.py:591), which would tape a
            # route the scenario never asked for.
            self._overruns.note(
                "more utterances were transcribed than the scenario has turns"
            )
            return iter([_Segment("")]), None
        return iter([_Segment(self._queue.pop(0))]), None


class ScriptedPerception:
    """Serves hand-written element graphs, and tapes them as the real one does.

    The graphs go out through ``replay._rebuild_snapshot`` and come back in
    through ``perception._snapshot_tape``, which is the same round trip a
    replay performs. Taping the rebuilt snapshot rather than the source dict is
    what makes the tape self-consistent: replaying it reconstructs the object
    the planner actually saw, not a richer one it never did.
    """

    def __init__(self, snapshots: list[dict]):
        from .replay import _rebuild_snapshot

        if not snapshots:
            snapshots = [blank_screen()]
        self._records = list(snapshots)
        self._rebuild = _rebuild_snapshot
        self.observations = 0

    def capture_snapshot(self):
        from ..agent.perception import _snapshot_tape
        from .recorder import get_recorder

        index = min(self.observations, len(self._records) - 1)
        self.observations += 1
        snapshot = self._rebuild(self._records[index])

        recorder = get_recorder()
        if recorder is not None:
            recorder.record_snapshot(_snapshot_tape(snapshot))
        return snapshot

    async def capture_snapshot_async(self):
        return self.capture_snapshot()


class ScriptedPump:
    """A microphone that hands out one utterance per listening window.

    ``drain()`` is the turn boundary. Both listening windows call it before
    collecting - ``main.py:554`` and ``main.py:709`` - so using it to advance
    the script means the pump never has to be told which turn is in progress.

    Chunks are paced, but the VAD no longer depends on that: the harness runs it
    on an audio clock (``harness/clock.py``), so the turn ends on the same chunk
    whatever the delivery rate. The pacing that remains is there to keep the
    recorded chunk offsets resembling a real microphone's, since ``TapePump``
    replays against them and the timing diff is graded against them.

    It used to be load-bearing, and badly so: silence accumulated against
    ``time.time()``, 275ms of it at 50ms per chunk was crossed 5.5 chunks in,
    and a scheduler hiccup either side of that boundary moved the turn-end by a
    chunk. Two of 27 tapes failed per run, never the same two.
    """

    #: Real delay between chunks. No longer a correctness constraint - see the
    #: class docstring - but still what gives the recorded offsets their shape.
    CHUNK_DELAY_S = 0.05

    def __init__(self, turns: int, chunk_bytes: int = 1024, speech: int = 6, silence: int = 20):
        quiet = b"\x00\x00" * (chunk_bytes // 2)
        self._script = []
        for turn in range(turns):
            # A different amplitude per turn, so two turns in one session never
            # produce byte-identical PCM. Real utterances never collide; a
            # generator that emits the same square wave every turn would be
            # exercising a degenerate case the harness has no reason to meet.
            # All of these are far above the 0.008-of-full-scale VAD threshold.
            loud = (6000 + 500 * turn).to_bytes(2, "little", signed=True) * (chunk_bytes // 2)
            self._script.append([loud] * speech + [quiet] * silence)
        self._active: list[bytes] = []

    def start(self, loop=None):
        return self

    def stop(self) -> None:
        return None

    def drain(self) -> int:
        """Load the next turn's audio. Returns the number of chunks dropped."""
        dropped = len(self._active)
        self._active = self._script.pop(0) if self._script else []
        return dropped

    async def get(self, timeout: Optional[float] = None) -> Optional[bytes]:
        from ..audio.pump import _tape

        if not self._active:
            # None, not an exception: this is how the follow-up window learns
            # nothing more is coming, and it is the path a real silent window
            # takes too.
            await asyncio.sleep(self.CHUNK_DELAY_S)
            return None
        await asyncio.sleep(self.CHUNK_DELAY_S)
        # Through the production tap, so the generated tape's audio stream is
        # written by the same code a live session writes it with.
        return _tape(self._active.pop(0))


class ScriptedComputerUse:
    """A ComputerUse that reports ready and performs nothing."""

    def __init__(self, results: list[dict]):
        self._results = list(results)
        self.calls: list[tuple[str, dict]] = []

    is_ready = True

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def perform(self, action: str, params: dict) -> dict:
        self.calls.append((action, params))
        if self._results:
            return self._results.pop(0)
        return {"ok": True}


class _NullFeedbackSounds:
    """The activation chime decodes an MP3 through pydub. Not during generation."""

    @staticmethod
    def play_chime() -> None:
        return None

    @staticmethod
    def play_end() -> None:
        return None


# Every Dispatcher leaf that touches the machine. All of them are replaced for
# every scenario, whether or not the scenario expects to reach them: a script
# that routes somewhere unintended must not be able to lock the workstation or
# move a real file to the recycle bin to prove it.
SIDE_EFFECTING_HANDLERS = (
    "_open_app", "_close_app", "_search_files", "_open_file",
    "_read_pdf", "_summarize_pdf", "_adjust_volume", "_lock_computer",
    "_open_calculator", "_delete_file",
)


def _scripted_handler(tool: str, scripted: dict[str, Any], overruns: Overruns):
    """Replace one Dispatcher leaf, keeping execute/emit/record above it real."""
    configured = scripted.get(tool)
    queue = list(configured) if isinstance(configured, list) else None

    async def _handler(self, params):
        if queue:
            return queue.pop(0)
        if queue is None and configured is not None:
            return configured
        overruns.note(
            f"'{tool}' was dispatched with params {params!r}, which this scenario "
            "does not script - either the routing is not what it assumes, or it "
            "needs a tool_results entry"
        )
        return {"status": "error", "error": f"unscripted tool '{tool}'"}

    return _handler


# -- snapshot construction -------------------------------------------------


def element(
    id: int,
    role: str,
    name: str,
    rect: tuple[int, int, int, int],
    *,
    frame: str = "app",
    value: str = "",
    placeholder: str = "",
    container: str = "",
    focused: bool = False,
    focusable: bool = True,
    enabled: bool = True,
    offscreen: bool = False,
    automation_id: str = "",
    source: str = "uia",
) -> dict:
    """One element in a hand-written graph, in the taped serialisation."""
    left, top, right, bottom = rect
    return {
        "id": id, "role": role, "name": name,
        "rect": [left, top, right, bottom],
        "center": [(left + right) // 2, (top + bottom) // 2],
        "value": value, "placeholder": placeholder, "frame": frame,
        "container": container, "focused": focused, "focusable": focusable,
        "enabled": enabled, "offscreen": offscreen,
        "automation_id": automation_id, "source": source,
    }


def screen(
    title: str,
    elements: list[dict],
    *,
    class_name: str = "ApplicationFrameWindow",
    width: int = 1920,
    height: int = 1080,
    dpi_scale: float = 1.0,
    sources: tuple[str, ...] = ("uia",),
) -> dict:
    """A hand-written snapshot, in the taped serialisation."""
    return {
        "window": {"title": title, "class_name": class_name,
                   "rect": [0, 0, width, height]},
        "width": width, "height": height, "dpi_scale": dpi_scale,
        "elements": elements, "graph_sources": list(sources),
        "ocr_lines": [], "image_width": width, "image_height": height,
    }


def blank_screen() -> dict:
    """The desktop with nothing interactive on it."""
    return screen("Desktop", [], class_name="Progman")


# -- the driver ------------------------------------------------------------


def _config_for(scenario: Scenario):
    """A Config that reaches the cloud path and nothing on the machine."""
    from ..config import Config

    return dataclasses.replace(
        Config(),
        use_cloud_llm=True,
        gemini_api_key="scripted",
        # False, so ui_tars_client is the same scripted client. A separate
        # local client would try to reach llama-server.
        use_ui_tars_local=False,
        # Deliberately not a multiple of ScriptedPump.CHUNK_DELAY_S - see the
        # note there. Production's own value is larger; what a generated tape
        # grades is the routing above the VAD, not the VAD's tuning.
        whisper_silence_duration_ms=275,
        followup_timeout_seconds=scenario.followup_timeout,
    )


#: Backstop on the agent loop, applied to the AgentLoop instance rather than to
#: Config - ``AgentLoop`` reads its limit through ``_config_int``, which imports
#: ``grace.config.Config`` directly and so never sees a patched one.
#:
#: This is NOT the shipped default, which is 0 (unlimited). See ``Overruns``
#: for why an unlimited loop turns a short script into a hang. Generous enough
#: that no scenario in the catalogue is shaped by it - the longest runs four
#: planner calls.
GENERATION_STEP_CAP = 12


async def _drive(
    scenario: Scenario,
    computer_use: ScriptedComputerUse,
    overruns: Overruns,
) -> None:
    """Run one scenario through the real activation path."""
    from ..tools.dispatcher import Dispatcher
    from .replay import CapturingWsServer, NullAudio, NullTts, NullWakeWord

    config = _config_for(scenario)
    replies = [reply for turn in scenario.turns for reply in turn.replies]

    with ExitStack() as stack:
        for target, replacement in (
            ("grace.main.Config", lambda: config),
            ("grace.main.AudioCapture", NullAudio),
            ("grace.main.WakeWordDetector", NullWakeWord),
            ("grace.main.KokoroEngine", NullTts),
            ("grace.main.TTSPlayer", NullTts),
            ("grace.main.ComputerUse", lambda *a, **k: computer_use),
            ("grace.main.FeedbackSounds", _NullFeedbackSounds),
            ("grace.main.WsEventServer", lambda *a, **k: CapturingWsServer()),
            ("grace.llm.gemma_client.GemmaClient._stream_gemini_response",
             _scripted_gemini(replies, overruns)),
        ):
            stack.enter_context(mock.patch(target, replacement))

        for name in SIDE_EFFECTING_HANDLERS:
            stack.enter_context(mock.patch.object(
                Dispatcher, name,
                _scripted_handler(name[1:], scenario.tool_results, overruns),
            ))

        # No importlib.reload: reloading grace.main re-executes its body and
        # rebinds these names back to the real classes, undoing every patch
        # above and opening a real microphone.
        import grace.main as main_mod

        app = main_mod.GraceApp()
        app._running = True
        # Silence is counted in audio, not wall time, so a turn ends on the same
        # chunk however loaded the machine is. See harness/clock.py.
        install_audio_clock_vad(app)
        app.pump = ScriptedPump(turns=len(scenario.turns))
        app.agent_loop._perception = ScriptedPerception(scenario.snapshots)
        app.agent_loop._max_iterations = GENERATION_STEP_CAP
        app.whisper._model = _ScriptedWhisperModel(
            [t.transcript for t in scenario.turns], overruns
        )
        app.whisper._initialized = True

        await app._handle_activation()


def generate(scenario: Scenario, out_root: str, *, overwrite: bool = True) -> str:
    """Generate one scenario's tape. Returns the tape directory."""
    from .recorder import Recorder, install_recorder

    directory = os.path.join(out_root, scenario.name)
    if overwrite and os.path.isdir(directory):
        # A regenerated tape replaces its predecessor. Appending to the JSONL
        # streams of an old run would produce a tape that never happened.
        shutil.rmtree(directory)

    recorder = Recorder(out_root, session_id=scenario.name)
    install_recorder(recorder)
    computer_use = ScriptedComputerUse(scenario.cua_results)
    overruns = Overruns()
    try:
        recorder.write_meta(
            _config_for(scenario),
            extra={
                "synthetic": True,
                "scenario": scenario.name,
                "covers": scenario.covers,
                "generator": "grace.harness.generate",
            },
        )
        asyncio.run(_drive(scenario, computer_use, overruns))
    finally:
        recorder.close()
        install_recorder(None)

    # After the session, not during it: the backend swallows exceptions raised
    # from a stubbed edge. The tape on disk is left in place so the wrong route
    # it took can be read.
    overruns.raise_if_any(scenario.name)
    return recorder.dir


# -- the scenario catalogue ------------------------------------------------


def _intent(tool: str, **params) -> list[str]:
    """An intent reply, split across chunks the way a stream delivers one."""
    import json

    body = json.dumps({"tool": tool, "params": params}, ensure_ascii=False)
    return [body[:12], body[12:40], body[40:]] if len(body) > 40 else [body]


def _plan(action: str, *, thought: str, expect: str = "", user_update: str = "",
          completed: bool = False, final: str = "", **params) -> list[str]:
    """A planner reply, split across chunks."""
    import json

    body = json.dumps({
        "thought": thought, "action": action, "params": params,
        "expect": expect, "is_completed": completed,
        "user_update": user_update or f"Running {action}…",
        "final_response": final,
    }, ensure_ascii=False)
    return [body[:30], body[30:90], body[90:]] if len(body) > 90 else [body]


_NOTEPAD = screen(
    "Untitled - Notepad",
    [
        element(0, "document", "Text Editor", (8, 60, 1912, 1040),
                container="Notepad", focused=True),
        element(1, "menuitem", "File", (8, 30, 60, 58), container="Menu"),
        element(2, "menuitem", "Edit", (60, 30, 112, 58), container="Menu"),
    ],
    class_name="Notepad",
)

_BROWSER = screen(
    "Grace test page - Edge",
    [
        element(0, "edit", "Address and search bar", (120, 60, 1700, 96),
                frame="chrome", container="Toolbar", placeholder="Search or enter web address"),
        element(1, "button", "Refresh", (80, 60, 116, 96), frame="chrome",
                container="Toolbar"),
        element(2, "edit", "Search", (700, 300, 1220, 348), frame="page",
                container="Main", placeholder="Search this site"),
        element(3, "button", "Submit", (1230, 300, 1330, 348), frame="page",
                container="Main"),
    ],
    class_name="Chrome_WidgetWin_1",
)


def _catalogue() -> list[Scenario]:
    return [
        # -- conversation --------------------------------------------------
        Scenario(
            name="conversation_plain",
            covers="A question that reaches no tool. CONVERSATION route.",
            turns=[Turn(
                "hello grace how are you today",
                [_intent("converse", response="I'm doing well, thank you. What can I help with?")],
            )],
        ),

        # -- fast path, one scenario per tool ------------------------------
        Scenario(
            name="fastpath_open_app",
            covers="open_app via FAST_PATH. Tool label carries the app name.",
            turns=[Turn("open notepad", [_intent("open_app", name="notepad")])],
            tool_results={"open_app": {"status": "ok", "text": "I've opened Notepad."}},
        ),
        Scenario(
            name="fastpath_close_app",
            covers=(
                "close_app via FAST_PATH, asked about first. This scenario used "
                "to record the opposite: the guard lived inside AgentLoop, so a "
                "close_app the intent model resolved in one shot ran "
                "unconfirmed, and the tape pinned that as intended behaviour. "
                "The guard now sits on the dispatch boundary, which both paths "
                "cross, so the question is asked and the answer resumes it."
            ),
            turns=[
                Turn("close notepad", [_intent("close_app", name="notepad")]),
                Turn("yes", []),
            ],
            tool_results={"close_app": {"status": "ok", "text": "I've closed notepad."}},
            followup_timeout=6,
        ),
        Scenario(
            name="fastpath_search_files",
            covers="search_files via FAST_PATH, with results spoken back.",
            turns=[Turn("find my tax return", [_intent("search_files", query="tax return")])],
            tool_results={"search_files": {
                "status": "ok",
                "text": "Here are the matching files:\n  1. C:\\Users\\x\\Documents\\tax return 2025.pdf",
                "files": ["C:\\Users\\x\\Documents\\tax return 2025.pdf"],
            }},
        ),
        Scenario(
            name="fastpath_open_file",
            covers="open_file via FAST_PATH.",
            turns=[Turn("open the budget spreadsheet",
                        [_intent("open_file", name="budget.xlsx")])],
            tool_results={"open_file": {"status": "ok", "text": "I've opened budget.xlsx."}},
        ),
        Scenario(
            name="fastpath_adjust_volume",
            covers="adjust_volume via FAST_PATH.",
            turns=[Turn("turn the volume up a bit",
                        [_intent("adjust_volume", amount=10, mode="increase")])],
            tool_results={"adjust_volume": {"status": "ok", "text": "Volume set to 60 percent.",
                                            "volume": 60}},
        ),
        Scenario(
            name="fastpath_lock_computer",
            covers="lock_computer via FAST_PATH - confirmed, then run.",
            turns=[
                Turn("lock my computer", [_intent("lock_computer")]),
                Turn("yes", []),
            ],
            tool_results={"lock_computer": {"status": "ok", "text": "I've locked your computer."}},
            followup_timeout=6,
        ),
        Scenario(
            name="fastpath_open_calculator",
            covers="open_calculator via FAST_PATH.",
            turns=[Turn("open the calculator", [_intent("open_calculator")])],
            tool_results={"open_calculator": {"status": "ok", "text": "I've opened the calculator."}},
        ),
        Scenario(
            name="fastpath_delete_file",
            covers=(
                "delete_file via FAST_PATH. The one that mattered most: this "
                "used to reach the recycle bin without anyone being asked, "
                "because the request was simple enough to route directly."
            ),
            turns=[
                Turn("delete old notes", [_intent("delete_file", name="old notes.txt")]),
                Turn("yes", []),
            ],
            tool_results={"delete_file": {
                "status": "ok", "action": "delete_file",
                "text": "I've moved 'old notes.txt' to the Recycle Bin.",
            }},
            followup_timeout=6,
        ),
        Scenario(
            name="fastpath_cua_list_windows",
            covers="cua_list_windows via FAST_PATH - the CUA branch of the dispatcher.",
            turns=[Turn("list my open windows", [_intent("cua_list_windows")])],
            cua_results=[{"ok": True, "windows": [{"title": "Untitled - Notepad"},
                                                  {"title": "Grace test page - Edge"}]}],
        ),

        # -- tool dispatch failure ------------------------------------------
        Scenario(
            name="fastpath_tool_error",
            covers=(
                "A fast-path tool that fails. The error text is what gets "
                "spoken, and ToolExecutionFinished must still be emitted - a "
                "port that leaks the exception leaves the pill spinning."
            ),
            turns=[Turn("open the budget spreadsheet",
                        [_intent("open_file", name="budget.xlsx")])],
            tool_results={"open_file": {
                "status": "error", "error": "File not found",
                "text": "I couldn't open budget.xlsx. File not found",
            }},
        ),

        # -- agentic --------------------------------------------------------
        Scenario(
            name="agentic_two_step",
            covers=(
                "An agentic goal: click a field, type into it, finish. Pins "
                "the ToolExecutionStarted{tool, step} shape the agent loop "
                "emits, which the dispatcher's own {label} form does not."
            ),
            turns=[Turn(
                "click the search box and type hello",
                [
                    _intent("cua_click", element_id=2),
                    _plan("cua_click", thought="Focus the site's own search box.",
                          expect="the search box has keyboard focus",
                          user_update="Clicking the search box…", element_id=2),
                    _plan("cua_type_text", thought="Type the query.",
                          expect="the search box contains hello",
                          user_update="Typing…", text="hello"),
                    _plan("converse", thought="Done.", completed=True,
                          user_update="Finishing up…",
                          final="I've typed hello into the search box."),
                ],
            )],
            snapshots=[_BROWSER],
            cua_results=[{"ok": True, "message": "clicked"}, {"ok": True, "message": "typed"}],
        ),
        Scenario(
            name="agentic_step_failed",
            covers=(
                "A step whose tool reports failure. Exercises "
                "_expectation_note's IT FAILED branch - the verification "
                "signal that lets the loop correct instead of retrying blindly."
            ),
            turns=[Turn(
                "click the submit button",
                [
                    _intent("cua_click", element_id=3),
                    _plan("cua_click", thought="Click submit.",
                          expect="the form submits", user_update="Clicking…", element_id=3),
                    _plan("cua_click", thought="That failed; try the search box first.",
                          expect="the search box has focus", user_update="Trying another way…",
                          element_id=2),
                    _plan("converse", thought="Done.", completed=True,
                          user_update="Finishing up…", final="I've submitted the form."),
                ],
            )],
            snapshots=[_BROWSER],
            cua_results=[{"ok": False, "error": "element is not enabled"},
                         {"ok": True, "message": "clicked"}],
        ),
        Scenario(
            name="agentic_completion_rejected",
            covers=(
                "The planner claims completion before touching anything. "
                "_verify_goal_completion rejects it, the hint goes to the "
                "scratchpad, and the loop is forced to actually act."
            ),
            turns=[Turn(
                "open the file menu in notepad",
                [
                    _intent("cua_click", element_id=1),
                    _plan("converse", thought="I think that's already done.",
                          completed=True, user_update="Finishing up…",
                          final="The File menu is open."),
                    _plan("cua_click", thought="Nothing has been done yet; open it.",
                          expect="the File menu is expanded",
                          user_update="Opening the File menu…", element_id=1),
                    _plan("converse", thought="Done.", completed=True,
                          user_update="Finishing up…", final="I've opened the File menu."),
                ],
            )],
            snapshots=[_NOTEPAD],
            cua_results=[{"ok": True, "message": "clicked"}],
        ),
        Scenario(
            name="agentic_unparseable_plan",
            covers=(
                "The planner returns prose instead of JSON. It must cost a "
                "step and set last_error, not abort the goal."
            ),
            turns=[Turn(
                "click the refresh button",
                [
                    _intent("cua_click", element_id=1),
                    "I think you should probably click the refresh button now.",
                    _plan("cua_click", thought="Refresh the page.",
                          expect="the page reloads", user_update="Refreshing…", element_id=1),
                    _plan("converse", thought="Done.", completed=True,
                          user_update="Finishing up…", final="I've refreshed the page."),
                ],
            )],
            snapshots=[_BROWSER],
            cua_results=[{"ok": True, "message": "clicked"}],
        ),
        Scenario(
            name="agentic_preexec_open_app",
            covers=(
                "An agentic goal whose intent is open_app: main.py pre-executes "
                "the launch before the loop starts, so the tape holds a dispatch "
                "that no planner step produced."
            ),
            turns=[Turn(
                "open notepad then type a note",
                [
                    _intent("open_app", name="notepad"),
                    _plan("cua_type_text", thought="Type the note.",
                          expect="the document contains text",
                          user_update="Typing…", text="a note"),
                    _plan("converse", thought="Done.", completed=True,
                          user_update="Finishing up…", final="I've typed the note."),
                ],
            )],
            snapshots=[_NOTEPAD],
            tool_results={"open_app": {"status": "ok", "text": "I've opened Notepad."}},
            cua_results=[{"ok": True, "message": "typed"}],
        ),
        Scenario(
            name="agentic_read_pdf",
            covers="read_pdf is an AGENTIC tool, so it routes through the loop.",
            turns=[Turn(
                "read my report pdf to me",
                [
                    _intent("read_pdf", path="C:\\Users\\x\\Documents\\report.pdf"),
                    _plan("read_pdf", thought="Extract the text.",
                          expect="the document text is available",
                          user_update="Reading document…",
                          path="C:\\Users\\x\\Documents\\report.pdf"),
                    _plan("converse", thought="Report what it says.", completed=True,
                          user_update="Finishing up…",
                          final="The report covers third quarter results."),
                ],
            )],
            snapshots=[_NOTEPAD],
            tool_results={"read_pdf": {
                "status": "ok", "action": "read_pdf",
                "text": "[Excerpt 1]: Third quarter results exceeded projections.",
            }},
        ),

        # -- safety ---------------------------------------------------------
        Scenario(
            name="safety_confirm_accepted",
            covers=(
                "SafetyGuard parks delete_file, the user says yes in the "
                "follow-up window, and _resolve_pending_confirmation resumes "
                "the parked step. A regression here means a confirmed deletion "
                "silently never happens."
            ),
            turns=[
                Turn(
                    "get rid of the draft file",
                    [
                        _intent("cua_click", element_id=0),
                        _plan("delete_file", thought="Remove the draft.",
                              expect="the file is in the recycle bin",
                              user_update="Deleting…", name="draft.txt"),
                    ],
                ),
                Turn(
                    "yes go ahead",
                    [_plan("converse", thought="Done.", completed=True,
                           user_update="Finishing up…",
                           final="I've moved draft.txt to the Recycle Bin.")],
                ),
            ],
            snapshots=[_NOTEPAD],
            tool_results={"delete_file": {
                "status": "ok", "action": "delete_file",
                "text": "I've moved 'draft.txt' to the Recycle Bin.",
            }},
            followup_timeout=6,
        ),
        Scenario(
            name="safety_confirm_declined",
            covers=(
                "The same park, declined. resume_pending(False) must finish "
                "the goal without dispatching anything - the tape's dispatch "
                "stream being empty is the assertion."
            ),
            turns=[
                Turn(
                    "get rid of the draft file",
                    [
                        _intent("cua_click", element_id=0),
                        _plan("delete_file", thought="Remove the draft.",
                              expect="the file is in the recycle bin",
                              user_update="Deleting…", name="draft.txt"),
                    ],
                ),
                Turn("no don't do that", []),
            ],
            snapshots=[_NOTEPAD],
            followup_timeout=6,
        ),
        Scenario(
            name="safety_confirm_hijacked",
            covers=(
                "A parked confirmation answered with an unrelated request. "
                "_confirmation_answer returns None, the pending step is "
                "cancelled, and the new request is handled instead. The "
                "failure mode this guards is the worst one in the system: "
                "treating an unrelated utterance as consent to delete."
            ),
            turns=[
                Turn(
                    "get rid of the draft file",
                    [
                        _intent("cua_click", element_id=0),
                        _plan("delete_file", thought="Remove the draft.",
                              expect="the file is in the recycle bin",
                              user_update="Deleting…", name="draft.txt"),
                    ],
                ),
                Turn(
                    "what can you do",
                    [_intent("converse", response="I can open apps, find files, and control your PC by voice.")],
                ),
            ],
            snapshots=[_NOTEPAD],
            followup_timeout=6,
        ),
        Scenario(
            name="safety_press_key_confirm",
            covers=(
                "SafetyGuard's parameter check: a normalised Alt+F4 is parked "
                "even though cua_press_key is not itself a guarded tool."
            ),
            turns=[
                Turn(
                    "press alt f4",
                    [
                        _intent("cua_press_key", key="Alt+F4"),
                        _plan("cua_press_key", thought="Close the window.",
                              expect="the window closes", user_update="Pressing key…",
                              key="Alt+F4"),
                    ],
                ),
                Turn("no cancel that", []),
            ],
            snapshots=[_NOTEPAD],
            followup_timeout=6,
        ),

        # -- degraded paths -------------------------------------------------
        Scenario(
            name="rate_limited_intent",
            covers=(
                "The intent call is rate limited. The turn apologises and ends "
                "without routing anywhere."
            ),
            turns=[Turn("open notepad", [RateLimited()])],
        ),
        Scenario(
            name="rate_limited_planner",
            covers="The planner is rate limited mid-goal; the loop reports it.",
            turns=[Turn(
                "click the search box",
                [_intent("cua_click", element_id=2), RateLimited()],
            )],
            snapshots=[_BROWSER],
        ),
        Scenario(
            name="empty_transcript",
            covers=(
                "The user activated Grace and said nothing. No intent call is "
                "made at all, and the turn ends through the silent branch."
            ),
            turns=[Turn("", [])],
        ),
        Scenario(
            name="unparseable_intent",
            covers=(
                "The intent model returns something the parser rejects. The "
                "router then decides from the transcript alone."
            ),
            turns=[Turn(
                "hello there",
                ["not json at all", _plan("converse", thought="Greet.", completed=True,
                                          user_update="Finishing up…",
                                          final="Hello. What can I do for you?")],
            )],
            snapshots=[blank_screen()],
        ),

        # -- follow-up ------------------------------------------------------
        Scenario(
            name="followup_chain_three_turns",
            covers=(
                "Wake word plus two follow-ups without a second wake word. "
                "The window restarts after each handled turn - it used to "
                "recurse, one stack frame per exchange."
            ),
            turns=[
                Turn("open notepad", [_intent("open_app", name="notepad")]),
                Turn("now turn the volume up",
                     [_intent("adjust_volume", amount=10, mode="increase")]),
                Turn("thanks that's all",
                     [_intent("converse", response="Happy to help. Just say Grace if you need me.")]),
            ],
            tool_results={
                "open_app": {"status": "ok", "text": "I've opened Notepad."},
                "adjust_volume": {"status": "ok", "text": "Volume set to 60 percent.", "volume": 60},
            },
            followup_timeout=6,
        ),
        Scenario(
            name="followup_agentic",
            covers="A follow-up utterance that routes to the agentic loop.",
            turns=[
                Turn("hello grace", [_intent("converse", response="Hello. What can I do?")]),
                Turn(
                    "click the refresh button",
                    [
                        _intent("cua_click", element_id=1),
                        _plan("cua_click", thought="Refresh.", expect="the page reloads",
                              user_update="Refreshing…", element_id=1),
                        _plan("converse", thought="Done.", completed=True,
                              user_update="Finishing up…", final="I've refreshed the page."),
                    ],
                ),
            ],
            snapshots=[_BROWSER],
            cua_results=[{"ok": True, "message": "clicked"}],
            followup_timeout=6,
        ),
    ]


SCENARIOS: list[Scenario] = _catalogue()


# -- CLI -------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="corpus", help="where to write the tapes")
    parser.add_argument("--only", action="append", default=[],
                        help="generate just these scenarios (repeatable)")
    parser.add_argument("--list", action="store_true", help="list the catalogue and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.list:
        for scenario in SCENARIOS:
            print(f"{scenario.name}\n    {scenario.covers}\n")
        return 0

    selected = SCENARIOS
    if args.only:
        by_name = {s.name: s for s in SCENARIOS}
        unknown = [name for name in args.only if name not in by_name]
        if unknown:
            print(f"unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        selected = [by_name[name] for name in args.only]

    os.makedirs(args.out, exist_ok=True)
    failures = []
    for scenario in selected:
        try:
            directory = generate(scenario, args.out)
        except Exception as exc:
            failures.append((scenario.name, exc))
            print(f"FAIL {scenario.name}: {type(exc).__name__}: {exc}")
            continue
        print(f"ok   {scenario.name} -> {directory}")

    print(f"\n{len(selected) - len(failures)}/{len(selected)} tapes generated")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
