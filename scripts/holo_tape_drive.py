"""Drive corpus tapes end to end with the real Holo 3.1-4B llama-server as the planner.

The replay harness pins the LLM edge to the recording (``replay.TapeLlm``).
This driver leaves that edge to a live llama-server instead: the scenario
(audio, transcript, element graphs, tool results) comes from the tape, every
desktop, mic and speaker call stays mocked exactly as ``replay`` and
``generate`` mock them, and every LLM decision - intent, planner, responder -
is the live model over the production ``GemmaClient`` SSE path.

What is NOT graded here is event parity with the recording; the planner is a
different brain and is free to route differently. What is reported is the
plain facts: per-step dispatches with their served results, the agent result,
and the raw model output behind each decision call, so failures can be read
without guessing at causes.

Pass shaping, in one llama-server session per command:

* Pass 1 (default) is all-live: the model decides the intent route as well.
* If pass 1 never reached an agent goal (the intent route went somewhere the
  live tape never went), pass 2 repeats the drive with the tape's recorded
  intent reply served back verbatim, so the *planner* decision - the thing the
  task names - still gets exercised against the live model. The pass is
  labelled in the report.

Usage::

    python scripts/holo_tape_drive.py \
        --tapes corpus/agentic_two_step corpus/agentic_read_pdf

The server is started for the whole run and stopped in ``finally`` - one
llama-server, one foreground command. ``--server-url`` points at an already
running llama-server instead of starting one.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import ExitStack
from typing import Any, Optional
from unittest import mock

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(repo_root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(repo_root, "src"))

from grace.harness.replay import (  # noqa: E402
    CapturingWsServer,
    NullAudio,
    NullTts,
    NullWakeWord,
    TapeExhausted,
    _config_from,
    _rebuild_snapshot,
)
from grace.harness.tape import Tape  # noqa: E402

# Generous hard cap for a live planner on short tapes. The tapes themselves
# took 2-3 planner calls; 12 is deliberately far above anything a healthy run
# needs, so it only stops a runaway.
GENERATION_STEP_CAP = 12


class _NullFeedbackSounds:
    """Silences the chime. Not an edge under test - the tape has no audio out."""

    @staticmethod
    def play_chime(*args, **kwargs): return None
    @staticmethod
    def play_end(*args, **kwargs): return None
    @staticmethod
    def play_success(*args, **kwargs): return None
    @staticmethod
    def play_cancel(*args, **kwargs): return None
    @staticmethod
    def play_error(*args, **kwargs): return None
    @staticmethod
    def play_listening(*args, **kwargs): return None
    @staticmethod
    def play_ack(*args, **kwargs): return None


class RepeatingSnapshots:
    """Serves the tape's recorded element graphs; the last one repeats.

    ``replay.TapePerception`` raises once the recording runs out. A live
    planner legitimately observes more often than the recorded one did (the
    verification path re-observes the same screen), so exhaustion would end a
    healthy run at a fiction. The recorded graphs are what the taped session
    observed; planning against the same view again is honest.
    """

    def __init__(self, tape: Tape):
        from grace.harness.generate import blank_screen

        records = [r["snapshot"] for r in tape.snapshots]
        self._records = records if records else [blank_screen()]
        self._index = -1

    def capture_snapshot(self):
        self._index = min(self._index + 1, len(self._records) - 1)
        return _rebuild_snapshot(self._records[self._index])

    async def capture_snapshot_async(self):
        return self.capture_snapshot()


class ToolKeyedDispatcher:
    """Serves the tape's recorded results to the tool that was dispatched.

    Replay serves results positionally, which is right when it asserts the
    same calls in the same order. A live planner may legally pick a different
    tool first, and positional serving would then hand a ``read_pdf`` excerpt
    to a ``search_files`` dispatch with nothing in the report saying so. This
    stand-in keeps the same guarantee (cuts nothing, asserts the call, runs
    the real emitters and the real safety gate) and answers a dispatch first
    with that tool's own recorded result, then - only if none was recorded -
    with any recorded result left, labelled as coming from a different tool.
    """

    def __init__(self, tape: Tape, inner):
        self._raw_records = [
            {"tool": r.get("tool"), "result": r["result"]} for r in tape.dispatch
        ]
        self._inner = inner
        self.calls: list[dict] = []
        self.last_source_tool: Optional[str] = None

    async def execute(self, intent, confirmed: bool = False):
        from grace.tools.dispatcher import Dispatcher

        refusal = Dispatcher.confirmation_required(intent, confirmed)
        if refusal is not None:
            return refusal

        self.calls.append({"tool": intent.tool, "params": intent.params})

        # Events emitted around a dispatch are part of the contract, so the
        # real emitters still run; only the side effect is suppressed.
        if not intent.tool.startswith("cua_"):
            await self._inner._emit_tool_started(intent.tool, intent.params)
        else:
            await self._inner._emit_cua_started(intent.tool[4:])
        await self._inner._emit_tool_finished()

        served, source = self._pop_result(intent.tool)
        if served is None:
            raise TapeExhausted(
                f"the backend dispatched {intent.tool}({intent.params}) but the "
                f"tape records no dispatch result to serve"
            )
        self.last_source_tool = source
        return served

    def _pop_result(self, tool: str):
        for i, record in enumerate(self._raw_records):
            if record["tool"] == tool:
                self._raw_records.pop(i)
                return record["result"], tool
        if self._raw_records:
            record = self._raw_records.pop(0)
            return record["result"], record["tool"]
        return None, None


def wait_for_server(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def _drive_config(tape: Tape, args) -> Any:
    """The tape's recorded config, pointed at the live server, single-model preset.

    ``gpu-single`` is the buildathon preset this drive reproduces: Holo 3.1-4B
    is the planner, and the element tree reaches it directly (no fleet
    decision ladder), matching how the corpus was recorded.
    """
    base = _config_from(tape)
    return dataclasses.replace(
        base,
        use_cloud_llm=False,
        gemini_api_key="",
        preset="gpu-single",
        llama_host=args.host,
        llama_port=args.port,
    )


def _install_llm_recorder(stack: ExitStack, records: list[dict]) -> None:
    """Keep the real client (live server), and log every raw model reply."""
    from grace.llm.gemma_client import GemmaClient

    async def _capture_text(self, prompt, system_prompt, temperature=0.2,
                            max_tokens=8192, messages=None, image_b64=None,
                            model=None):
        raw = await self._original_generate_text(
            prompt, system_prompt, temperature=temperature,
            max_tokens=max_tokens, messages=messages, image_b64=image_b64,
            model=model,
        )
        records.append({
            "call": "generate_text",
            "image": bool(image_b64),
            "prompt_tail": (prompt or "")[-240:],
            "raw": raw,
        })
        return raw

    async def _capture_intent(self, user_message, system_prompt):
        raw = await self._original_generate_intent(user_message, system_prompt)
        records.append({
            "call": "generate_intent",
            "image": False,
            "prompt_tail": ((system_prompt or "")[-80:] + " || "
                            + (user_message or "")[-240:]),
            "raw": raw,
        })
        return raw

    stack.callback(lambda: delattr(GemmaClient, "_original_generate_text"))
    stack.callback(lambda: delattr(GemmaClient, "_original_generate_intent"))
    GemmaClient._original_generate_text = GemmaClient.generate_text
    GemmaClient._original_generate_intent = GemmaClient.generate_intent
    stack.enter_context(mock.patch.object(GemmaClient, "generate_text", _capture_text))
    stack.enter_context(mock.patch.object(GemmaClient, "generate_intent", _capture_intent))

    return stack


def _install_intent_from_tape(stack: ExitStack, tape: Tape, records: list[dict]) -> None:
    """Serve the tape's recorded intent reply instead of a live one.

    Used only on a second pass, when the live intent route never reached an
    agent goal: it keeps the *planner* decision live (the thing this drive
    exists to exercise) while the routing into the loop comes from the tape.
    The served reply is logged into ``records`` like any other model output.
    """
    from grace.llm.gemma_client import GemmaClient

    first = tape.llm[0] if tape.llm else {}
    recorded = first.get("response") or "".join(first.get("chunks") or [])
    if not recorded:
        raise RuntimeError(f"{tape.dir} holds no recorded LLM reply to serve")

    async def _served_intent(self, user_message, system_prompt):
        records.append({
            "call": "generate_intent (served from tape)",
            "image": False,
            "prompt_tail": ((system_prompt or "")[-80:] + " || "
                            + (user_message or "")[-240:]),
            "raw": recorded,
        })
        return recorded

    stack.enter_context(mock.patch.object(GemmaClient, "generate_intent", _served_intent))


async def drive_tape(tape_dir: str, args, tmp_enrollment: str, intent_from_tape: bool) -> dict:
    """Run one tape through the real activation path with a live planner."""
    import grace.main as main_mod
    from grace.agent.memory import PersistentMemoryStore as RealMemoryStore
    from grace.enrollment.storage import EnrollmentStorage as RealEnrollmentStorage
    from grace.harness.replay import TapePump, TapeWhisper

    tape = Tape.load(tape_dir)
    config = _drive_config(tape, args)
    records: list[dict] = []
    notes: list[str] = []
    executed: list[dict] = []
    captured: dict = {}

    with ExitStack() as stack:
        stack.enter_context(mock.patch("grace.main.Config", lambda: config))
        stack.enter_context(mock.patch("grace.main.AudioCapture", NullAudio))
        stack.enter_context(mock.patch("grace.main.WakeWordDetector", lambda *a, **k: NullWakeWord()))
        stack.enter_context(mock.patch("grace.main.KokoroEngine", lambda *a, **k: NullTts()))
        stack.enter_context(mock.patch("grace.main.TTSPlayer", lambda *a, **k: NullTts()))
        ws = CapturingWsServer()
        stack.enter_context(mock.patch("grace.main.WsEventServer", lambda *a, **k: ws))
        stack.enter_context(mock.patch("grace.main.ComputerUse", lambda *a, **k: mock.MagicMock()))
        stack.enter_context(mock.patch("grace.main.FeedbackSounds", _NullFeedbackSounds))
        stack.enter_context(mock.patch(
            "grace.main.WhisperStreaming", lambda *a, **k: TapeWhisper(tape)))
        stack.enter_context(mock.patch(
            "grace.agent.memory.PersistentMemoryStore",
            lambda *a, **k: RealMemoryStore(db_path=":memory:")))
        stack.enter_context(mock.patch(
            "grace.enrollment.storage.EnrollmentStorage",
            lambda *a, **k: RealEnrollmentStorage(base_dir=tmp_enrollment)))
        stack.enter_context(mock.patch.object(
            main_mod.GraceApp, "refresh_word_biasing", lambda self: []))
        _install_llm_recorder(stack, records)
        if intent_from_tape:
            _install_intent_from_tape(stack, tape, records)
            notes.append("pass 2: the tape's recorded intent reply was served; "
                         "the planner itself stayed live")

        app = main_mod.GraceApp()
        app._running = True
        app._loop = asyncio.get_running_loop()

        from grace.harness.clock import install_audio_clock_vad
        install_audio_clock_vad(app)
        pump = TapePump(tape)
        app.pump = pump
        app.agent_loop._perception = RepeatingSnapshots(tape)
        app.agent_loop._max_iterations = GENERATION_STEP_CAP

        dispatcher = ToolKeyedDispatcher(tape, app.dispatcher)
        app.dispatcher = dispatcher
        app.agent_loop._dispatcher = dispatcher

        orig_execute = dispatcher.execute

        async def _record_execute(intent, confirmed: bool = False):
            dispatcher.last_source_tool = None
            result = await orig_execute(intent, confirmed=confirmed)
            entry = {
                "tool": intent.tool,
                "params": {k: v for k, v in intent.params.items()
                           if not k.startswith("_")},
                "confirmed": confirmed,
                "served_from": getattr(dispatcher, "last_source_tool", None),
                "status": result.get("status"),
                "text": result.get("text"),
            }
            executed.append(entry)
            return result

        dispatcher.execute = _record_execute

        orig_goal_runner = app._run_agent_goal

        async def _capture_goal(transcript: str):
            result = await orig_goal_runner(transcript)
            captured["agent_result"] = result
            return result

        app._run_agent_goal = _capture_goal

        try:
            await app._handle_activation()
        except TapeExhausted as exc:
            notes.append(f"tape exhausted: {exc}")
            captured["tape_exhausted"] = True
        except Exception as exc:  # keep the facts, fail the report honestly.
            notes.append(f"activation raised {type(exc).__name__}: {exc}")
        finally:
            await app.gemma.close()

    if pump.ran_dry:
        notes.append("recorded audio ran out before the VAD closed the turn")

    agent_result = captured.get("agent_result")
    event_types = [e.get("type") for e in ws.events]
    dispatch_failures = [d for d in executed if d["status"] == "error"]
    goal_status = (agent_result or {}).get("status")
    goal_ok = goal_status == "ok"

    return {
        "tape": os.path.basename(os.path.normpath(tape_dir)),
        "pass": "intent_from_tape" if intent_from_tape else "all_live",
        "goal_status": goal_status,
        "ok": goal_ok,
        "final_response": (agent_result or {}).get("final_response"),
        "steps": [
            {
                "n": i + 1,
                "action": s.get("action"),
                "params": {k: v for k, v in (s.get("params") or {}).items()
                           if not k.startswith("_")},
                "thought": s.get("thought"),
                "expect": s.get("expect"),
                "is_completed": s.get("is_completed"),
                "verdict": _step_verdict(s),
            }
            for i, s in enumerate((agent_result or {}).get("steps") or [])
        ],
        "dispatches": executed,
        "dispatch_failures": dispatch_failures,
        "event_types": event_types,
        "llm_records": records,
        "notes": notes,
    }


def _step_verdict(step: dict) -> str:
    """PASS/FAIL for one recorded step, from facts only.

    A step recorded with an empty thought under action ``converse`` is the
    loop's placeholder for a plan that never parsed. Everything else is
    reported; the dispatch results below say what the tool actually returned.
    """
    if step.get("action") == "converse" and not (step.get("thought") or "").strip():
        return "FAIL (unparseable plan)"
    return "PASS"


def start_server(args) -> subprocess.Popen:
    cmd = [
        args.server,
        "-m", args.model,
        "--mmproj", args.mmproj,
        "-ngl", str(args.ngl),
        "-c", "8192",
        "--port", str(args.port),
        "--host", args.host,
    ]
    server_dir = os.path.dirname(os.path.abspath(args.server))
    print(f"Starting llama-server on port {args.port}...")
    return subprocess.Popen(cmd, cwd=server_dir,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes", nargs="+", required=True,
                        help="tape directories to drive (e.g. corpus/agentic_two_step)")
    parser.add_argument("--model", default=os.path.join(
        os.path.expanduser("~"), "Downloads", "Holo-3.1-4B.Q4_K_M.gguf"))
    parser.add_argument("--mmproj", default=os.path.join(
        os.path.expanduser("~"), "Downloads", "Holo-3.1-4B.mmproj-q8_0.gguf"))
    parser.add_argument("--server", default=os.path.join(repo_root, "llama cpp", "llama-server.exe"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0,
                        help="port for the stack (0 = pick a free one)")
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--server-url", default=None,
                        help="an already-running llama-server to use instead of starting one")
    parser.add_argument("--json", default=None, help="write the report as JSON here")
    args = parser.parse_args(argv)

    if not args.port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            args.port = s.getsockname()[1]
    url = args.server_url or f"http://{args.host}:{args.port}"

    proc: Optional[subprocess.Popen] = None
    pre_existing = _llama_pids()
    exit_code = 0
    temp_enrollment = os.path.join(tempfile.gettempdir(), "grace-holo-drive", "enrollment")
    os.makedirs(temp_enrollment, exist_ok=True)

    try:
        if not args.server_url:
            for label, path in (("model", args.model), ("mmproj", args.mmproj),
                                ("llama-server", args.server)):
                if not os.path.exists(path):
                    print(f"ERROR: {label} not found at: {path}")
                    return 2
            proc = start_server(args)
            if not wait_for_server(url, 120):
                print("ERROR: llama-server did not become healthy in 120s")
                return 1
            print(f"llama-server healthy at {url} (pid {proc.pid})")

        results = []
        started = time.time()
        budget = 480.0
        for tape_dir in args.tapes:
            if time.time() - started > budget:
                print(f"ERROR: time budget {budget:.0f}s exhausted before {tape_dir}")
                exit_code = 1
                break
            print(f"\n=== Driving tape (all-live): {tape_dir} ===", flush=True)
            result = asyncio.run(drive_tape(tape_dir, args, temp_enrollment,
                                            intent_from_tape=False))
            results.append(result)
            print_report(result)
            if not result["ok"]:
                exit_code = 1
                if not result["llm_records"] or _ran_no_goal(result):
                    print("\n-> all-live pass never reached a planner; "
                          "repeating with the tape's recorded intent served")
                    result2 = asyncio.run(drive_tape(tape_dir, args, temp_enrollment,
                                                     intent_from_tape=True))
                    results.append(result2)
                    print_report(result2)
                    if not result2["ok"]:
                        exit_code = 1

        if args.json:
            out = os.path.abspath(args.json)
            with open(out, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(results, fh, indent=2, ensure_ascii=False)
            print(f"\nJSON report written to {out}")

    finally:
        if proc is not None:
            print("\nStopping llama-server...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
            alive = proc.poll() is None
            print(f"llama-server stopped (pid {proc.pid}, exited: {not alive}).")
        remaining = _llama_pids()
        new_left = [pid for pid in remaining if pid not in pre_existing]
        if new_left:
            print(f"WARNING: llama-server processes still running: {new_left}")
        else:
            print("No llama-server processes remain.")

    return exit_code


def _ran_no_goal(result: dict) -> bool:
    return result["goal_status"] in (None, "ok") and not result["steps"]


def _llama_pids() -> list[int]:
    pids: list[int] = []
    try:
        import psutil
        for proc in psutil.process_iter(["name"]):
            if (proc.info.get("name") or "").lower() == "llama-server.exe":
                pids.append(proc.pid)
    except Exception:
        pass
    return pids


def print_report(result: dict) -> None:
    print(f"\n--- {result['tape']} (pass: {result['pass']}) ---")
    print(f"goal status   : {result['goal_status']}")
    print(f"final response: {result['final_response']!r}")
    for note in result["notes"]:
        print(f"note: {note}")
    print("steps taken by the planner:")
    for step in result["steps"]:
        print(f"  #{step['n']} {step['verdict']} {step['action']} "
              f"params={json.dumps(step['params'], ensure_ascii=False)}")
        if step["thought"]:
            print(f"      thought: {step['thought']}")
        if step["expect"]:
            print(f"      expect : {step['expect']}")
    if not result["steps"]:
        print("  (none - no agent goal ran)")
    print("dispatches (tool -> status, marked when the served result came from another tool):")
    for d in result["dispatches"]:
        served_from = d.get("served_from")
        mark = (""
                if not served_from or served_from == d["tool"]
                else f" (served from {served_from}'s recorded result)")
        print(f"  {d['tool']}{mark}"
              f"{' (confirmed)' if d['confirmed'] else ''} -> {d['status']}")
        if d["params"]:
            print(f"      {json.dumps(d['params'], ensure_ascii=False)}")
        if d.get("text"):
            print(f"      result text: {_truncate(str(d['text']), 200)}")
    if not result["dispatches"]:
        print("  (none)")
    print("raw model outputs:")
    any_fail = any(s["verdict"].startswith("FAIL") for s in result["steps"])
    for i, rec in enumerate(result["llm_records"]):
        body = rec.get("raw")
        body = body if isinstance(body, str) and body else "<no reply>"
        print(f"  [{i+1}] {rec.get('call')}"
              f"{'' if not rec.get('image') else ' (with image)'}")
        print(f"      prompt tail: {(rec.get('prompt_tail') or '')!r}")
        if any_fail:
            print(f"      raw: {body}")
        else:
            print(f"      raw: {_truncate(body, 400)}")


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


if __name__ == "__main__":
    raise SystemExit(main())
