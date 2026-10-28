"""Typed-command entrypoint: text enters, and the exact voice path runs.

Everything under test here is the guarantee the entrypoint exists for: a typed
string follows the identical routing a heard string takes - the same stop and
repeat reflexes, the same parked-confirmation answers, the same dispatcher
safety gate - and `--no-tts` only silences the audio engines, never the flow.

The model is fake, the mic, speaker and screen stacks are mocked, memory and
enrollment live in temp paths (RULES.md rule 1), and the WebSocket server is
the replay harness's capturing stand-in, which still validates the frozen
event contract. Serial only (rule 12): no -n auto, no parallelism.
"""
import asyncio
import dataclasses
import tempfile
from typing import Optional, Union
from unittest import mock

import pytest

pytestmark = pytest.mark.integration


class FakeGemma:
    """Deterministic GemmaClient stand-in; never touches a network."""

    def __init__(self, *args, **kwargs):
        self.intent_reply: Optional[str] = None
        self.plan_replies: list[str] = []
        self.intent_calls: list[str] = []
        self.plan_calls: list[dict] = []
        self.closed = False

    async def generate_intent(self, user_message: str, system_prompt=None, **kwargs):
        self.intent_calls.append(user_message)
        return self.intent_reply

    async def generate_text(self, prompt=None, system_prompt=None, temperature=0.2,
                            max_tokens=8192, messages=None, image_b64=None,
                            model=None, **kwargs):
        self.plan_calls.append({"image": bool(image_b64), "prompt_tail": (prompt or "")[-200:]})
        if self.plan_replies:
            return self.plan_replies.pop(0)
        return None

    async def chat(self, messages, **kwargs):
        return None

    async def abort(self):
        return None

    async def close(self):
        self.closed = True


class RecordingDispatcher:
    """A Dispatcher whose side effects are decided by the test.

    ``results[tool]`` holds either one result dict returned for every call, or
    a list served in order (the last entry repeats), so a test can program the
    gate's "refuse once, allow once confirmed" story. Every call is recorded,
    including how it was confirmed, so tests can assert the flow was walked.
    """

    def __init__(self, *args, **kwargs):
        self.calls: list[dict] = []
        self.results: dict[str, Union[dict, list]] = {}

    def set_kokoro_engine(self, engine):
        self.kokoro_engine = engine

    def set_training_scheduler(self, scheduler):
        self.training_scheduler = scheduler

    async def execute(self, intent, confirmed: bool = False):
        self.calls.append({
            "tool": intent.tool,
            "params": {k: v for k, v in intent.params.items() if not k.startswith("_")},
            "confirmed": confirmed,
        })
        canned = self.results.get(intent.tool)
        if isinstance(canned, list):
            canned = canned[min(len(canned) - 1, len(self.calls) - 1)]
        if canned is None:
            canned = {"status": "ok", "text": "done."}
        return dict(canned)


class FakePerception:
    """One fake screen: the blank generated view. Same API RepeatingSnapshots uses."""

    def __init__(self):
        self.count = 0

    def capture_snapshot(self):
        self.count += 1
        from grace.harness.generate import blank_screen
        from grace.harness.replay import _rebuild_snapshot

        return _rebuild_snapshot(blank_screen())

    async def capture_snapshot_async(self):
        return self.capture_snapshot()


def build_typed_app(intent_reply: Optional[str] = None, plan_replies=None,
                    dispatcher_results=None, tts_off: bool = False,
                    tts_forbidden: bool = False):
    """A GraceApp wired for a typed-command test.

    Same heavyweight-component patches the config-wiring tests use, plus a
    FakeGemma, a RecordingDispatcher, a CapturingWsServer (contract surface),
    and real memory/enrollment classes pointed at temp or in-memory storage.
    Returns (app, gemma, dispatcher, ws_server).
    """
    import grace.main as main_mod
    import grace.harness.replay as replay_mod
    from grace.agent.memory import PersistentMemoryStore as RealMemoryStore
    from grace.config import Config
    from grace.enrollment.storage import EnrollmentStorage as RealEnrollmentStorage

    config = dataclasses.replace(Config(), use_cloud_llm=False, gemini_api_key="")

    gemma = FakeGemma()
    gemma.intent_reply = intent_reply
    gemma.plan_replies = list(plan_replies or [])
    dispatcher = RecordingDispatcher()
    dispatcher.results.update(dispatcher_results or {})
    ws = replay_mod.CapturingWsServer()

    def _forbidden_engine(*args, **kwargs):
        raise AssertionError("a --no-tts app must never construct a real TTS engine")

    patches = [
        mock.patch("grace.main.Config", lambda: config),
        mock.patch("grace.main.AudioCapture", replay_mod.NullAudio),
        mock.patch("grace.main.WakeWordDetector", lambda *a, **k: replay_mod.NullWakeWord()),
        mock.patch("grace.main.WhisperStreaming", lambda *a, **k: mock.MagicMock()),
        mock.patch("grace.main.ComputerUse", lambda *a, **k: mock.MagicMock()),
        mock.patch("grace.main.GemmaClient", lambda *a, **k: gemma),
        mock.patch("grace.main.Dispatcher", lambda *a, **k: dispatcher),
        mock.patch("grace.main.WsEventServer", lambda *a, **k: ws),
        mock.patch("grace.main.FeedbackSounds", mock.MagicMock()),
        mock.patch("grace.agent.memory.PersistentMemoryStore",
                   lambda *a, **k: RealMemoryStore(db_path=":memory:")),
        mock.patch("grace.enrollment.storage.EnrollmentStorage",
                   lambda *a, **k: RealEnrollmentStorage(base_dir=tempfile.mkdtemp(
                       prefix="grace-typed-test-", dir=tempfile.gettempdir()))),
    ]
    if tts_forbidden:
        patches.extend([
            mock.patch("grace.main.KokoroEngine", _forbidden_engine),
            mock.patch("grace.main.TTSPlayer", _forbidden_engine),
        ])
    else:
        patches.extend([
            mock.patch("grace.main.KokoroEngine", replay_mod.NullTts),
            mock.patch("grace.main.TTSPlayer", replay_mod.NullTts),
        ])

    stack = []
    for patch in patches:
        patch.start()
        stack.append(patch)
    try:
        app = main_mod.GraceApp(tts_off=tts_off)
    finally:
        for patch in reversed(stack):
            patch.stop()

    # The agentic loop must never touch the real screen in a test.
    app.agent_loop._perception = FakePerception()
    app._running = True
    return app, gemma, dispatcher, ws


class TestTypedEntrypointAgentGoal:
    """A typed string reaches the agentic loop and executes with the gate."""

    def test_typed_command_reaches_the_agent_loop_and_executes(self):
        app, gemma, dispatcher, ws = build_typed_app(
            intent_reply='{"tool": "cua_click", "params": {"element_id": 2}}',
            plan_replies=[
                '{"thought": "Click it.", "action": "cua_click", '
                '"params": {"element_id": 2}, "expect": "the button is pressed", '
                '"is_completed": true, "final_response": "I clicked it."}',
            ],
        )
        try:
            result = asyncio.run(app._handle_text_activation("click the sign in button"))

            assert result is True
            # Cockpit 9: `done: true` ends the task and speaks `say`; the
            # step's cua_click is never executed, so nothing reaches the
            # desktop.
            assert dispatcher.calls == []
            assert app._last_spoken_text == "I clicked it."
            assert gemma.intent_calls == ["click the sign in button"]
            kinds = [e["type"] for e in ws.events]
            # The skipped action never reaches the desktop, so no tool events
            # are emitted; the turn ends with the spoken say instead.
            assert "ToolExecutionStarted" not in kinds
            assert "ConversationFinished" in kinds
        finally:
            asyncio.run(app.gemma.close())
            assert gemma.closed

    def test_typed_stop_reflex_cancels_instead_of_routing(self):
        app, gemma, dispatcher, ws = build_typed_app(
            intent_reply='{"tool": "cua_click", "params": {"element_id": 2}}',
        )
        try:
            result = asyncio.run(app._handle_text_activation("stop"))

            assert result is False
            assert dispatcher.calls == []
            assert gemma.intent_calls == []  # no model call behind a reflex
        finally:
            asyncio.run(app.gemma.close())


class TestTypedEntrypointSafetyGate:
    """A typed confirmation answer goes through the same parked yes/no flow."""

    def _parked_app(self):
        return build_typed_app(
            intent_reply='{"tool": "delete_file", "params": {"name": "draft.txt"}}',
            dispatcher_results={
                "delete_file": [
                    {"status": "confirmation_required",
                     "confirmation_prompt": "May I permanently delete draft.txt?",
                     "text": ""},
                    {"status": "ok", "text": "draft.txt deleted."},
                ],
            },
        )

    def test_yes_answer_confirms_through_the_same_dispatcher(self):
        app, gemma, dispatcher, ws = self._parked_app()
        try:
            with mock.patch("builtins.input", lambda *a, **k: "yes"):
                asyncio.run(app._run_typed_session("delete the file draft.txt"))

            assert len(dispatcher.calls) == 2
            assert dispatcher.calls[0]["confirmed"] is False
            assert dispatcher.calls[1]["confirmed"] is True
            assert dispatcher.calls[1]["tool"] == "delete_file"
            assert "ConversationFinished" in [e["type"] for e in ws.events]
        finally:
            asyncio.run(app.gemma.close())

    def test_no_answer_never_dispatches_confirmed(self):
        app, gemma, dispatcher, ws = self._parked_app()
        try:
            with mock.patch("builtins.input", lambda *a, **k: "no"):
                asyncio.run(app._run_typed_session("delete the file draft.txt"))

            assert len(dispatcher.calls) == 1
            assert dispatcher.calls[0]["confirmed"] is False
        finally:
            asyncio.run(app.gemma.close())

    def test_unclear_answer_reasks_and_keeps_parking(self):
        app, gemma, dispatcher, ws = self._parked_app()
        try:
            answers = iter(["I am not sure", "yes"])
            with mock.patch("builtins.input", lambda *a, **k: next(answers)):
                asyncio.run(app._run_typed_session("delete the file draft.txt"))

            confirmed = [c for c in dispatcher.calls if c["confirmed"]]
            assert len(confirmed) == 1  # only the final 'yes' approved anything
        finally:
            asyncio.run(app.gemma.close())


class TestNoTts:
    def test_no_tts_swaps_both_engines_for_null(self):
        app, gemma, dispatcher, ws = build_typed_app(tts_off=True, tts_forbidden=True)
        try:
            from grace.harness.replay import NullTts

            assert isinstance(app.kokoro, NullTts)
            assert isinstance(app.tts_player, NullTts)
            # The dispatcher still got a working engine handle.
            assert dispatcher.kokoro_engine is app.kokoro
            assert app.response_gen._kokoro is app.kokoro
            assert app.response_gen._player is app.tts_player
        finally:
            asyncio.run(app.gemma.close())

    def test_no_tts_still_runs_the_full_typed_path(self):
        app, gemma, dispatcher, ws = build_typed_app(
            tts_off=True, tts_forbidden=True,
            intent_reply='{"tool": "delete_file", "params": {"name": "draft.txt"}}',
        )
        try:
            with mock.patch("builtins.input", lambda *a, **k: "no"):
                asyncio.run(app._run_typed_session("delete the file draft.txt"))

            assert len(dispatcher.calls) == 1
            assert dispatcher.calls[0]["confirmed"] is False
            assert "ConversationFinished" in [e["type"] for e in ws.events]
        finally:
            asyncio.run(app.gemma.close())
