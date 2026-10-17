"""A harness stub must accept everything the production call site passes.

Adding a `model` parameter to `GemmaClient.chat` - one keyword, one caller -
broke three stubs at once. None of them failed loudly: the scripted generator's
`_stream` raised `TypeError`, which `generate_text` catches along with
everything else and turns into `None`, which the planner reads as "the model
returned nothing", which the loop correctly treats as an unusable plan and gives
up on after three. The visible symptom was "AgentLoop: giving up after 3
consecutive plans that could not be used" - a message about the agent, from a
mismatched function signature two layers down.

That is the whole hazard: the stubs stand in for production objects but nothing
holds them to production's shape, so they drift silently and the failure surfaces
somewhere unrelated. These tests compare the signatures directly.
"""

import inspect

import pytest

from grace.agent.perception import PerceptionEngine
from grace.audio.capture import AudioCapture
from grace.audio.pump import AudioPump
from grace.audio.wake_word import WakeWordDetector
from grace.automation.computer_use import ComputerUse
from grace.stt.whisper_stream import WhisperStreaming
from grace.harness.generate import (
    ScriptedComputerUse,
    ScriptedPerception,
    ScriptedPump,
    _scripted_gemini,
)
from grace.harness.replay import (
    CapturingWsServer,
    NullAudio,
    NullComputerUse,
    NullTts,
    NullWakeWord,
    TapeDispatcher,
    TapeLlm,
    TapePerception,
    TapePump,
    TapeWhisper,
    _NullFeedbackSounds,
)
from grace.llm.gemma_client import GemmaClient
from grace.response.feedback import FeedbackSounds
from grace.tools.dispatcher import Dispatcher
from grace.tts.kokoro_engine import KokoroEngine
from grace.ws_server import WsEventServer

pytestmark = pytest.mark.integration


def missing_keywords(real_func, stub_func) -> set[str]:
    stub_sig = inspect.signature(stub_func)
    for param in stub_sig.parameters.values():
        if param.kind == param.VAR_KEYWORD:
            return set()
    real_sig = inspect.signature(real_func)
    real = {
        name for name, param in real_sig.parameters.items()
        if name not in ("self", "cls")
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    stub = {
        name for name, param in stub_sig.parameters.items()
        if name not in ("self", "cls")
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    return real - stub


from grace.tts.player import TTSPlayer

STUB_METHOD_PAIRS = [
    # (RealClass, StubClass, method_name)
    (GemmaClient, TapeLlm, "chat"),
    (GemmaClient, TapeLlm, "generate_text"),
    (GemmaClient, TapeLlm, "generate_intent"),
    (Dispatcher, TapeDispatcher, "execute"),
    (PerceptionEngine, TapePerception, "capture_snapshot"),
    (PerceptionEngine, ScriptedPerception, "capture_snapshot"),
    (AudioPump, TapePump, "get"),
    (AudioPump, ScriptedPump, "get"),
    (WhisperStreaming, TapeWhisper, "transcribe"),
    (WsEventServer, CapturingWsServer, "emit"),
    (FeedbackSounds, _NullFeedbackSounds, "play_chime"),
    (FeedbackSounds, _NullFeedbackSounds, "play_success"),
    (FeedbackSounds, _NullFeedbackSounds, "play_cancel"),
    (AudioCapture, NullAudio, "get_chunk"),
    (AudioCapture, NullAudio, "get_rms"),
    (WakeWordDetector, NullWakeWord, "start"),
    (WakeWordDetector, NullWakeWord, "stop"),
    (KokoroEngine, NullTts, "synthesize"),
    (TTSPlayer, NullTts, "play"),
    (TTSPlayer, NullTts, "stop"),
    (ComputerUse, NullComputerUse, "perform"),
    (ComputerUse, ScriptedComputerUse, "perform"),
]


@pytest.mark.parametrize("real_cls,stub_cls,method", STUB_METHOD_PAIRS)
def test_all_harness_stubs_accept_what_real_classes_do(real_cls, stub_cls, method):
    real_fn = getattr(real_cls, method)
    stub_fn = getattr(stub_cls, method)
    missing = missing_keywords(real_fn, stub_fn)
    assert missing == set(), (
        f"{stub_cls.__name__}.{method} does not accept {sorted(missing)}, which "
        f"{real_cls.__name__}.{method} requires."
    )


def test_the_scripted_generator_stub_accepts_the_real_stream_signature():
    class _Overruns:
        def note(self, _message):
            pass

    stub = _scripted_gemini([], _Overruns())
    real_fn = GemmaClient._stream_gemini_response
    missing = missing_keywords(real_fn, stub)
    assert missing == set(), (
        f"the scripted Gemini stub does not accept {sorted(missing)}; every "
        f"call passing one is recorded as an empty model response"
    )
