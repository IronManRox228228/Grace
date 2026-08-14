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

from grace.harness.replay import TapeDispatcher, TapeLlm
from grace.llm.gemma_client import GemmaClient
from grace.tools.dispatcher import Dispatcher

pytestmark = pytest.mark.integration


def keyword_names(func) -> set[str]:
    return {
        name for name, param in inspect.signature(func).parameters.items()
        if name not in ("self", "cls")
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }


@pytest.mark.parametrize("method", ["chat", "generate_text", "generate_intent"])
def test_the_replay_llm_accepts_what_the_real_one_does(method):
    real = keyword_names(getattr(GemmaClient, method))
    stub = keyword_names(getattr(TapeLlm, method))
    missing = real - stub
    assert missing == set(), (
        f"TapeLlm.{method} does not accept {sorted(missing)}, which "
        f"GemmaClient.{method} does. Every replay that passes one will get a "
        f"TypeError swallowed as an empty response."
    )


def test_the_replay_dispatcher_accepts_what_the_real_one_does():
    missing = keyword_names(Dispatcher.execute) - keyword_names(TapeDispatcher.execute)
    assert missing == set(), (
        f"TapeDispatcher.execute does not accept {sorted(missing)}"
    )


def test_the_scripted_generator_stub_accepts_the_real_stream_signature():
    # The generator patches `_stream_gemini_response` itself, so its stub has to
    # match that method rather than `chat`.
    from grace.harness.generate import _scripted_gemini

    class _Overruns:
        def note(self, _message):
            pass

    stub = _scripted_gemini([], _Overruns())
    real = keyword_names(GemmaClient._stream_gemini_response)
    missing = real - keyword_names(stub)
    assert missing == set(), (
        f"the scripted Gemini stub does not accept {sorted(missing)}; every "
        f"call passing one is recorded as an empty model response"
    )
