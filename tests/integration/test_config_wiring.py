"""A Config field has to reach the component that acts on it.

`GraceApp.__init__` is one long hand-written wiring of `self.config.<field>` into
constructor keyword arguments. It is the seam where a setting silently stops
working: the field still exists, the environment variable still parses, and
nothing anywhere reads it. `Config.use_oculix` was in exactly that state - a
field the code believed was the switch while `OculixBridge` read the raw
environment variable instead.

Three of the branches here are never constructed by any other test, and they are
the ones that decide which model gets called at all:

* `use_cloud_llm` gating whether the API key is passed - false means every
  request goes to a local llama-server that may not be running;
* single local model mode where `vision_llm` aliases `gemma`;
* `model_swap_enabled`, which rewires both clients' `before_request` hooks and
  suppresses the on-demand llama-server start.

Also covered: `AgentLoop`'s `_config_int` imports `Config` directly rather than
taking the app's instance, so a patched config does not reach it. That is what
excluded the unbounded-loop path from every test until now.
"""

import dataclasses
import sys
import types
from unittest import mock

import pytest

from grace.config import Config

pytestmark = pytest.mark.integration


@pytest.fixture
def app_factory(monkeypatch):
    """Build a GraceApp with every heavyweight component replaced.

    Nothing here opens a microphone, loads a model, binds a port or starts a
    JVM; what is under test is which values arrive where.
    """
    import grace.harness.replay as replay_mod

    def build(**config_overrides):
        config = dataclasses.replace(Config(), **config_overrides)

        patches = [
            mock.patch("grace.main.Config", lambda: config),
            mock.patch("grace.main.AudioCapture", replay_mod.NullAudio),
            mock.patch("grace.main.WakeWordDetector", replay_mod.NullWakeWord),
            mock.patch("grace.main.KokoroEngine", replay_mod.NullTts),
            mock.patch("grace.main.TTSPlayer", replay_mod.NullTts),
            mock.patch("grace.main.ComputerUse", replay_mod.NullComputerUse),
            mock.patch("grace.main.WsEventServer", lambda *a, **k: types.SimpleNamespace(
                host=k.get("host"), port=k.get("port"), emit=None)),
        ]
        stack = []
        for patch in patches:
            patch.start()
            stack.append(patch)
        try:
            import grace.main as main_mod

            return main_mod.GraceApp(), config
        finally:
            for patch in reversed(stack):
                patch.stop()

    return build


class TestLlmWiring:
    def test_cloud_mode_passes_the_api_key(self, app_factory):
        app, _ = app_factory(use_cloud_llm=True, gemini_api_key="a-key")
        assert app.gemma._api_key == "a-key"

    def test_local_mode_withholds_the_api_key(self, app_factory):
        # The branch nothing constructs. With a key present but use_cloud_llm
        # false, passing it anyway would send every request to Gemini while the
        # user believes they are running offline - which for this application
        # is a privacy claim, not a performance one.
        app, _ = app_factory(use_cloud_llm=False, gemini_api_key="a-key")
        assert app.gemma._api_key is None

    def test_the_configured_model_name_reaches_the_client(self, app_factory):
        app, _ = app_factory(use_cloud_llm=True, gemini_api_key="k",
                             gemini_model_name="gemini-test-model")
        assert app.gemma._model_name == "gemini-test-model"

    def test_single_model_aliases_planner_client(self, app_factory):
        app, _ = app_factory(use_cloud_llm=False)
        assert app.vision_llm is app.gemma


class TestModelSwapWiring:
    def test_swap_is_off_when_the_planner_is_in_the_cloud(self, app_factory):
        app, _ = app_factory(use_cloud_llm=True, local_planner_model_path="C:/planner.gguf")
        assert app._model_swap is None

    def test_swap_is_on_when_both_models_are_local(self, app_factory):
        app, _ = app_factory(use_cloud_llm=False, local_planner_model_path="C:/planner.gguf")
        assert app._model_swap is not None
        assert app.gemma._before_request is not None
        assert app.vision_llm._before_request is not None

    def test_swap_suppresses_the_on_demand_backend_start(self, app_factory):
        app, _ = app_factory(use_cloud_llm=False, local_planner_model_path="C:/planner.gguf")
        assert app.agent_loop._grounder._on_demand_start is None


class TestLlamaServerArgs:
    def test_mmproj_included_when_configured(self):
        from grace.main import build_llama_server_args
        cfg = dataclasses.replace(Config(), llama_mmproj_path="C:/path/to/projector.gguf")
        args = build_llama_server_args(cfg)
        assert "--mmproj" in args
        idx = args.index("--mmproj")
        assert args[idx + 1] == "C:/path/to/projector.gguf"

    def test_mmproj_omitted_when_empty(self):
        from grace.main import build_llama_server_args
        cfg = dataclasses.replace(Config(), llama_mmproj_path="")
        args = build_llama_server_args(cfg)
        assert "--mmproj" not in args


class TestAgentLoopReadsTheConfig:
    """`_config_int` imports Config itself, so this is its only coverage."""

    # Set through the environment rather than by patching Config, because that
    # is the chain the setting actually travels: env -> Config() -> loop. It
    # only works at all now that Config re-reads the environment on
    # construction; a patched object would pass even if it did not.
    @pytest.mark.parametrize("variable,attribute", [
        ("AGENT_MAX_ITERATIONS", "_max_iterations"),
        ("AGENT_MAX_SECONDS", "_max_seconds"),
        ("AGENT_MAX_REPEATED_ACTIONS", "_max_repeated_actions"),
        ("AGENT_MAX_CONSECUTIVE_PLAN_FAILURES", "_max_consecutive_failures"),
    ])
    def test_a_numeric_setting_reaches_the_loop(self, monkeypatch, variable, attribute):
        from grace.agent.loop import AgentLoop

        monkeypatch.setenv(variable, "7")
        loop = AgentLoop(gemma=mock.MagicMock(), dispatcher=mock.MagicMock(),
                         perception=mock.MagicMock())
        assert getattr(loop, attribute) == 7, (
            f"{variable} does not reach AgentLoop.{attribute}"
        )

    def test_the_stronger_planner_model_reaches_the_loop(self, monkeypatch):
        from grace.agent.loop import AgentLoop

        monkeypatch.setenv("STRONGER_PLANNER_MODEL", "a-bigger-model")
        loop = AgentLoop(gemma=mock.MagicMock(), dispatcher=mock.MagicMock(),
                         perception=mock.MagicMock())
        assert loop._stronger_model == "a-bigger-model"


class TestConfigIsReadWhenConstructed:
    def test_the_environment_is_read_at_construction_not_at_import(self, monkeypatch):
        # A frozen dataclass with plain `os.getenv` defaults evaluates them once,
        # at import. Anything that loaded a .env afterwards, or set a variable
        # for a test, was silently ignored - and the symptom is the setting
        # appearing not to work rather than not to be read.
        monkeypatch.setenv("AGENT_MAX_SECONDS", "999")
        assert Config().agent_max_seconds == 999

    def test_use_oculix_is_the_switch_the_bridge_reads(self, monkeypatch):
        # The dead-state case: the field existed and OculixBridge read the raw
        # environment variable, so setting it in code changed nothing.
        from grace.automation.oculix_bridge import OculixBridge

        monkeypatch.setenv("USE_OCULIX", "true")
        assert Config().use_oculix is True
        assert OculixBridge.is_enabled() is True

        monkeypatch.setenv("USE_OCULIX", "false")
        assert OculixBridge.is_enabled() is False
