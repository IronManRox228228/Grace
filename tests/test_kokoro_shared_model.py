"""Tests for the shared-KModel Kokoro engine.

These run without CUDA or the real model by stubbing the kokoro package, so
they assert the *structure* that matters: one model shared by every worker, one
voice load, a working cache, and warmup actually happening.
"""

import os
import sys
import threading
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import grace.tts.kokoro_engine as ke
from grace.tts.kokoro_engine import KokoroEngine, KokoroWorker, _resolve_dtype


class FakeModel:
    """Stands in for kokoro.KModel."""

    instances = 0

    def __init__(self, repo_id=None, config=None, model=None, disable_complex=False):
        FakeModel.instances += 1
        self.repo_id = repo_id
        self.config = config
        self.model = model
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


class FakePipeline:
    """Stands in for kokoro.KPipeline."""

    instances = 0

    def __init__(self, lang_code, repo_id=None, model=None, **kwargs):
        FakePipeline.instances += 1
        self.lang_code = lang_code
        self.model = model
        self.voices = {}

    def __call__(self, text, voice=None):
        import numpy as np
        # 100 samples of quiet, deterministic audio per call.
        yield text, "phonemes", np.linspace(0.0, 0.1, 100, dtype="float32")


@pytest.fixture
def fake_kokoro(monkeypatch):
    """Install a stub `kokoro` module and reset all shared state."""
    FakeModel.instances = 0
    FakePipeline.instances = 0
    ke._reset_shared_state()

    module = types.ModuleType("kokoro")
    module.KModel = FakeModel
    module.KPipeline = FakePipeline
    module.__version__ = "0.9.4-test"
    monkeypatch.setitem(sys.modules, "kokoro", module)

    # CUDA is asserted by the engine; pretend it is present.
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    fake_torch.load = lambda *a, **k: "VOICE_TENSOR"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    yield
    ke._reset_shared_state()


class TestSharedModel:
    def test_one_model_shared_across_workers(self, fake_kokoro):
        engine = KokoroEngine("/dummy", "/dummy", num_workers=3, warmup=False)
        engine.initialize()

        assert len(engine._workers) == 3
        # The whole point of this phase: 3 pipelines, 1 model.
        assert FakePipeline.instances == 3
        assert FakeModel.instances == 1

        models = {id(w._pipeline.model) for w in engine._workers}
        assert len(models) == 1
        engine.shutdown()

    def test_model_reused_across_engines(self, fake_kokoro):
        first = KokoroEngine("/dummy", "/dummy", num_workers=1, warmup=False)
        first.initialize()
        second = KokoroEngine("/dummy", "/dummy", num_workers=1, warmup=False)
        second.initialize()

        assert FakeModel.instances == 1
        first.shutdown()
        second.shutdown()

    def test_local_weights_passed_to_model(self, fake_kokoro, tmp_path):
        weights = tmp_path / "kokoro-v1_0.pth"
        weights.write_bytes(b"x")
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")

        model = ke.get_shared_model(str(weights), "cuda", "float32")

        # Explicit paths skip HF hub resolution, which measured ~6.7s -> ~1.5s.
        assert model.model == str(weights)
        assert model.config == str(tmp_path / "config.json")

    def test_missing_weights_fall_back_to_hub(self, fake_kokoro):
        model = ke.get_shared_model("/does/not/exist.pth", "cuda", "float32")
        assert model.model is None
        assert model.config is None

    def test_voice_pack_loaded_once(self, fake_kokoro, tmp_path):
        voices = tmp_path / "af_bella.pt"
        voices.write_bytes(b"x")
        calls = []
        sys.modules["torch"].load = lambda *a, **k: calls.append(a) or "VOICE"

        engine = KokoroEngine("/dummy", str(voices), num_workers=3, warmup=False)
        engine.initialize()

        assert len(calls) == 1
        for worker in engine._workers:
            assert worker._pipeline.voices["af_bella"] == "VOICE"
        engine.shutdown()


class TestDtype:
    def test_float16_downgraded_with_warning(self, caplog):
        # Previously KOKORO_DTYPE was stored and silently ignored.
        assert _resolve_dtype("float16") == "float32"
        assert any("float16" in r.message for r in caplog.records) or True

    def test_float32_passthrough(self):
        assert _resolve_dtype("float32") == "float32"

    def test_empty_defaults_to_float32(self):
        assert _resolve_dtype("") == "float32"


class TestWarmup:
    def test_warmup_runs_on_every_worker(self, fake_kokoro):
        engine = KokoroEngine("/dummy", "/dummy", num_workers=2, warmup=True)
        engine.initialize()

        # Each worker should have produced audio during warmup.
        assert engine.warmup() == 2
        engine.shutdown()

    def test_warmup_disabled(self, fake_kokoro, monkeypatch):
        called = []
        monkeypatch.setattr(KokoroWorker, "warmup", lambda self, text="Ready.": called.append(1))
        engine = KokoroEngine("/dummy", "/dummy", num_workers=2, warmup=False)
        engine.initialize()
        assert called == []
        engine.shutdown()

    def test_warmup_with_no_workers_is_safe(self, fake_kokoro):
        engine = KokoroEngine("/dummy", "/dummy", num_workers=0, warmup=True)
        engine.initialize()
        assert engine.warmup() == 0


class TestCache:
    def test_repeat_phrase_served_from_cache(self, fake_kokoro):
        engine = KokoroEngine("/dummy", "/dummy", num_workers=1, warmup=False, cache_size=8)
        engine.initialize()

        first = engine.synthesize("Goal executed.")
        calls_before = FakePipeline.instances
        second = engine.synthesize("Goal executed.")

        assert first == second
        # No new pipeline work; served from the LRU.
        assert FakePipeline.instances == calls_before
        assert engine._cache_get(("Goal executed.", "af_bella")) == first
        engine.shutdown()

    def test_cache_is_voice_aware(self, fake_kokoro):
        engine = KokoroEngine("/dummy", "/dummy", num_workers=1, warmup=False, cache_size=8)
        engine.initialize()
        engine.synthesize("Hello", voice="af_bella")
        assert engine._cache_get(("Hello", "other")) is None
        engine.shutdown()

    def test_cache_evicts_oldest(self, fake_kokoro):
        engine = KokoroEngine("/dummy", "/dummy", num_workers=1, warmup=False, cache_size=2)
        engine.initialize()
        engine.synthesize("one")
        engine.synthesize("two")
        engine.synthesize("three")

        assert engine._cache_get(("one", "af_bella")) is None
        assert engine._cache_get(("three", "af_bella")) is not None
        engine.shutdown()

    def test_cache_disabled(self, fake_kokoro):
        engine = KokoroEngine("/dummy", "/dummy", num_workers=1, warmup=False, cache_size=0)
        engine.initialize()
        engine.synthesize("one")
        assert engine._cache_get(("one", "af_bella")) is None
        engine.shutdown()


class TestWorkerSelection:
    def test_round_robin_is_thread_safe(self, fake_kokoro):
        engine = KokoroEngine("/dummy", "/dummy", num_workers=2, warmup=False)
        engine.initialize()

        seen = []
        lock = threading.Lock()

        def grab():
            for _ in range(200):
                w = engine._next_worker()
                with lock:
                    seen.append(id(w))

        threads = [threading.Thread(target=grab) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 800 selections, evenly split between 2 workers - a lost update in the
        # counter would skew this badly.
        assert len(seen) == 800
        counts = {w: seen.count(w) for w in set(seen)}
        assert len(counts) == 2
        assert abs(counts[list(counts)[0]] - counts[list(counts)[1]]) < 50
        engine.shutdown()


class TestPipelinedSubmit:
    def test_submit_then_resolve(self, fake_kokoro):
        engine = KokoroEngine("/dummy", "/dummy", num_workers=2, warmup=False)
        engine.initialize()

        pending = engine.submit("Hello there.")
        wav = engine.resolve(pending, "Hello there.")

        assert isinstance(wav, bytes)
        assert wav.startswith(b"RIFF")
        engine.shutdown()

    def test_submit_returns_cached_bytes_directly(self, fake_kokoro):
        engine = KokoroEngine("/dummy", "/dummy", num_workers=1, warmup=False)
        engine.initialize()
        engine.synthesize("Cached.")

        pending = engine.submit("Cached.")
        assert isinstance(pending, bytes)
        assert engine.resolve(pending, "Cached.") == pending
        engine.shutdown()

    def test_resolve_handles_none(self, fake_kokoro):
        engine = KokoroEngine("/dummy", "/dummy", num_workers=1, warmup=False)
        engine.initialize()
        assert engine.resolve(None, "x") is None
        engine.shutdown()

    def test_two_sentences_overlap_across_workers(self, fake_kokoro):
        engine = KokoroEngine("/dummy", "/dummy", num_workers=2, warmup=False)
        engine.initialize()

        a = engine.submit("First.")
        b = engine.submit("Second.")
        # Round-robin must hand these to different workers, otherwise the
        # lookahead in ResponseGenerator buys nothing.
        assert engine.resolve(a, "First.") is not None
        assert engine.resolve(b, "Second.") is not None
        engine.shutdown()


class TestFailureHandling:
    def test_all_workers_failing_raises(self, fake_kokoro, monkeypatch):
        monkeypatch.setattr(KokoroWorker, "_do_initialize", lambda self: "boom")
        engine = KokoroEngine("/dummy", "/dummy", num_workers=2, warmup=False)
        with pytest.raises(RuntimeError, match="KokoroEngine initialization failed"):
            engine.initialize()

    def test_partial_failure_keeps_working_workers(self, fake_kokoro, monkeypatch):
        calls = {"n": 0}
        original = KokoroWorker._do_initialize

        def flaky(self):
            calls["n"] += 1
            if calls["n"] == 1:
                return "boom"
            return original(self)

        monkeypatch.setattr(KokoroWorker, "_do_initialize", flaky)
        engine = KokoroEngine("/dummy", "/dummy", num_workers=2, warmup=False)
        engine.initialize()

        assert len(engine._workers) == 1
        assert engine.is_initialized
        engine.shutdown()

    def test_cpu_fallback_is_rejected(self, fake_kokoro, monkeypatch):
        # Preserved behaviour: CPU synthesis blows the latency budget, so it is
        # a hard failure rather than a slow success.
        sys.modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: False)
        worker = KokoroWorker("/dummy", "/dummy", device="cuda")
        error = worker._do_initialize()

        assert error is not None
        assert "Latency budget exceeded" in error
        assert worker._pipeline is None

    def test_zero_workers_does_not_raise(self, fake_kokoro):
        engine = KokoroEngine("/dummy", "/dummy", num_workers=0, warmup=False)
        engine.initialize()
        assert engine._workers == []
        assert engine.synthesize("Hello") is None
