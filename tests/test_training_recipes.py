"""Unit tests verifying all 4 fine-tuning recipes run deterministic smoke steps and export artifacts."""

from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def _bench_results_to_tmp(tmp_path, monkeypatch):
    """Smoke tests must never append rows to the tracked bench/results/*.jsonl.

    Each full-suite run used to add ~7 rows to four tracked files, dirtying
    the repo. Point the benchmark writer's default directory at pytest's
    tmp_path for the duration of these tests; real training/benchmark runs
    outside tests keep writing to bench/results/ unchanged.
    """
    import bench.harness as bench_harness

    monkeypatch.setattr(bench_harness, "DEFAULT_RESULTS_DIR", tmp_path / "bench_results")


def test_tool_caller_smoke_run(tmp_path):
    from training.common import TrainingConfig
    import training.train_tool_caller as train_tc
    out_dir = tmp_path / "tc_out"
    cfg = TrainingConfig(
        model_name="lfm-tool-test",
        output_dir=out_dir,
        batch_size=2,
        epochs=1,
        smoke_run=True,
        device="cpu",
    )
    res = train_tc.run_training(cfg)
    assert res["total_steps"] == 2
    assert "gguf_path" in res
    assert Path(res["gguf_path"]).is_file()


def test_element_selection_smoke_run(tmp_path):
    from training.common import TrainingConfig
    import training.train_element_selection as train_es
    out_dir = tmp_path / "es_out"
    cfg = TrainingConfig(
        model_name="gliner-test",
        output_dir=out_dir,
        batch_size=2,
        epochs=1,
        smoke_run=True,
        device="cpu",
    )
    res = train_es.run_training(cfg)
    assert res["total_steps"] == 2
    assert "onnx_path" in res
    assert Path(res["onnx_path"]).is_file()


def test_contrastive_heads_smoke_run(tmp_path):
    from training.common import TrainingConfig
    import training.train_contrastive_heads as train_ch
    out_dir = tmp_path / "ch_out"
    cfg = TrainingConfig(
        model_name="ch-test",
        output_dir=out_dir,
        batch_size=2,
        epochs=1,
        smoke_run=True,
        device="cpu",
    )
    res = train_ch.run_training(cfg)
    assert res["total_steps"] == 2
    assert "cached_actions_path" in res
    assert Path(res["cached_actions_path"]).is_file()
    assert "onnx_path" in res
    assert Path(res["onnx_path"]).is_file()


def test_vision_mark_smoke_run(tmp_path):
    from training.common import TrainingConfig
    import training.train_vision_mark as train_vm
    out_dir = tmp_path / "vm_out"
    cfg = TrainingConfig(
        model_name="vm-test",
        output_dir=out_dir,
        batch_size=2,
        epochs=1,
        smoke_run=True,
        device="cpu",
    )
    res = train_vm.run_training(cfg)
    assert res["total_steps"] == 2
    assert "lfm_top1_accuracy" in res
    assert "challenger_top1_accuracy" in res


def test_data_prep_working_set(tmp_path):
    from training.data_prep import prep_working_set
    corpus_dir = tmp_path / "corpus"
    working_set = tmp_path / "working_set"
    corpus_dir.mkdir()
    working_set.mkdir()

    counts = prep_working_set(corpus_dir, working_set)
    assert "audio_files_converted" in counts
    assert "vision_images_converted" in counts
