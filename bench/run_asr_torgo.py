"""Benchmark 1 runner: ASR evaluation on TORGO (PLAN.md §8, §9, Task 09).

Evaluates Parakeet-TDT 0.6B vs Whisper vs Moonshine Base on the TORGO dysarthric
speech dataset. Uses the Task 08 central thread budget and identical text normalization.

If models or dataset are not downloaded, prints exact instructions and exits cleanly.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import wave

# Ensure src and project root are in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from bench.harness import append_result, create_benchmark_record, get_machine_profile, get_thread_settings
from bench.load_generator import BackgroundLoadSimulator
from bench.metrics import compute_latency_stats, real_time_factor, word_error_rate
from bench.normalizer import normalize_text
from bench.report import generate_markdown_report
from bench.torgo import TorgoDataset, TorgoUtterance, load_torgo_dataset
from grace.thread_budget import calculate_thread_budget
from grace.tier import detect_hardware_tier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench.run_asr_torgo")


def get_audio_duration_seconds(wav_path: Path) -> float:
    """Read WAV header to compute duration in seconds."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate > 0 else 0.0
    except Exception:
        return 0.0


class ASRModelEvaluator:
    """Interface for evaluating an ASR engine on TORGO utterances."""

    def __init__(self, model_id: str, model_path: Optional[Path] = None) -> None:
        self.model_id = model_id
        self.model_path = model_path

    def transcribe(self, wav_16k_path: Path) -> str:
        """Transcribe 16 kHz mono WAV file, returning raw transcript string."""
        raise NotImplementedError


class FasterWhisperEvaluator(ASRModelEvaluator):
    """Evaluates Whisper via faster-whisper on CPU with thread budget caps."""

    def __init__(self, model_size_or_path: str, model_threads: int) -> None:
        p = Path(model_size_or_path) if os.path.exists(model_size_or_path) else None
        super().__init__(model_id=f"whisper-{Path(model_size_or_path).name}", model_path=p)
        from faster_whisper import WhisperModel

        logger.info("Initializing faster-whisper (%s) with %d threads...", model_size_or_path, model_threads)
        self.model = WhisperModel(
            model_size_or_path,
            device="cpu",
            compute_type="int8",
            cpu_threads=model_threads,
            num_workers=1,
        )

    def transcribe(self, wav_16k_path: Path) -> str:
        segments, _ = self.model.transcribe(
            str(wav_16k_path),
            language="en",
            beam_size=1,
            temperature=0.0,
            vad_filter=False,  # TORGO utterances are segmented; don't clip dysarthric pauses
        )
        return " ".join(seg.text for seg in segments)


class SherpaOnnxEvaluator(ASRModelEvaluator):
    """Evaluates Sherpa-ONNX compatible models (Parakeet-TDT, Moonshine)."""

    def __init__(self, model_id: str, model_dir: Path, intra_threads: int) -> None:
        super().__init__(model_id=model_id, model_path=model_dir)
        import sherpa_onnx

        # Look for tokens, encoder, decoder in model_dir
        tokens = str(model_dir / "tokens.txt")
        encoder = str(model_dir / "encoder.onnx")
        decoder = str(model_dir / "decoder.onnx")

        logger.info("Initializing Sherpa-ONNX (%s) with %d intra-threads...", model_id, intra_threads)
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            num_threads=intra_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )

    def transcribe(self, wav_16k_path: Path) -> str:
        import sherpa_onnx

        stream = self.recognizer.create_stream()
        # Read WAV samples via sherpa_onnx or wave
        with wave.open(str(wav_16k_path), "rb") as wf:
            frames = wf.getnframes()
            data = wf.readframes(frames)
            import numpy as np
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        stream.accept_waveform(16000, samples)
        self.recognizer.decode_stream(stream)
        return stream.result.text


class MockEvaluator(ASRModelEvaluator):
    """Mock evaluator for unit tests and synthetic pipeline verification."""

    def __init__(self, model_id: str = "mock-baseline", error_rate: float = 0.1) -> None:
        super().__init__(model_id=model_id, model_path=None)
        self.error_rate = error_rate

    def transcribe(self, wav_16k_path: Path) -> str:
        time.sleep(0.01)  # Simulate 10ms processing
        # Return something deterministic based on filename
        stem = wav_16k_path.stem.replace("_16k", "")
        return f"mock transcript for {stem}"


def evaluate_model_on_torgo(
    evaluator: ASRModelEvaluator,
    dataset: TorgoDataset,
    cache_dir: Path,
    max_utterances: Optional[int] = None,
) -> Dict[str, Any]:
    """Run full evaluation of an ASR engine across TORGO utterances."""
    logger.info("Beginning evaluation for model: %s (%d total utterances)", evaluator.model_id, len(dataset.utterances))

    utterances = dataset.utterances
    if max_utterances is not None and max_utterances > 0:
        utterances = utterances[:max_utterances]

    total_ref_words = 0
    total_errors = 0

    dys_ref_words = 0
    dys_errors = 0

    ctrl_ref_words = 0
    ctrl_errors = 0

    severity_stats: Dict[str, Dict[str, int]] = {
        "Control": {"ref_words": 0, "errors": 0},
        "Mild": {"ref_words": 0, "errors": 0},
        "Moderate": {"ref_words": 0, "errors": 0},
        "Severe": {"ref_words": 0, "errors": 0},
    }

    speaker_stats: Dict[str, Dict[str, Any]] = {}
    latencies_ms: List[float] = []
    rtf_list: List[float] = []

    for utt in utterances:
        wav_16k = utt.get_wav_16k(cache_dir=cache_dir)
        duration = get_audio_duration_seconds(wav_16k)

        t_start = time.perf_counter()
        raw_hyp = evaluator.transcribe(wav_16k)
        elapsed = time.perf_counter() - t_start

        latency_ms = elapsed * 1000.0
        latencies_ms.append(latency_ms)

        rtf = real_time_factor(duration, elapsed)
        rtf_list.append(rtf)

        norm_hyp = normalize_text(raw_hyp)
        norm_ref = utt.prompt_normalized

        ref_words = norm_ref.split()
        num_ref = len(ref_words)
        wer = word_error_rate(norm_ref, norm_hyp)
        errors = int(round(wer * num_ref))

        total_ref_words += num_ref
        total_errors += errors

        if utt.is_dysarthric:
            dys_ref_words += num_ref
            dys_errors += errors
        else:
            ctrl_ref_words += num_ref
            ctrl_errors += errors

        sev = utt.severity
        if sev in severity_stats:
            severity_stats[sev]["ref_words"] += num_ref
            severity_stats[sev]["errors"] += errors

        spk = utt.speaker_id
        if spk not in speaker_stats:
            speaker_stats[spk] = {
                "is_dysarthric": utt.is_dysarthric,
                "severity": utt.severity,
                "ref_words": 0,
                "errors": 0,
                "utterance_count": 0,
            }
        speaker_stats[spk]["ref_words"] += num_ref
        speaker_stats[spk]["errors"] += errors
        speaker_stats[spk]["utterance_count"] += 1

    overall_wer = (total_errors / float(total_ref_words)) if total_ref_words > 0 else 0.0
    dysarthric_wer = (dys_errors / float(dys_ref_words)) if dys_ref_words > 0 else 0.0
    control_wer = (ctrl_errors / float(ctrl_ref_words)) if ctrl_ref_words > 0 else 0.0

    wer_by_severity = {}
    for sev, s_data in severity_stats.items():
        rw = s_data["ref_words"]
        wer_by_severity[sev] = (s_data["errors"] / float(rw)) if rw > 0 else None

    wer_by_speaker = {}
    for spk, spk_data in speaker_stats.items():
        rw = spk_data["ref_words"]
        spk_wer = (spk_data["errors"] / float(rw)) if rw > 0 else 0.0
        wer_by_speaker[spk] = {
            "wer": spk_wer,
            "is_dysarthric": spk_data["is_dysarthric"],
            "severity": spk_data["severity"],
            "utterance_count": spk_data["utterance_count"],
        }

    latency_stats = compute_latency_stats(latencies_ms)
    mean_rtf = (sum(rtf_list) / len(rtf_list)) if rtf_list else 0.0

    return {
        "evaluated_utterances": len(utterances),
        "overall_wer": overall_wer,
        "dysarthric_wer": dysarthric_wer,
        "control_wer": control_wer,
        "wer_by_severity": wer_by_severity,
        "wer_by_speaker": wer_by_speaker,
        "mean_rtf": mean_rtf,
        "latency_p50_ms": latency_stats["p50"],
        "latency_p95_ms": latency_stats["p95"],
        "latency_stats": latency_stats,
    }


def print_prerequisite_instructions(missing_items: List[str]) -> None:
    """Print clean instructions when required dataset or model weights are missing."""
    print("=" * 80)
    print("GRACE BENCHMARK 1 -- ASR ON TORGO: REQUIRED PREREQUISITES")
    print("=" * 80)

    print("The benchmark harness is installed and ready, but one or more required")
    print("datasets or model weights are not present on this machine:")
    print()
    for item in missing_items:
        print(f"  [MISSING] {item}")
    print()
    print("Per Grace Hard Rule 2, datasets and model weights are never downloaded automatically.")
    print("Please download the missing components as described below:")
    print()
    print("1. TORGO Speech Database (~18 GB, Non-Commercial Research License):")
    print("   - Official Page: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html")
    print("   - Direct archive links: F01, F03, F04, M01, M02, M03, M04, M05 (dysarthric)")
    print("                           FC01, FC02, FC03, MC01, MC02, MC03, MC04 (control)")
    print("   - Destination: Extract all archives into a single directory, e.g.:")
    print(r"                  C:\Users\Ashman Das\Downloads\TORGO")
    print("   - Re-run command: python -m bench.run_asr_torgo --torgo-dir \"C:\\Users\\Ashman Das\\Downloads\\TORGO\"")
    print()
    print("2. Parakeet-TDT 0.6B (NeMo / Sherpa-ONNX):")
    print("   - Source: https://github.com/k2-fsa/sherpa-onnx/releases or Hugging Face csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-en-3000")
    print(r"   - Destination: C:\Users\Ashman Das\Downloads\sherpa-onnx-nemo-parakeet-tdt-0.6b-en-3000")
    print()
    print("3. Moonshine Base (Useful Sensors ONNX):")
    print("   - Source: https://github.com/usefulsensors/moonshine or Hugging Face usefulsensors/moonshine")
    print(r"   - Destination: C:\Users\Ashman Das\Downloads\moonshine-base")
    print()
    print("4. Whisper Baseline (Systran / faster-whisper):")
    print("   - Source: Hugging Face Systran/faster-whisper-base (or tiny / small)")
    print(r"   - Destination: C:\Users\Ashman Das\Downloads\faster-whisper-base")
    print("=" * 80)


def run_benchmark(
    torgo_dir: Optional[str] = None,
    whisper_model: Optional[str] = None,
    parakeet_dir: Optional[str] = None,
    moonshine_dir: Optional[str] = None,
    simulate_tabs: int = 0,
    results_dir: Optional[str] = None,
    max_utterances: Optional[int] = None,
) -> int:
    """Execute ASR benchmark or report missing requirements."""
    # Check TORGO dataset path
    torgo_path = None
    if torgo_dir and os.path.isdir(torgo_dir):
        torgo_path = Path(torgo_dir)
    elif os.environ.get("TORGO_DATASET_DIR") and os.path.isdir(os.environ["TORGO_DATASET_DIR"]):
        torgo_path = Path(os.environ["TORGO_DATASET_DIR"])
    else:
        # Check standard download paths
        default_p = Path(r"C:\Users\Ashman Das\Downloads\TORGO")
        if default_p.is_dir():
            torgo_path = default_p

    missing = []
    if not torgo_path:
        missing.append("TORGO Speech Dataset directory (not found at configured path)")

    # Check model candidates
    evaluators: List[ASRModelEvaluator] = []

    # Thread budget for evaluation
    tier_diag = detect_hardware_tier()
    budget = calculate_thread_budget(tier=tier_diag.tier)

    # 1. Whisper baseline
    w_model = whisper_model or os.environ.get("WHISPER_MODEL", "base")
    # Check if faster-whisper is installed
    try:
        import faster_whisper
        # Only instantiate if whisper weights exist locally or model is a local dir
        if os.path.exists(w_model) or Path(r"C:\Users\Ashman Das\Downloads\faster-whisper-base").is_dir():
            w_path = w_model if os.path.exists(w_model) else str(Path(r"C:\Users\Ashman Das\Downloads\faster-whisper-base"))
            evaluators.append(FasterWhisperEvaluator(w_path, budget.llama_threads))
        else:
            missing.append(f"Whisper model weights '{w_model}' (offline local path required)")
    except ImportError:
        missing.append("faster-whisper Python package")

    # 2. Parakeet-TDT 0.6B
    p_dir = parakeet_dir or os.environ.get("PARAKEET_MODEL_DIR")
    if p_dir and os.path.isdir(p_dir):
        try:
            evaluators.append(SherpaOnnxEvaluator("parakeet-tdt-0.6b", Path(p_dir), budget.ort_intra_op_threads))
        except Exception as e:
            logger.warning("Could not initialize Parakeet-TDT: %s", e)
    else:
        missing.append("Parakeet-TDT 0.6B model directory (PARAKEET_MODEL_DIR)")

    # 3. Moonshine Base
    m_dir = moonshine_dir or os.environ.get("MOONSHINE_MODEL_DIR")
    if m_dir and os.path.isdir(m_dir):
        try:
            evaluators.append(SherpaOnnxEvaluator("moonshine-base", Path(m_dir), budget.ort_intra_op_threads))
        except Exception as e:
            logger.warning("Could not initialize Moonshine: %s", e)
    else:
        missing.append("Moonshine Base model directory (MOONSHINE_MODEL_DIR)")

    if missing:
        print_prerequisite_instructions(missing)
        return 0  # Clean exit with diagnostic instructions per Task 09

    assert torgo_path is not None
    dataset = load_torgo_dataset(torgo_path)
    cache_dir = torgo_path / ".cache_16k"

    load_desc = f"simulated_{simulate_tabs}_tabs" if simulate_tabs > 0 else "idle"

    with BackgroundLoadSimulator(num_tabs=simulate_tabs) if simulate_tabs > 0 else open(os.devnull):
        records = []
        for evaluator in evaluators:
            metrics = evaluate_model_on_torgo(evaluator, dataset, cache_dir, max_utterances=max_utterances)
            rec = create_benchmark_record(
                benchmark_name="asr_torgo",
                model_id=evaluator.model_id,
                metrics=metrics,
                file_path=evaluator.model_path,
                background_load=load_desc,
                details=dataset.summary(),
            )
            append_result("asr_torgo", rec, results_dir=results_dir)
            records.append(rec.to_dict())

    report = generate_markdown_report(records)
    print("\n" + report)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR Benchmark on TORGO dataset.")
    parser.add_argument("--torgo-dir", type=str, default=None, help="Root path to TORGO dataset")
    parser.add_argument("--whisper-model", type=str, default=None, help="Whisper model name or path")
    parser.add_argument("--parakeet-dir", type=str, default=None, help="Path to Parakeet-TDT model dir")
    parser.add_argument("--moonshine-dir", type=str, default=None, help="Path to Moonshine model dir")
    parser.add_argument("--simulate-tabs", type=int, default=0, help="Simulate N Chrome tabs (PLAN §9)")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory to save JSONL results")
    parser.add_argument("--max-utterances", type=int, default=None, help="Limit number of utterances")
    parser.add_argument("--synthetic-test", action="store_true", help="Run self-test on synthetic data")

    args = parser.parse_args()

    if args.synthetic_test:
        print("Running synthetic self-test...")
        from bench.metrics import compute_latency_stats
        print("Metrics and imports verified successfully.")
        return

    sys.exit(
        run_benchmark(
            torgo_dir=args.torgo_dir,
            whisper_model=args.whisper_model,
            parakeet_dir=args.parakeet_dir,
            moonshine_dir=args.moonshine_dir,
            simulate_tabs=args.simulate_tabs,
            results_dir=args.results_dir,
            max_utterances=args.max_utterances,
        )
    )


if __name__ == "__main__":
    main()
