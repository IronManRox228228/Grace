"""Markdown report generator for benchmark results (PLAN.md §8, §9, Task 09).

Parses benchmark JSONL output files and formats clean, publication-ready GitHub
Flavored Markdown tables detailing WER across speakers and severities, alongside
RTF and p50/p95 CPU latency statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

# Ensure bench and src are in path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from bench.harness import DEFAULT_RESULTS_DIR, load_results


def format_wer(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100.0:.2f}%"


def format_ms(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f} ms"


def format_rtf(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val:.3f}x"


def generate_markdown_report(records: List[Dict[str, Any]]) -> str:
    """Generate Markdown report from benchmark execution records."""
    if not records:
        return (
            "# Benchmark Report: ASR on TORGO\n\n"
            "> [!NOTE]\n"
            "> No benchmark records found. Run `python -m bench.run_asr_torgo` after "
            "downloading the TORGO dataset to populate real metrics.\n"
        )

    lines: List[str] = []
    lines.append("# Benchmark Report: ASR on TORGO (PLAN §8, §9)\n")

    # Group by model_id
    models = sorted(list({r["model_id"] for r in records}))

    # 1. Executive Summary Table
    lines.append("## 1. Overall Performance & Latency\n")
    lines.append(
        "| Model | Overall WER | Dysarthric WER | Control WER | Mean RTF | Latency p50 | Latency p95 | Threads | Background Load |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
    )

    for r in records:
        m = r.get("metrics", {})
        threads = r.get("thread_settings", {}).get("model_threads", "N/A")
        load = r.get("machine_profile", {}).get("background_load", "idle")

        lines.append(
            f"| **{r['model_id']}** "
            f"| {format_wer(m.get('overall_wer'))} "
            f"| {format_wer(m.get('dysarthric_wer'))} "
            f"| {format_wer(m.get('control_wer'))} "
            f"| {format_rtf(m.get('mean_rtf'))} "
            f"| {format_ms(m.get('latency_p50_ms'))} "
            f"| {format_ms(m.get('latency_p95_ms'))} "
            f"| {threads} "
            f"| {load} |"
        )
    lines.append("")

    # 2. WER by Severity Breakdown Table
    lines.append("## 2. Accuracy by Severity Level (WER %)\n")
    lines.append("| Model | Control | Mild | Moderate | Severe |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for r in records:
        by_sev = r.get("metrics", {}).get("wer_by_severity", {})
        ctrl = format_wer(by_sev.get("Control"))
        mild = format_wer(by_sev.get("Mild"))
        mod = format_wer(by_sev.get("Moderate"))
        sev = format_wer(by_sev.get("Severe"))
        lines.append(f"| **{r['model_id']}** | {ctrl} | {mild} | {mod} | {sev} |")
    lines.append("")

    # 3. Per-Speaker WER Breakdown Table
    # Collect all unique speakers across records
    all_speakers: Dict[str, Dict[str, Any]] = {}
    for r in records:
        spk_data = r.get("metrics", {}).get("wer_by_speaker", {})
        for s_id, s_info in spk_data.items():
            if s_id not in all_speakers:
                all_speakers[s_id] = {
                    "severity": s_info.get("severity", "Unknown"),
                    "is_dysarthric": s_info.get("is_dysarthric", True),
                    "utterances": s_info.get("utterance_count", 0),
                }

    if all_speakers:
        lines.append("## 3. Per-Speaker WER Breakdown\n")
        header = "| Speaker | Group | Severity | Utterances | " + " | ".join(f"{m} WER" for m in models) + " |"
        delim = "| :--- | :---: | :---: | :---: | " + " | ".join(":---:" for _ in models) + " |"
        lines.append(header)
        lines.append(delim)

        for s_id in sorted(all_speakers.keys()):
            info = all_speakers[s_id]
            grp = "Dysarthric" if info["is_dysarthric"] else "Control"
            row = [s_id, grp, info["severity"], str(info["utterances"])]
            for m in models:
                rec = next((r for r in records if r["model_id"] == m), None)
                if rec:
                    spk_wer = rec.get("metrics", {}).get("wer_by_speaker", {}).get(s_id, {}).get("wer")
                    row.append(format_wer(spk_wer))
                else:
                    row.append("N/A")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # 4. Machine Profile & Environment Diagnostics
    latest_r = records[-1]
    mach = latest_r.get("machine_profile", {})
    runt = latest_r.get("runtime", {})
    t_set = latest_r.get("thread_settings", {})

    lines.append("## 4. Hardware Environment & Runtime Profile\n")
    lines.append(f"- **Operating System:** {mach.get('os', 'Unknown')}")
    lines.append(f"- **CPU:** {mach.get('cpu', 'Unknown')} ({mach.get('cpu_count_physical')} physical, {mach.get('cpu_count_logical')} logical cores)")
    lines.append(f"- **Memory:** {mach.get('ram_total_gb')} GB Total ({mach.get('ram_available_gb')} GB Available, {mach.get('memory_channels', 'unknown')})")
    lines.append(f"- **Selected Tier:** `{mach.get('tier_chosen', 'cpu')}` ({mach.get('tier_rationale', '')})")
    lines.append(f"- **Thread Budget:** {t_set.get('model_threads')} model threads ({t_set.get('rationale', '')})")
    lines.append(f"- **GPU Status:** Available={mach.get('gpu_available')}, Disabled={mach.get('gpu_disabled')}")
    lines.append(f"- **Runtimes:** Python {runt.get('python')}, ctranslate2={runt.get('ctranslate2')}, faster_whisper={runt.get('faster_whisper')}, sherpa_onnx={runt.get('sherpa_onnx')}")
    lines.append(f"- **Git Commit:** `{latest_r.get('git_commit', 'unknown')}`")
    lines.append(f"- **Timestamp:** {latest_r.get('timestamp', 'unknown')}\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark results as a Markdown table.")
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to results JSONL file (default: bench/results/asr_torgo.jsonl)",
    )
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default="asr_torgo",
        help="Benchmark name to load from bench/results/ (default: asr_torgo)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output markdown file path (if omitted, writes to stdout)",
    )
    args = parser.parse_args()

    if args.results_file:
        records = []
        p = Path(args.results_file)
        if p.is_file():
            import json
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
    else:
        records = load_results(args.benchmark_name)

    report_md = generate_markdown_report(records)

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report_md, encoding="utf-8")
        print(f"Report written to {out_p}")
    else:
        print(report_md)


if __name__ == "__main__":
    main()
