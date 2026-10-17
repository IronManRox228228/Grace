#!/usr/bin/env python3
"""Latency report generator for Grace turn timings.

Reads per-turn timing records from the local JSONL log and computes
p50 and p95 latencies for each stage:
  - Wake -> ASR Done
  - ASR Done -> First Model Token
  - First Model Token -> Action Done
  - Action Done -> Spoken Reply Done
  - End-to-end Turn Duration

Usage:
    python scripts/latency_report.py [--path PATH] [--json]
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Optional


def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile (0 <= p <= 100) of a non-empty float list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    if len(sorted_data) == 1:
        return sorted_data[0]
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


def load_records(path: str) -> list[dict[str, Any]]:
    """Load JSONL records from the specified path."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warning] line {line_no} malformed JSON: {exc}", file=sys.stderr)
    return records


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate p50, p95, min, max for each milestone and stage."""
    metrics: dict[str, list[float]] = {
        "wake_to_asr": [],
        "asr_to_first_token": [],
        "first_token_to_action": [],
        "action_to_speech": [],
        "total_turn": [],
    }

    stage_metrics: dict[str, list[float]] = {}

    for r in records:
        events = r.get("events") or {}
        stages = r.get("stages") or {}
        total = r.get("total_ms")

        if total is not None:
            metrics["total_turn"].append(float(total))

        # Milestones
        wake = events.get("wake", 0.0)
        asr_done = events.get("asr_done")
        first_token = events.get("first_model_token")
        action_done = events.get("action_done")
        speech_done = events.get("spoken_reply_done")

        if asr_done is not None:
            metrics["wake_to_asr"].append(max(0.0, asr_done - wake))
        if asr_done is not None and first_token is not None:
            metrics["asr_to_first_token"].append(max(0.0, first_token - asr_done))
        if first_token is not None and action_done is not None:
            metrics["first_token_to_action"].append(max(0.0, action_done - first_token))
        if action_done is not None and speech_done is not None:
            metrics["action_to_speech"].append(max(0.0, speech_done - action_done))

        for name, dur in stages.items():
            if dur is not None:
                stage_metrics.setdefault(name, []).append(float(dur))

    results: dict[str, Any] = {
        "turn_count": len(records),
        "milestones": {},
        "stages": {},
    }

    for key, values in metrics.items():
        if values:
            results["milestones"][key] = {
                "count": len(values),
                "p50_ms": round(_percentile(values, 50), 1),
                "p95_ms": round(_percentile(values, 95), 1),
                "min_ms": round(min(values), 1),
                "max_ms": round(max(values), 1),
            }

    for key, values in sorted(stage_metrics.items()):
        if values:
            results["stages"][key] = {
                "count": len(values),
                "p50_ms": round(_percentile(values, 50), 1),
                "p95_ms": round(_percentile(values, 95), 1),
                "min_ms": round(min(values), 1),
                "max_ms": round(max(values), 1),
            }

    return results


def print_table(analysis: dict[str, Any], log_path: str) -> None:
    """Print a clean formatted latency report."""
    count = analysis.get("turn_count", 0)
    print("=" * 78)
    print(f" Grace Latency Report ({count} recorded turn{'s' if count != 1 else ''})")
    print(f" Source: {log_path}")
    print("=" * 78)

    if count == 0:
        print("\nNo turn latency records found.")
        print("Run Grace and complete one or more voice turns to populate metrics.\n")
        return

    col_w = [34, 8, 10, 10, 8]
    header = (
        f"{'Milestone / Transition':<{col_w[0]}} "
        f"{'Count':>{col_w[1]}} "
        f"{'p50 (ms)':>{col_w[2]}} "
        f"{'p95 (ms)':>{col_w[3]}} "
        f"{'Max (ms)':>{col_w[4]}}"
    )
    print("\n[Core Milestone Transitions]")
    print("-" * 78)
    print(header)
    print("-" * 78)

    milestone_labels = {
        "wake_to_asr": "1. Wake -> ASR Done",
        "asr_to_first_token": "2. ASR Done -> First Token",
        "first_token_to_action": "3. First Token -> Action Done",
        "action_to_speech": "4. Action Done -> Spoken Reply",
        "total_turn": "Total Turn (End-to-End)",
    }

    for key, label in milestone_labels.items():
        data = analysis.get("milestones", {}).get(key)
        if not data:
            continue
        print(
            f"{label:<{col_w[0]}} "
            f"{data['count']:>{col_w[1]}} "
            f"{data['p50_ms']:>{col_w[2]}.1f} "
            f"{data['p95_ms']:>{col_w[3]}.1f} "
            f"{data['max_ms']:>{col_w[4]}.1f}"
        )

    stages = analysis.get("stages", {})
    if stages:
        print("\n[Detailed Subsystem Stages]")
        print("-" * 78)
        print(header)
        print("-" * 78)
        for name, data in stages.items():
            print(
                f"{name:<{col_w[0]}} "
                f"{data['count']:>{col_w[1]}} "
                f"{data['p50_ms']:>{col_w[2]}.1f} "
                f"{data['p95_ms']:>{col_w[3]}.1f} "
                f"{data['max_ms']:>{col_w[4]}.1f}"
            )

    print("=" * 78 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Grace turn latency report")
    parser.add_argument("--path", default=None, help="Path to latency.jsonl log")
    parser.add_argument("--json", action="store_true", help="Output report as JSON")
    args = parser.parse_args()

    path = args.path
    if not path:
        try:
            from grace.config import Config
            path = Config().latency_log_path
        except Exception:
            path = os.path.join(os.getenv("LOCALAPPDATA", "."), "Grace", "logs", "latency.jsonl")

    records = load_records(path)
    analysis = analyze_records(records)

    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        print_table(analysis, path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
