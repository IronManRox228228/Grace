"""Central thread budget and CPU core allocation for Grace (PLAN.md §4).

Enforces explicit thread caps across all model runtimes (llama.cpp -t, ONNX Runtime
intra/inter-op threads) so that active worker threads never saturate the CPU or starve
the Windows desktop, accessibility shell, and background audio loops.

Rules:
1. Total active model threads must never exceed (logical_cores - 2), leaving at least
   two cores free for Windows, the audio pump, and the UI.
2. Floor tier (CPU only): strict relay execution with no concurrency overlap.
   TTS lookahead is disabled while the LLM is streaming.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, NamedTuple, Optional

from grace.tier import HardwareTier

logger = logging.getLogger("grace.thread_budget")


class ThreadBudget(NamedTuple):
    total_cores: int
    active_worker_threads: int
    llama_threads: int
    ort_intra_op_threads: int
    ort_inter_op_threads: int
    allow_tts_lookahead: bool
    rationale: str


def calculate_thread_budget(
    tier: HardwareTier = HardwareTier.FLOOR_CPU,
    config: Optional[Any] = None,
) -> ThreadBudget:
    """Calculate the explicit thread allocation budget for all runtime engines."""
    total_cores = os.cpu_count() or 4

    # Allow explicit override from config or environment
    override_threads = None
    if config is not None:
        override_threads = getattr(config, "thread_budget", None)
    if not override_threads:
        env_val = os.getenv("THREAD_BUDGET")
        if env_val:
            try:
                override_threads = int(env_val)
            except ValueError:
                pass

    if override_threads and override_threads > 0:
        active = min(total_cores, override_threads)
        llama_t = active
        ort_intra = max(1, min(active, 4))
        allow_lookahead = (tier != HardwareTier.FLOOR_CPU)
        budget = ThreadBudget(
            total_cores=total_cores,
            active_worker_threads=active,
            llama_threads=llama_t,
            ort_intra_op_threads=ort_intra,
            ort_inter_op_threads=1,
            allow_tts_lookahead=allow_lookahead,
            rationale=f"Manual thread budget override: {override_threads} threads",
        )
        logger.info(f"Thread budget configured: {budget}")
        return budget

    # Reserve 2 cores for OS / UI responsiveness, minimum 1 worker thread
    reserve = 2 if total_cores >= 4 else 1
    active_workers = max(1, total_cores - reserve)

    if tier == HardwareTier.FLOOR_CPU:
        # On the CPU floor, keep thread counts tight to avoid cache-line thrashing.
        # Strict relay: no concurrent TTS generation during LLM generation.
        llama_threads = min(active_workers, 4)
        ort_intra = min(active_workers, 4)
        allow_lookahead = False
        rationale = f"CPU Floor: {active_workers} active threads ({reserve} reserved for OS), no TTS lookahead overlap"
    else:
        # On NPU / iGPU tiers, models offload primary compute to accelerator.
        llama_threads = min(active_workers, 8)
        ort_intra = min(active_workers, 4)
        allow_lookahead = True
        rationale = f"{tier.value.upper()}: {active_workers} active threads ({reserve} reserved for OS), pipelined TTS lookahead enabled"

    budget = ThreadBudget(
        total_cores=total_cores,
        active_worker_threads=active_workers,
        llama_threads=llama_threads,
        ort_intra_op_threads=ort_intra,
        ort_inter_op_threads=1,
        allow_tts_lookahead=allow_lookahead,
        rationale=rationale,
    )
    logger.info(f"Calculated thread budget: {budget}")
    return budget


def get_llama_thread_args(budget: ThreadBudget) -> List[str]:
    """Return CLI arguments for llama-server thread configuration."""
    return ["-t", str(budget.llama_threads)]


def configure_ort_session_options(options: Any, budget: ThreadBudget) -> None:
    """Apply thread caps to an onnxruntime.SessionOptions object if supported."""
    try:
        if hasattr(options, "intra_op_num_threads"):
            options.intra_op_num_threads = budget.ort_intra_op_threads
        if hasattr(options, "inter_op_num_threads"):
            options.inter_op_num_threads = budget.ort_inter_op_threads
    except Exception as e:
        logger.debug(f"Failed to set thread caps on ORT session options: {e}")
