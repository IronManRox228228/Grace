"""Simulated background load generator (PLAN.md §9, Task 09).

Simulates N Chrome-like tabs by allocating controllable memory and intermittent
CPU spikes without requiring an actual browser. Used to benchmark Grace under
realistic multitasking load conditions.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any, List, Optional

import psutil

logger = logging.getLogger(__name__)


class BackgroundLoadSimulator:
    """Simulates resource consumption of N background browser tabs.

    Attributes:
        num_tabs: Number of tabs to simulate (default: 20 per PLAN §9).
        mb_per_tab: Approximate heap memory in MB allocated per tab (default: 50.0).
        cpu_load_pct: Target CPU activity percentage across background threads (default: 5.0).
    """

    def __init__(
        self,
        num_tabs: int = 20,
        mb_per_tab: float = 50.0,
        cpu_load_pct: float = 5.0,
    ) -> None:
        self.num_tabs = max(0, num_tabs)
        self.mb_per_tab = max(1.0, mb_per_tab)
        self.cpu_load_pct = min(max(0.0, cpu_load_pct), 50.0)

        self._stop_event = threading.Event()
        self._cpu_threads: List[threading.Thread] = []
        self._memory_blocks: List[bytearray] = []
        self._allocated_mb: float = 0.0
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def allocated_mb(self) -> float:
        return self._allocated_mb

    def start(self) -> None:
        """Allocate memory blocks and launch background simulated activity threads."""
        if self._is_running or self.num_tabs == 0:
            return

        self._stop_event.clear()

        # Check safety: preserve at least 1.5 GB of free host RAM
        try:
            mem = psutil.virtual_memory()
            free_mb = mem.available / (1024 * 1024)
            requested_mb = self.num_tabs * self.mb_per_tab

            # Cap requested memory if host is tight
            safe_alloc_mb = max(0.0, min(requested_mb, free_mb - 1500.0))
            if safe_alloc_mb < requested_mb:
                logger.warning(
                    "BackgroundLoadSimulator: Capping allocation from %.1f MB to %.1f MB "
                    "to preserve system stability (Available: %.1f MB)",
                    requested_mb, safe_alloc_mb, free_mb
                )

            actual_tabs = int(safe_alloc_mb // self.mb_per_tab) if safe_alloc_mb > 0 else 0
            block_bytes = int(self.mb_per_tab * 1024 * 1024)

            # Allocate in 50MB tab chunks
            for _ in range(actual_tabs):
                # Allocate and touch pages
                block = bytearray(block_bytes)
                # Touch first and last bytes to ensure page commit
                block[0] = 1
                block[-1] = 1
                self._memory_blocks.append(block)

            self._allocated_mb = len(self._memory_blocks) * self.mb_per_tab
        except Exception as e:
            logger.error("Failed to allocate simulated tab memory: %s", e)
            self._allocated_mb = 0.0

        # Launch background CPU tick thread if cpu_load_pct > 0
        if self.cpu_load_pct > 0:
            thread = threading.Thread(
                target=self._cpu_worker,
                name="GraceBench-SimulatedTabWorker",
                daemon=True,
            )
            self._cpu_threads.append(thread)
            thread.start()

        self._is_running = True
        logger.info(
            "BackgroundLoadSimulator started: %d tabs (%.1f MB allocated, %.1f%% simulated CPU)",
            len(self._memory_blocks), self._allocated_mb, self.cpu_load_pct
        )

    def _cpu_worker(self) -> None:
        """Intermittently burn CPU cycles to simulate JavaScript timers/render tasks."""
        duty_cycle = self.cpu_load_pct / 100.0
        period = 0.1  # 100ms window
        active_duration = period * duty_cycle
        sleep_duration = max(0.001, period * (1.0 - duty_cycle))

        while not self._stop_event.is_set():
            t_start = time.perf_counter()
            # Active burn loop
            while (time.perf_counter() - t_start) < active_duration:
                # Small math calculation
                _ = sum(i * i for i in range(500))
                if self._stop_event.is_set():
                    break
            # Sleep remainder of period
            time.sleep(sleep_duration)

    def stop(self) -> None:
        """Release allocated memory and stop background threads."""
        if not self._is_running:
            return

        self._stop_event.set()
        for t in self._cpu_threads:
            t.join(timeout=1.0)
        self._cpu_threads.clear()

        # Free memory
        self._memory_blocks.clear()
        self._allocated_mb = 0.0
        gc.collect()

        self._is_running = False
        logger.info("BackgroundLoadSimulator stopped cleanly and memory released.")

    def __enter__(self) -> "BackgroundLoadSimulator":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
