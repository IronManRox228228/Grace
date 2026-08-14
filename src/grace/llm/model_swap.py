"""Hot-swaps a single local llama-server process between two GGUF models.

Only exists for the fully-local setup (planner and grounder both on-device):
one GPU's VRAM budget can't hold both models loaded at once, so instead of
running two llama-server processes side by side, exactly one is ever
resident on the GPU. Swapping stops that process and starts the other model
in its place; the "resting" model has no process at all - its weights sit in
the OS page cache in system RAM until it's swapped back in, which is what
keeps the *next* load fast rather than a full re-read from disk.

Not used when the planner is cloud (Gemini): then only the grounder ever
needs the GPU, there is no contention, and llama-server just starts once.
"""

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

import aiohttp

logger = logging.getLogger("grace.model_swap")


@dataclass
class LocalModelSpec:
    """Everything needed to start llama-server for one role's model."""

    name: str
    model_path: str
    mmproj_path: Optional[str] = None
    context_window: int = 8192
    ngl: int = 999
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"


class ModelSwapManager:
    """Owns the single local llama-server process and swaps its model on demand."""

    def __init__(
        self,
        host: str,
        port: int,
        specs: dict[str, LocalModelSpec],
        llama_server_exe: str,
        health_timeout: float = 120.0,
    ):
        self._host = host
        self._port = port
        self._specs = specs
        self._exe = llama_server_exe
        self._health_timeout = health_timeout
        self._process: Optional[subprocess.Popen] = None
        self._active: Optional[str] = None
        # Serializes swap_to calls - the planner and grounder can each ask for
        # the GPU from different steps in close succession.
        self._lock = asyncio.Lock()

    @property
    def active(self) -> Optional[str]:
        return self._active

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    async def swap_to(self, name: str) -> bool:
        """Ensure `name`'s model is the one loaded on the GPU.

        A no-op if it already is. Callers are expected to call this before
        every request rather than once - the GPU may have been swapped away
        by the other role in between.
        """
        if name not in self._specs:
            raise KeyError(f"No local model spec registered for {name!r}")

        async with self._lock:
            if self._active == name and self._process is not None and self._process.poll() is None:
                return True
            logger.info(f"Swapping local model: {self._active!r} -> {name!r}")
            await self._stop()
            ok = await self._start(self._specs[name])
            self._active = name if ok else None
            return ok

    async def _stop(self) -> None:
        if self._process is None:
            return
        proc, self._process = self._process, None
        proc.terminate()
        try:
            await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("llama-server did not exit in time; killing")
            proc.kill()
            await asyncio.to_thread(proc.wait)

    async def _start(self, spec: LocalModelSpec) -> bool:
        args = [
            self._exe,
            "--host", self._host,
            "--port", str(self._port),
            "-m", spec.model_path,
            "-c", str(spec.context_window),
            "-ngl", str(spec.ngl),
            "--cache-type-k", spec.cache_type_k,
            "--cache-type-v", spec.cache_type_v,
        ]
        if spec.mmproj_path and os.path.exists(spec.mmproj_path):
            args += ["--mmproj", spec.mmproj_path]

        server_dir = os.path.dirname(self._exe)
        logger.info(f"Starting llama-server for {spec.name!r}: {spec.model_path}")
        self._process = subprocess.Popen(
            args,
            cwd=server_dir or None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return await self._wait_for_health()

    async def _wait_for_health(self) -> bool:
        """Poll /health, then confirm with a tiny completion once it's up.

        /health can return 200 while the model is still loading onto the
        GPU, so a real (if trivial) completion is what actually confirms the
        swap is usable.
        """
        deadline = asyncio.get_event_loop().time() + self._health_timeout
        async with aiohttp.ClientSession() as session:
            while asyncio.get_event_loop().time() < deadline:
                if self._process is not None and self._process.poll() is not None:
                    logger.error("llama-server exited during startup")
                    return False
                try:
                    async with session.get(
                        f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            payload = {
                                "messages": [{"role": "user", "content": "Hi"}],
                                "max_tokens": 1,
                                "temperature": 0.0,
                                "stream": False,
                            }
                            async with session.post(
                                f"{self.base_url}/v1/chat/completions",
                                json=payload,
                                timeout=aiohttp.ClientTimeout(total=5),
                            ) as ready_resp:
                                if ready_resp.status == 200:
                                    return True
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        logger.error(f"llama-server did not become healthy within {self._health_timeout}s")
        return False

    async def shutdown(self) -> None:
        await self._stop()
        self._active = None
