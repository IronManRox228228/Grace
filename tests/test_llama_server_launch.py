"""Tests for llama-server launch args and dead-process health checks (real-desktop.md #6)."""

import asyncio
import collections
import dataclasses
import time
from unittest.mock import MagicMock

import pytest

from grace.config import Config
from grace.main import GraceApp, build_llama_server_args
from grace.llm.model_swap import LocalModelSpec, ModelSwapManager


def test_build_llama_server_args_emits_fa_on():
    """llama.cpp requires '-fa on|off|auto', bare '-fa' swallows following flags."""
    cfg = dataclasses.replace(
        Config(),
        llama_flash_attn=True,
        llama_mmproj_path="C:/path/to/mmproj.gguf",
    )
    args = build_llama_server_args(cfg)
    assert "-fa" in args
    fa_idx = args.index("-fa")
    assert args[fa_idx + 1] == "on"
    assert args[fa_idx + 2] == "--mmproj"


def test_model_swap_manager_emits_fa_on():
    """ModelSwapManager must also pass '-fa on' when flash_attn is enabled."""
    spec = LocalModelSpec(
        name="holo",
        model_path="C:/path/to/model.gguf",
        mmproj_path="C:/path/to/mmproj.gguf",
        flash_attn=True,
    )
    mgr = ModelSwapManager(
        host="127.0.0.1",
        port=8080,
        specs={"holo": spec},
        llama_server_exe="C:/path/to/llama-server.exe",
    )
    args = mgr._build_args(spec)
    assert "-fa" in args
    fa_idx = args.index("-fa")
    assert args[fa_idx + 1] == "on"


@pytest.mark.asyncio
async def test_wait_for_llama_server_dead_process_fails_within_2s(caplog):
    """When llama-server dies on startup, health wait must fail fast (<2s) and log stderr."""
    app = GraceApp.__new__(GraceApp)
    app.config = Config()
    proc = MagicMock()
    proc.poll.return_value = 1  # process exited with error
    app._llama_process = proc
    app._llama_stderr_tail = collections.deque(
        [f"fatal error line {i}" for i in range(35)], maxlen=30
    )

    start = time.monotonic()
    result = await app._wait_for_llama_server(timeout=10)
    elapsed = time.monotonic() - start

    assert result is False
    assert elapsed < 2.0
    assert "fatal error line 34" in caplog.text
    # Should only have logged the last 30 lines (line 0 to 4 pushed out)
    assert "fatal error line 0" not in caplog.text


@pytest.mark.asyncio
async def test_model_swap_dead_process_fails_within_2s():
    """ModelSwapManager health wait fails fast (<2s) when process exits."""
    mgr = ModelSwapManager(
        host="127.0.0.1",
        port=8080,
        specs={},
        llama_server_exe="llama-server.exe",
        health_timeout=10.0,
    )
    proc = MagicMock()
    proc.poll.return_value = 1
    mgr._process = proc
    mgr._stderr_tail = collections.deque(["error line 1"], maxlen=30)

    start = time.monotonic()
    ok = await mgr._wait_for_health()
    elapsed = time.monotonic() - start

    assert ok is False
    assert elapsed < 2.0
