"""Tests for scripts/real_run.ps1 path resolution and wrapper logic."""

import re
from pathlib import Path

import pytest


def test_real_run_script_path_resolution_and_structure():
    """scripts/real_run.ps1 must resolve repo root correctly and wrap --capture-run."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "real_run.ps1"
    assert script_path.is_file(), f"Script not found at {script_path}"

    content = script_path.read_text(encoding="utf-8")

    # 1. Repo root must be resolved with Split-Path -Parent $PSScriptRoot once, NOT going up twice
    assert "Resolve-Path (Join-Path $scriptDir \"..\")" not in content, (
        "Script must not navigate up twice with Join-Path ..; it should resolve repo root once from $PSScriptRoot"
    )
    assert re.search(r"\$repoRoot\s*=\s*Split-Path\s+-Parent\s+\$PSScriptRoot", content, re.IGNORECASE), (
        "Script must define $repoRoot = Split-Path -Parent $PSScriptRoot"
    )

    # 2. grace.exe must be resolved relative to $repoRoot
    assert re.search(r"Join-Path\s+\$repoRoot\s+[\"']venv\\Scripts\\grace\.exe[\"']", content, re.IGNORECASE), (
        "Script must resolve grace.exe relative to $repoRoot"
    )

    # 3. No WebSocket client
    assert "ClientWebSocket" not in content, "Script should have no WebSocket client; run capture is in-process"
    assert "ws://127.0.0.1:8765" not in content, "Script should not connect to WebSocket"

    # 4. Runs grace.exe with --capture-run
    assert "--capture-run" in content, "Script must pass --capture-run to grace.exe"
    assert "--no-tts" in content, "Script must pass --no-tts to grace.exe"
    assert "--text" in content, "Script must pass --text to grace.exe"
