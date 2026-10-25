"""Tests preventing silent NameErrors in perception and provider layers (real-desktop.md #1)."""

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest

from grace.perception.element_graph import ElementGraphBuilder, WindowRef


def test_no_undefined_names_in_src():
    """Fail if any undefined name (F821/F822/F823) exists in src/."""
    cmd = [sys.executable, "-m", "pyflakes", "src"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Filter for undefined names
    undefined_lines = [
        line for line in res.stdout.splitlines()
        if "undefined name" in line
    ]
    assert not undefined_lines, f"Found undefined names in src:\n" + "\n".join(undefined_lines)


def test_provider_name_error_reraises_in_tests():
    """A NameError in a perception provider must re-raise in tests, not be swallowed."""
    builder = ElementGraphBuilder()
    mock_uia = MagicMock()
    mock_uia.snapshot.side_effect = NameError("simulated missing import")
    builder._uia = mock_uia

    win = WindowRef(hwnd=123, title="Test Window", class_name="TestClass")
    with pytest.raises(NameError, match="simulated missing import"):
        builder.build(win)


def test_provider_failure_logs_warning_with_exception_type(caplog):
    """Provider failures must log at WARNING with the exception type, never silently at DEBUG."""
    import logging
    builder = ElementGraphBuilder()
    mock_uia = MagicMock()
    # RuntimeError is not NameError/ImportError/AttributeError so it won't re-raise
    mock_uia.snapshot.side_effect = RuntimeError("device disconnected")
    builder._uia = mock_uia

    win = WindowRef(hwnd=123, title="Test Window", class_name="TestClass")
    with caplog.at_level(logging.WARNING):
        graph = builder.build(win)

    assert len(graph) == 0
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("UIA snapshot failed [RuntimeError]: device disconnected" in r.message for r in warning_records)
