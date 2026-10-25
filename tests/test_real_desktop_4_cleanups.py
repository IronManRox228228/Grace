"""Tests for real-desktop-4 cleanups (section 3).

- Activation fallback: drop technique 4 (minimise/restore).
- scripts/diag_activate.py: handles U+200B and resets foreground.
- Capture log redaction: logger names are not redacted, while secret values remain redacted.
"""

import io
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from grace.automation.computer_use import ComputerUse
from grace.harness.redact import redact_string
from grace.harness.run_capture import RedactingFileHandler


class TestActivationFallbackTechniques:
    def test_technique_4_is_dropped_from_activation_sequence(self):
        """Technique 4 (minimize/restore) must not be tried during activation."""
        cu = ComputerUse()
        tried_techniques = []

        def mock_apply(hwnd, tech):
            tried_techniques.append(tech)

        cu._apply_activation_technique = mock_apply
        cu._verify_foreground = lambda hwnd, timeout_ms: (False, 0, "")

        cu._activate_hwnd(12345, "Test Window")

        assert 4 not in tried_techniques
        assert tried_techniques == [1, 2, 3, 5]

    def test_technique_1_succeeds_immediately(self):
        cu = ComputerUse()
        tried_techniques = []

        def mock_apply(hwnd, tech):
            tried_techniques.append(tech)

        cu._apply_activation_technique = mock_apply
        cu._verify_foreground = lambda hwnd, timeout_ms: (True, 12345, "Test Window")

        res = cu._activate_hwnd(12345, "Test Window")
        assert res["ok"] is True
        assert tried_techniques == [1]


class TestLoggerNameAndSecretRedaction:
    def test_logger_name_is_not_redacted(self, tmp_path):
        log_file = tmp_path / "test_log.txt"
        handler = RedactingFileHandler(filename=log_file, vault_secrets=["super_secret_vault_token"])

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)

        record = logging.LogRecord(
            name="grace.agent.loop",
            level=logging.INFO,
            pathname="loop.py",
            lineno=42,
            msg="AgentLoop step 2: processing user request",
            args=(),
            exc_info=None,
        )

        formatted = handler.format(record)
        assert "[INFO] grace.agent.loop:" in formatted
        assert "<redacted>" not in formatted

    def test_secret_value_is_still_redacted_in_log(self, tmp_path):
        log_file = tmp_path / "test_log.txt"
        handler = RedactingFileHandler(filename=log_file, vault_secrets=["my_vault_password_123"])

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)

        record = logging.LogRecord(
            name="grace.agent.loop",
            level=logging.INFO,
            pathname="loop.py",
            lineno=50,
            msg="Connecting using key sk-123456789012345678901234 and vault my_vault_password_123",
            args=(),
            exc_info=None,
        )

        formatted = handler.format(record)
        assert "[INFO] grace.agent.loop:" in formatted
        assert "sk-123456789012345678901234" not in formatted
        assert "my_vault_password_123" not in formatted
        assert "<redacted>" in formatted

    def test_redact_string_preserves_grace_logger_names(self):
        log_line = "2026-10-25 18:20:45,198 [INFO] grace.automation.computer_use: ComputerUse backend ready"
        redacted = redact_string(log_line)
        assert "grace.automation.computer_use" in redacted
        assert "<redacted>" not in redacted

    def test_redact_string_redacts_passphrases_and_keys(self):
        text = "Password is correct-horse-battery-staple and token is ghp_123456789012345678901234567890123456"
        redacted = redact_string(text)
        assert "correct-horse-battery-staple" not in redacted
        assert "ghp_123456789012345678901234567890123456" not in redacted
        assert "<redacted>" in redacted


class TestDiagActivateScript:
    def test_diag_activate_handles_zero_width_space(self, monkeypatch):
        from scripts.diag_activate import main

        test_args = ["diag_activate.py", "test"]
        monkeypatch.setattr(sys, "argv", test_args)

        # Mock find_target_window with title containing U+200B
        monkeypatch.setattr(
            "scripts.diag_activate.find_target_window",
            lambda query: (9999, "Title\u200bWithZeroWidth"),
        )
        monkeypatch.setattr(
            "scripts.diag_activate.verify_active",
            lambda target, delay_s: (True, 9999, "Title\u200bWithZeroWidth"),
        )
        monkeypatch.setattr("scripts.diag_activate.TECHNIQUES", [("Dummy Technique", lambda hwnd: None)])

        # Ensure main runs without UnicodeEncodeError
        main()
