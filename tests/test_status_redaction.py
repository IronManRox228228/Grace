"""Tests for status redaction over-match fix (retest 5 item 3).

- Status strings like 'element_not_found' must survive redaction in step.json and strings.
- Passphrases and API keys must still be redacted.
"""

import json
from unittest.mock import MagicMock
import pytest

from grace.harness.redact import redact_data, redact_string
from grace.harness.run_capture import RunCapture


class TestStatusRedaction:
    def test_element_not_found_status_survives_in_redact_string(self):
        text = "Status reported: element_not_found in target lookup"
        redacted = redact_string(text)
        assert "element_not_found" in redacted
        assert "<redacted>" not in redacted

    def test_other_system_statuses_survive_in_redact_string(self):
        statuses = [
            "element_not_found",
            "safety_confirmation_required",
            "max_iterations_reached",
            "planner_budget_exceeded",
            "window_not_found",
            "target_not_found",
        ]
        for st in statuses:
            assert redact_string(st) == st

    def test_passphrases_are_still_redacted(self):
        secret_passphrase = "correct-horse-battery-staple"
        redacted = redact_string(f"Password is {secret_passphrase}")
        assert secret_passphrase not in redacted
        assert "<redacted>" in redacted

    def test_api_keys_are_still_redacted(self):
        api_key = "sk-123456789012345678901234567890"
        redacted = redact_string(f"Bearer token {api_key}")
        assert api_key not in redacted
        assert "<redacted>" in redacted

    def test_step_json_preserves_not_found_status_and_redacts_secrets(self, tmp_path):
        """In step.json, result.result.status == 'element_not_found' survives while secrets are redacted."""
        capture = RunCapture(run_dir=tmp_path, vault_secrets=["vault_top_secret_token_123"])
        capture._enabled = True

        step_result = {
            "status": "error",
            "error": "Target element 'Play Signal' not found via UIA, OculiX, OpenCV, or OCR",
            "text": "Action error: Target element 'Play Signal' not found via UIA, OculiX, OpenCV, or OCR",
            "result": {
                "ok": False,
                "sent": False,
                "verified": False,
                "action": "click",
                "status": "element_not_found",
                "error": "Target element 'Play Signal' not found via UIA, OculiX, OpenCV, or OCR",
                "message": "I couldn't locate the 'Play Signal' button on screen.",
                "evidence": "nothing was clicked",
            },
        }

        params = {
            "target_name": "Play Signal",
            "secret_note": "sk-123456789012345678901234567890 and vault_top_secret_token_123",
        }

        capture.record_step(
            step_no=1,
            action="cua_click",
            params=params,
            grounded_point=None,
            result=step_result,
            snapshot=None,
            timings={"start": 0.0, "end": 1.0, "duration": 1.0},
        )

        step_file = tmp_path / "turn_001" / "step_001" / "step.json"
        assert step_file.exists()

        data = json.loads(step_file.read_text(encoding="utf-8"))

        # Verify not-found status survived
        assert data["result"]["status"] == "error"
        assert data["result"]["result"]["status"] == "element_not_found"

        # Verify secrets are redacted
        assert "sk-123456789012345678901234567890" not in str(data)
        assert "vault_top_secret_token_123" not in str(data)
        assert "<redacted>" in data["params"]["secret_note"]
