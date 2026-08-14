"""The emit() tap: contract enforcement and recording at the one real seam.

These test WsEventServer.emit() itself rather than the recorder in isolation,
because the properties that matter are placement properties: the tap has to run
before the connected-clients check, and a contract violation must be fatal in
strict mode and survivable otherwise.
"""

import asyncio
import importlib
import json
import os

import pytest

from grace.harness.contract import ContractViolation
from grace.harness.recorder import reset_recorder_for_tests


@pytest.fixture
def ws_module(monkeypatch, tmp_path):
    """A freshly imported ws_server with recording on and strict mode off.

    ws_server reads GRACE_CONTRACT_STRICT at import time (it is a whole-process
    decision, like the recorder), so the module has to be reimported after the
    environment is set.
    """
    reset_recorder_for_tests()
    monkeypatch.setenv("GRACE_RECORD_DIR", str(tmp_path))
    monkeypatch.delenv("GRACE_CONTRACT_STRICT", raising=False)

    import grace.ws_server as ws_server
    module = importlib.reload(ws_server)
    yield module

    reset_recorder_for_tests()
    importlib.reload(ws_server)


def _events(recorder) -> list[dict]:
    with open(os.path.join(recorder.dir, "events.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line)["event"] for line in fh if line.strip()]


def test_events_are_taped_even_with_no_frontend_attached(ws_module):
    """A tape recorded headlessly must still be complete.

    The whole replay strategy depends on driving the backend with no UI, so the
    tap cannot sit behind the `if not self._clients: return` early exit.
    """
    from grace.harness.recorder import get_recorder

    server = ws_module.WsEventServer()
    assert not server.is_connected

    asyncio.run(server.emit({"type": "WakeWordDetected"}))
    asyncio.run(server.emit({"type": "FinalTranscript", "text": "open notepad"}))

    assert _events(get_recorder()) == [
        {"type": "WakeWordDetected"},
        {"type": "FinalTranscript", "text": "open notepad"},
    ]


def test_violation_is_logged_and_the_turn_survives(ws_module, caplog):
    """In production a malformed diagnostic event must never abort a turn."""
    server = ws_module.WsEventServer()

    with caplog.at_level("ERROR"):
        asyncio.run(server.emit({"type": "NotAThing"}))

    assert any("Contract violation" in record.message for record in caplog.records)


def test_violation_raises_under_strict_mode(monkeypatch, tmp_path):
    """The dev/CI gate: a contract bug fails loudly where it is cheap to fix."""
    reset_recorder_for_tests()
    monkeypatch.setenv("GRACE_CONTRACT_STRICT", "1")
    monkeypatch.delenv("GRACE_RECORD_DIR", raising=False)

    import grace.ws_server as ws_server
    module = importlib.reload(ws_server)
    try:
        server = module.WsEventServer()
        with pytest.raises(ContractViolation):
            asyncio.run(server.emit({"type": "ToolExecutionStarted"}))  # missing 'label'
    finally:
        monkeypatch.delenv("GRACE_CONTRACT_STRICT", raising=False)
        importlib.reload(ws_server)
        reset_recorder_for_tests()


def test_valid_events_pass_through_strict_mode(monkeypatch):
    reset_recorder_for_tests()
    monkeypatch.setenv("GRACE_CONTRACT_STRICT", "1")
    monkeypatch.delenv("GRACE_RECORD_DIR", raising=False)

    import grace.ws_server as ws_server
    module = importlib.reload(ws_server)
    try:
        server = module.WsEventServer()
        for event in (
            {"type": "Idle"},
            {"type": "UnderstandingStarted", "label": "Understanding request…"},
            {"type": "ToolExecutionFinished", "tool": "cua_click", "status": "ok"},
        ):
            asyncio.run(server.emit(event))
    finally:
        monkeypatch.delenv("GRACE_CONTRACT_STRICT", raising=False)
        importlib.reload(ws_server)
        reset_recorder_for_tests()
