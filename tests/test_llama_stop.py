"""`_stop_llama_server` must never run `taskkill /T` on a PID that has exited.

Windows reuses PIDs. After llama-server exits, its PID can belong to an
unrelated process, and `taskkill /F /T` would kill that process and its
whole tree.
"""

import subprocess
from unittest.mock import MagicMock, patch

from grace.main import GraceApp


def _app_with(proc):
    app = GraceApp.__new__(GraceApp)
    app._llama_process = proc
    return app


def _proc(alive):
    proc = MagicMock()
    proc.pid = 4242
    proc.poll.return_value = None if alive else 0
    return proc


def test_exited_process_is_never_taskkilled():
    proc = _proc(alive=False)
    app = _app_with(proc)
    with patch("grace.main.subprocess.run") as run:
        app._stop_llama_server()
    assert not run.called
    assert app._llama_process is None


def test_live_process_tree_is_killed_before_terminate():
    proc = _proc(alive=True)
    order = []
    proc.terminate.side_effect = lambda: order.append("terminate")
    app = _app_with(proc)
    with patch("grace.main.subprocess.run",
               side_effect=lambda args, **kw: order.append(args[0])) as run:
        app._stop_llama_server()
    assert run.call_args[0][0] == ["taskkill", "/F", "/T", "/PID", "4242"]
    assert order == ["taskkill", "terminate"]
    assert app._llama_process is None


def test_taskkill_failure_still_terminates():
    proc = _proc(alive=True)
    app = _app_with(proc)
    with patch("grace.main.subprocess.run",
               side_effect=subprocess.TimeoutExpired("taskkill", 10)):
        app._stop_llama_server()
    assert proc.terminate.called
    assert app._llama_process is None
