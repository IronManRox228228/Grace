"""Session tape recorder: the oracle the Rust port is graded against.

Enabled by ``GRACE_RECORD_DIR``. Each run writes one session directory::

    <GRACE_RECORD_DIR>/<session-id>/
        meta.json        run metadata: config snapshot, git sha, start time
        events.jsonl     every WsEventServer.emit() payload, with offsets
        llm.jsonl        every LLM exchange - prompt in, ordered chunks out
        stt.jsonl        every transcription - PCM digest in, text out
        pcm/<digest>.raw the raw int16 mono PCM for each transcription
        snapshots.jsonl  every perception snapshot (element graph)
        dispatch.jsonl   every tool dispatch - intent in, result out

Two design points that matter for the migration:

**LLM chunks are recorded individually, not joined.** ``response/generator.py``
splits sentences off the token stream *as it arrives*, and the ResponseChunk /
SpeechChunk pairs follow wherever those boundaries land. A Rust SSE reader with
different buffering would produce the same final text but a different chunk
sequence - a real UI and TTS regression that joining the stream at record time
would erase the evidence of.

**Timestamps are offsets from session start, not wall clock.** Tapes have to
diff cleanly across runs and machines. Absolute times never appear.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from typing import Any, Optional

logger = logging.getLogger("grace.harness.recorder")

ENV_RECORD_DIR = "GRACE_RECORD_DIR"


def _git_sha() -> Optional[str]:
    """The commit the tape was recorded at - a tape without one is unusable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class Recorder:
    """Append-only writer for one session tape.

    Every ``record_*`` method is best-effort: a recorder failure must never
    break a turn for the user, so all of them swallow their own exceptions and
    log instead. The tape is worth less than the session.
    """

    def __init__(self, root: str, session_id: Optional[str] = None):
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(root, self.session_id)
        self.pcm_dir = os.path.join(self.dir, "pcm")
        os.makedirs(self.pcm_dir, exist_ok=True)

        # Continuous mic audio is ~32 KB/s, so a long session is a large blob.
        # It is on by default because it is the only source of VAD and
        # wake-word parity evidence, and it stops being recordable the moment
        # the Python audio stack is deleted. Set GRACE_RECORD_AUDIO=0 to skip
        # it when taping a session for its LLM and dispatch behaviour only.
        self.record_audio = os.getenv("GRACE_RECORD_AUDIO", "1").lower() not in (
            "0", "false", "no",
        )

        self._start = time.perf_counter()
        self._lock = threading.Lock()
        self._handles: dict[str, Any] = {}
        self._counters: dict[str, int] = {}

        logger.info(f"Recording session tape to {self.dir}")

    # -- lifecycle ---------------------------------------------------------

    def write_meta(self, config: Any, extra: Optional[dict] = None) -> None:
        """Record what the tape was produced by. Called once, at startup.

        *extra* is merged into the metadata. The generator uses it to stamp a
        tape as synthetic, which a reader must be able to tell at a glance: a
        scripted tape grades the routing and event logic it was written to
        exercise, and nothing about the real world it never touched.
        """
        fields = {}
        for name in dir(config):
            if name.startswith("_"):
                continue
            value = getattr(config, name, None)
            if isinstance(value, (str, int, float, bool)) or value is None:
                # Never let a key reach a tape that may be shared or committed.
                fields[name] = "<redacted>" if "api_key" in name else value

        self._write_json(
            os.path.join(self.dir, "meta.json"),
            {
                "session_id": self.session_id,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "git_sha": _git_sha(),
                "config": fields,
                **(extra or {}),
            },
        )

    def close(self) -> None:
        with self._lock:
            for handle in self._handles.values():
                try:
                    handle.close()
                except Exception:
                    pass
            self._handles.clear()

    # -- boundaries --------------------------------------------------------

    def record_event(self, event: dict) -> None:
        """Boundary 1: an event was emitted to the frontend."""
        self._append("events", {"event": event})

    def record_llm(
        self,
        *,
        kind: str,
        request: Any,
        chunks: Optional[list[str]] = None,
        response: Any = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Boundary 2: one LLM exchange.

        *request* is recorded verbatim - the exact prompt text, not a summary.
        Prompt-string byte equality between Python and Rust is one of the
        migration's hardest gates and cannot be checked against a paraphrase.
        """
        record: dict[str, Any] = {"kind": kind, "request": request}
        if chunks is not None:
            record["chunks"] = chunks
        if response is not None:
            record["response"] = response
        if error is not None:
            record["error"] = error
        if duration_ms is not None:
            record["duration_ms"] = round(duration_ms, 1)
        record["request_digest"] = llm_request_digest(request)
        self._append("llm", record)

    def record_stt(self, pcm: bytes, text: str, duration_ms: Optional[float] = None) -> None:
        """Boundary 3: raw PCM in, transcript out. The PCM is the replay input."""
        digest = _digest(pcm)
        path = os.path.join(self.pcm_dir, f"{digest}.raw")
        try:
            if not os.path.exists(path):
                with open(path, "wb") as fh:
                    fh.write(pcm)
        except Exception as exc:
            logger.warning(f"Recorder: failed to write PCM {digest}: {exc}")

        record: dict[str, Any] = {"pcm_digest": digest, "pcm_bytes": len(pcm), "text": text}
        if duration_ms is not None:
            record["duration_ms"] = round(duration_ms, 1)
        self._append("stt", record)

    def record_audio_chunk(self, chunk: bytes) -> None:
        """Boundary 6: one PCM chunk, as the consumer received it.

        Recorded at the pump's ``get()`` - the consumer side - rather than at
        the producer, because what has to be reproducible is the chunk sequence
        the VAD and the wake-word detector actually saw, drops included.

        This is the only way VAD and wake-word parity can be checked at all.
        The VAD's turn-end has to fire on the *same chunk index*, not merely
        "eventually": a threshold that is slightly off ends turns a chunk or
        two late, which is inaudible in a demo and truncates real speech. The
        concatenated utterance stored by record_stt cannot show that, and the
        chunk timing is unrecoverable once the Python audio stack is gone.

        Chunks go to one append-only blob with a JSONL index rather than one
        file each: a ten-minute session is ~19k chunks.
        """
        try:
            with self._lock:
                blob = self._handles.get("audio.raw")
                if blob is None:
                    blob = open(os.path.join(self.dir, "audio.raw"), "ab")
                    self._handles["audio.raw"] = blob
                byte_offset = blob.tell()
                blob.write(chunk)
                blob.flush()
        except Exception as exc:
            logger.warning(f"Recorder: failed to append audio chunk: {exc}")
            return

        self._append("audio", {"offset": byte_offset, "bytes": len(chunk)})

    def record_snapshot(self, snapshot: Any) -> None:
        """Boundary 4: a perception snapshot (element graph).

        Element IDs are positional - assigned by index after overlap dedup - so
        the whole serialized node list, in order, is the thing that must match.
        """
        self._append("snapshots", {"snapshot": _jsonable(snapshot)})

    def record_dispatch(self, tool: str, params: Any, result: Any) -> None:
        """Boundary 5: a tool dispatch. In replay this is an assertion, not a call."""
        self._append(
            "dispatch",
            {"tool": tool, "params": _jsonable(params), "result": _jsonable(result)},
        )

    # -- internals ---------------------------------------------------------

    def _append(self, stream: str, record: dict) -> None:
        try:
            with self._lock:
                index = self._counters.get(stream, 0)
                self._counters[stream] = index + 1
                handle = self._handles.get(stream)
                if handle is None:
                    handle = open(
                        os.path.join(self.dir, f"{stream}.jsonl"),
                        "a", encoding="utf-8", newline="\n",
                    )
                    self._handles[stream] = handle

                handle.write(
                    json.dumps(
                        {"i": index, "t_ms": self._offset_ms(), **record},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
        except Exception as exc:
            logger.warning(f"Recorder: failed to append to {stream}: {exc}")

    def _offset_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000.0, 1)

    @staticmethod
    def _write_json(path: str, payload: dict) -> None:
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Recorder: failed to write {path}: {exc}")


# The fields of an LLM request that determine what the model produces. The
# digest covers exactly these, so a replay can find the recorded response.
#
# "backend" and "model" are recorded for diagnostics but deliberately excluded:
# they say which endpoint served the request, not what was asked. Including
# them would make every tape unreplayable the moment the request came from
# somewhere else - which is precisely the situation the Rust port creates.
LLM_REQUEST_KEY_FIELDS = ("messages", "temperature", "max_tokens", "stream")


def llm_request_digest(request: Any) -> str:
    """Stable digest over the prompt-determining fields of an LLM request."""
    if isinstance(request, dict):
        key = {field: request.get(field) for field in LLM_REQUEST_KEY_FIELDS}
    else:
        key = request
    return _digest(_canonical(key))


def _canonical(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False).encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of arbitrary backend objects to JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": len(value), "digest": _digest(bytes(value))}
    for attr in ("to_dict", "_asdict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return _jsonable(method())
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        return {
            k: _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")
        }
    return repr(value)


_recorder: Optional[Recorder] = None
_recorder_lock = threading.Lock()
_resolved = False


def get_recorder() -> Optional[Recorder]:
    """The active recorder, or None when ``GRACE_RECORD_DIR`` is unset.

    Resolved once. Recording is a whole-session decision; flipping it mid-run
    would produce a tape with a hole in it, which is worse than no tape.
    """
    global _recorder, _resolved
    if _resolved:
        return _recorder

    with _recorder_lock:
        if _resolved:
            return _recorder
        root = os.getenv(ENV_RECORD_DIR, "").strip()
        if root:
            try:
                _recorder = Recorder(root)
            except Exception as exc:
                logger.error(f"Recorder: could not start ({exc}); continuing unrecorded")
                _recorder = None
        _resolved = True
    return _recorder


def install_recorder(recorder: Optional[Recorder]) -> None:
    """Install a recorder directly, bypassing environment resolution.

    ``GRACE_RECORD_DIR`` names a root and lets the Recorder pick a timestamped
    session id, which is right for a live session and wrong for a generated
    one: a scenario's tape should live in a directory named after the scenario,
    so a corpus is readable and a regenerated tape replaces its predecessor
    instead of accumulating beside it.
    """
    global _recorder, _resolved
    with _recorder_lock:
        _recorder = recorder
        _resolved = True


def reset_recorder_for_tests() -> None:
    """Drop the memoized recorder so a test can re-resolve the environment."""
    global _recorder, _resolved
    with _recorder_lock:
        if _recorder is not None:
            _recorder.close()
        _recorder = None
        _resolved = False
