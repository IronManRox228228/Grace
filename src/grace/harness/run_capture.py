"""In-process run capture for real-desktop and evaluation runs.

Provides `--capture-run [DIR]` logging and structured step artifacts
(screenshots, metadata, planner prompts/raw text, step outcomes, summary).
Applies privacy redaction to all outputs and never captures vault secrets.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional, Union

from grace.harness.redact import redact_data, redact_string

# Minimal valid 1x1 PNG bytes for fallback / mocks without real screenshot bytes
_MINIMAL_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?"
    b"\x00\x05\xfe\x02\xfe\xa74e\xed\x00\x00\x00\x00IEND\xaeB`\x82"
)


def get_git_commit() -> str:
    """Return the short HEAD git commit hash if available, or an empty string."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            text=True,
        ).strip()
        return out if out else ""
    except Exception:
        return ""


def load_vault_secret_values() -> list[str]:
    """Retrieve all secret values currently stored in the DPAPI vault."""
    try:
        from grace.agent.memory import PersistentMemoryStore

        store = PersistentMemoryStore()
        secrets = store.list_secrets()
        vals = []
        for s in secrets:
            sec = store.get_secret(s["name"])
            if sec and sec.get("value"):
                vals.append(str(sec["value"]))
        store.close()
        return vals
    except Exception:
        return []


class RedactingFileHandler(logging.FileHandler):
    """FileHandler that applies redaction and strips vault secrets from formatted records."""

    def __init__(
        self,
        filename: Union[str, Path],
        mode: str = "a",
        encoding: str = "utf-8",
        delay: bool = False,
        vault_secrets: Optional[list[str]] = None,
    ):
        super().__init__(str(filename), mode=mode, encoding=encoding, delay=delay)
        self.vault_secrets = [v for v in (vault_secrets or []) if v]

    def format(self, record: logging.LogRecord) -> str:
        orig_name = record.name
        token = f"__LOGGER_NAME_{id(record)}__"
        record.name = token
        try:
            formatted = super().format(record)
            formatted = redact_string(formatted)
            for s in self.vault_secrets:
                if s and s in formatted:
                    formatted = formatted.replace(s, "<redacted>")
            formatted = formatted.replace(token, orig_name)
            return formatted
        finally:
            record.name = orig_name


class RunCapture:
    """Manages in-process run capture hooks and file outputs.

    When disabled or initialized without a directory, all methods are zero-overhead no-ops.
    """

    def __init__(
        self,
        run_dir: Optional[Union[str, Path]] = None,
        enabled: bool = True,
        vault_secrets: Optional[list[str]] = None,
    ):
        self.enabled = bool(enabled and run_dir)
        self.run_dir = Path(run_dir) if run_dir else None
        self._file_handler: Optional[RedactingFileHandler] = None
        # One entry per agent loop / goal: {"turn", "goal", "status", "step_count", ...}.
        # Step artifacts live in run_dir/turn_NNN/step_NNN so a new task in the
        # same session never overwrites the previous task's steps.
        self._turns: list[dict[str, Any]] = []
        self._turn_has_steps = False

        if self.enabled:
            if vault_secrets is not None:
                self.vault_secrets = [v for v in vault_secrets if v]
            else:
                self.vault_secrets = load_vault_secret_values()
        else:
            self.vault_secrets = []

    @property
    def is_enabled(self) -> bool:
        return bool(self.enabled and self.run_dir)

    @property
    def has_summary(self) -> bool:
        """True when the current turn already has its summary recorded."""
        return bool(self._turns and self._turns[-1].get("_summarised"))

    def begin_turn(self, goal: str = "") -> int:
        """Start a new turn (one agent loop or goal) and return its number.

        A turn that was just begun and has no steps or summary yet is reused,
        so the router and the agent loop can both announce the same goal.
        """
        if self._turns:
            cur = self._turns[-1]
            if (
                not self._turn_has_steps
                and not cur.get("_summarised")
                and cur.get("goal") in ("", goal)
            ):
                cur["goal"] = goal or cur.get("goal", "")
                return cur["turn"]
        number = len(self._turns) + 1
        self._turns.append({"turn": number, "goal": goal or ""})
        self._turn_has_steps = False
        return number

    def _turn_dir(self) -> Path:
        """Directory of the current turn (turn 1 is started on demand)."""
        if not self._turns:
            self.begin_turn("")
        return self.run_dir / f"turn_{self._turns[-1]['turn']:03d}"

    def _step_dir(self, step_no: int) -> Path:
        self._turn_has_steps = True
        return self._turn_dir() / f"step_{step_no:03d}"

    def _redact_str(self, text: str) -> str:
        if not text:
            return text
        out = redact_string(text)
        for s in self.vault_secrets:
            if s and s in out:
                out = out.replace(s, "<redacted>")
        return out

    def _redact(self, data: Any) -> Any:
        redacted = redact_data(data)
        if self.vault_secrets:

            def _strip(val: Any) -> Any:
                if isinstance(val, str):
                    for s in self.vault_secrets:
                        if s and s in val:
                            val = val.replace(s, "<redacted>")
                    return val
                if isinstance(val, dict):
                    return {k: _strip(v) for k, v in val.items()}
                if isinstance(val, list):
                    return [_strip(v) for v in val]
                if isinstance(val, tuple):
                    return tuple(_strip(v) for v in val)
                return val

            redacted = _strip(redacted)
        return redacted

    def start_logging(self) -> None:
        """Attach a FileHandler to the root logger writing to log.txt."""
        if not self.is_enabled or self._file_handler is not None:
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.run_dir / "log.txt"
        self._file_handler = RedactingFileHandler(
            log_path,
            mode="a",
            encoding="utf-8",
            vault_secrets=self.vault_secrets,
        )
        self._file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        self._file_handler.setFormatter(formatter)
        root = logging.getLogger()
        if root.level > logging.INFO or root.level == 0:
            root.setLevel(logging.INFO)
        root.addHandler(self._file_handler)

    def record_intent(
        self,
        raw_output: str,
        intent: Any,
        route: str,
    ) -> None:
        """Write intent.json containing raw LLM output, parsed intent, and route."""
        if not self.is_enabled:
            return

        self._turn_dir().mkdir(parents=True, exist_ok=True)
        intent_dict: Any = None
        if hasattr(intent, "to_dict"):
            intent_dict = intent.to_dict()
        elif hasattr(intent, "__dict__"):
            intent_dict = {
                "tool": getattr(intent, "tool", None),
                "params": getattr(intent, "params", None),
                "response": getattr(intent, "response", None),
            }
        elif isinstance(intent, dict):
            intent_dict = intent
        else:
            intent_dict = str(intent)

        data = {
            "raw_output": raw_output or "",
            "parsed_intent": intent_dict,
            "route": str(route or ""),
        }
        redacted = self._redact(data)
        out_path = self._turn_dir() / "intent.json"
        out_path.write_text(json.dumps(redacted, indent=2, default=str), encoding="utf-8")

    def record_screenshot(
        self,
        step_no: int,
        snapshot: Any,
        view: Any = None,
    ) -> None:
        """Write turn_NNN/step_NNN/screenshot.png and screenshot_meta.json."""
        if not self.is_enabled:
            return

        step_dir = self._step_dir(step_no)
        step_dir.mkdir(parents=True, exist_ok=True)

        width = getattr(snapshot, "width", 1920) or 1920
        height = getattr(snapshot, "height", 1080) or 1080
        image_width = getattr(snapshot, "image_width", 0) or width
        image_height = getattr(snapshot, "image_height", 0) or height
        scale = float(image_width) / float(width) if width else 1.0

        meta = {
            "original_size": [width, height],
            "sent_size": [image_width, image_height],
            "scale": scale,
        }
        (step_dir / "screenshot_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        # Image bytes: exact image sent to planner (or downscaled snapshot)
        png_bytes: Optional[bytes] = None
        if view and getattr(view, "image_b64", None):
            try:
                import base64

                png_bytes = base64.b64decode(view.image_b64)
            except Exception:
                png_bytes = None

        if png_bytes is None and snapshot and getattr(snapshot, "png_bytes", None):
            png_bytes = snapshot.png_bytes

        if not png_bytes:
            png_bytes = _MINIMAL_PNG_BYTES

        (step_dir / "screenshot.png").write_bytes(png_bytes)

    def record_planner(
        self,
        step_no: int,
        prompt: str,
        raw: str,
    ) -> None:
        """Write step_NNN/planner_prompt.txt and step_NNN/planner_raw.txt."""
        if not self.is_enabled:
            return

        step_dir = self._step_dir(step_no)
        step_dir.mkdir(parents=True, exist_ok=True)

        (step_dir / "planner_prompt.txt").write_text(
            self._redact_str(prompt or ""), encoding="utf-8"
        )
        (step_dir / "planner_raw.txt").write_text(
            self._redact_str(raw or ""), encoding="utf-8"
        )

    def record_step(
        self,
        step_no: int,
        action: str,
        params: dict,
        grounded_point: Optional[tuple[int, int] | list[int]],
        result: Any,
        snapshot: Any,
        timings: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write step_NNN/step.json with action, params, click point, result, elements, window, timings."""
        if not self.is_enabled:
            return

        step_dir = self._step_dir(step_no)
        step_dir.mkdir(parents=True, exist_ok=True)

        elements = []
        if snapshot and getattr(snapshot, "graph", None) and hasattr(snapshot.graph, "elements"):
            elements = snapshot.graph.elements
        elif snapshot and getattr(snapshot, "ui_elements", None):
            elements = snapshot.ui_elements

        sources: dict[str, int] = {}
        for el in (elements or []):
            src = getattr(el, "source", None) or (
                el.get("source") if isinstance(el, dict) else "unknown"
            )
            src_str = str(src or "unknown")
            sources[src_str] = sources.get(src_str, 0) + 1

        act_win = getattr(snapshot, "active_window", None) if snapshot else None
        hwnd = getattr(act_win, "hwnd", None) if act_win else None
        title = getattr(act_win, "title", "") if act_win else ""

        clean_params = {
            k: v for k, v in (params or {}).items() if not str(k).startswith("_")
        }

        step_data = {
            "action": action,
            "params": clean_params,
            "grounded_point": list(grounded_point) if grounded_point is not None else None,
            "result": result or {},
            "element_count": len(elements or []),
            "element_sources": sources,
            "window_title": title,
            "hwnd": hwnd,
            "timings": timings or {},
        }
        redacted = self._redact(step_data)
        (step_dir / "step.json").write_text(
            json.dumps(redacted, indent=2, default=str), encoding="utf-8"
        )

    def record_summary(
        self,
        goal: str,
        status: str,
        step_count: int,
        total_time: float,
        model: str = "",
        tier: str = "",
        git_commit: Optional[str] = None,
    ) -> None:
        """Write summary.json with goal, final status, step count, total time, model, tier, git commit."""
        if not self.is_enabled:
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)
        commit = git_commit if git_commit is not None else get_git_commit()

        if not self._turns:
            self.begin_turn(goal)
        turn = self._turns[-1]
        turn.update(
            {
                "goal": goal,
                "status": status,
                "step_count": step_count,
                "total_time": round(total_time, 3) if isinstance(total_time, float) else total_time,
                "_summarised": True,
            }
        )

        summary_data = {
            "goal": goal,
            "status": status,
            "step_count": step_count,
            "total_time": round(total_time, 3) if isinstance(total_time, float) else total_time,
            "model": model or "",
            "tier": tier or "",
            "git_commit": commit,
            "turns": [
                {k: v for k, v in t.items() if not k.startswith("_")} for t in self._turns
            ],
        }
        redacted = self._redact(summary_data)
        (self.run_dir / "summary.json").write_text(
            json.dumps(redacted, indent=2, default=str), encoding="utf-8"
        )

    def close(self) -> None:
        """Flush and remove logging handler."""
        if self._file_handler is not None:
            try:
                self._file_handler.flush()
                self._file_handler.close()
                logging.getLogger().removeHandler(self._file_handler)
            except Exception:
                pass
            self._file_handler = None
