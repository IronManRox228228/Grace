"""Task Memory and Data Scratchpad for Grace Agentic Loop.

Tracks execution history, intermediate results, extracted data,
and iteration bounds during multi-step autonomous tasks.
"""

import contextlib
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("grace.agent.memory")


class PersistentMemoryStore:
    """SQLite-backed cross-session persistent memory for Grace.

    Hardened with DPAPI (CryptProtectData, user scope) for at-rest encryption
    of memory.db (R20) and external salt storage in memory.salt (R23).
    Supports in-place migration of existing plaintext databases with backup.
    """

    def __init__(self, db_path: Optional[str] = None):
        import threading
        self._lock = threading.Lock()
        if not db_path:
            if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
                db_path = ":memory:"
            else:
                db_path = os.environ.get("GRACE_MEMORY_DB")
                if not db_path:
                    base_dir = os.path.expanduser("~/.grace")
                    os.makedirs(base_dir, exist_ok=True)
                    db_path = os.path.join(base_dir, "memory.db")
        self._db_path = db_path
        self._salt_path = os.path.splitext(db_path)[0] + ".salt" if db_path != ":memory:" else None
        self.salt = self._init_salt()
        self._use_dpapi = False
        self._init_dpapi_and_conn()
        self._init_db()

    def _init_salt(self) -> str:
        if not self._salt_path:
            import secrets
            return secrets.token_hex(32)
        if os.path.exists(self._salt_path):
            try:
                with open(self._salt_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
            except Exception as exc:
                logger.warning(f"Failed to read memory.salt ({exc}); regenerating")
        import secrets
        salt = secrets.token_hex(32)
        try:
            with open(self._salt_path, "w", encoding="utf-8") as f:
                f.write(salt)
        except Exception as exc:
            logger.warning(f"Failed to write memory.salt ({exc})")
        return salt

    def _init_dpapi_and_conn(self) -> None:
        if self._db_path == ":memory:":
            self._use_dpapi = False
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            return

        has_win32crypt = False
        try:
            import win32crypt
            has_win32crypt = True
        except ImportError:
            pass

        if not has_win32crypt:
            logger.warning("DPAPI protection unavailable (win32crypt missing); operating memory.db in plaintext mode")
            self._use_dpapi = False
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            return

        self._use_dpapi = True
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)

        if os.path.exists(self._db_path) and os.path.getsize(self._db_path) > 0:
            try:
                with open(self._db_path, "rb") as f:
                    raw_data = f.read()
                if raw_data.startswith(b"SQLite format 3\x00"):
                    # Existing plaintext database: migrate in place with .bak backup
                    import shutil
                    backup_path = self._db_path + ".bak"
                    shutil.copy2(self._db_path, backup_path)
                    logger.info("Migrated existing plaintext memory.db to DPAPI-protected format (backup at %s)", backup_path)
                    self._conn.deserialize(raw_data)
                    self._flush_encrypted()
                else:
                    try:
                        _, decrypted = win32crypt.CryptUnprotectData(raw_data, None, None, None, 1)
                        self._conn.deserialize(decrypted)
                    except Exception as exc:
                        logger.warning("Failed to decrypt DPAPI memory.db (%s); falling back to plaintext mode", exc)
                        self._use_dpapi = False
                        self._conn.close()
                        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            except Exception as exc:
                logger.warning("Error reading memory.db (%s); falling back to plaintext mode", exc)
                self._use_dpapi = False
                self._conn.close()
                self._conn = sqlite3.connect(self._db_path, check_same_thread=False)

    def _flush_encrypted(self) -> None:
        if not getattr(self, "_use_dpapi", False) or self._db_path == ":memory:":
            return
        try:
            import win32crypt
            serialized = self._conn.serialize()
            encrypted = win32crypt.CryptProtectData(serialized, "GraceMemory", None, None, None, 1)
            with open(self._db_path, "wb") as f:
                f.write(encrypted)
        except Exception as exc:
            logger.warning(f"Failed to flush encrypted memory.db: {exc}")

    def close(self):
        """Clean up memory store resources."""
        with self._lock:
            if getattr(self, "_use_dpapi", False):
                self._flush_encrypted()
            if hasattr(self, "_conn") and self._conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
        import gc
        gc.collect()

    def _init_db(self):
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS task_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_goal TEXT,
                            action TEXT,
                            params TEXT,
                            result TEXT,
                            timestamp REAL
                        )
                        """
                    )
                    self._conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS user_preferences (
                            key TEXT PRIMARY KEY,
                            value TEXT,
                            updated_at REAL
                        )
                        """
                    )
                self._flush_encrypted()
            except Exception as exc:
                logger.warning(f"Persistent memory DB init failed: {exc}")

    def save_step(self, user_goal: str, action: str, params: dict, result: dict) -> None:
        """Persist a completed action step to SQLite disk database."""
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        "INSERT INTO task_history (user_goal, action, params, result, timestamp) VALUES (?, ?, ?, ?, ?)",
                        (user_goal, action, json.dumps(params), json.dumps(result), time.time()),
                    )
                self._flush_encrypted()
            except Exception as exc:
                logger.debug(f"Failed to save step to persistent memory: {exc}")

    def set_preference(self, key: str, value: str) -> None:
        """Set a persistent user preference or remembered value."""
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO user_preferences (key, value, updated_at) VALUES (?, ?, ?)",
                        (key, value, time.time()),
                    )
                self._flush_encrypted()
            except Exception as exc:
                logger.debug(f"Failed to set preference in persistent memory: {exc}")

    def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a persistent user preference or remembered value."""
        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute("SELECT value FROM user_preferences WHERE key = ?", (key,))
                row = cursor.fetchone()
                return row[0] if row else default
            except Exception as exc:
                logger.debug(f"Failed to get preference from persistent memory: {exc}")
                return default

    def get_recent_history(self, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve recent task history across sessions."""
        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute(
                    "SELECT user_goal, action, params, result, timestamp FROM task_history ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
                rows = cursor.fetchall()
                return [
                    {
                        "user_goal": r[0],
                        "action": r[1],
                        "params": json.loads(r[2]),
                        "result": json.loads(r[3]),
                        "timestamp": r[4],
                    }
                    for r in rows
                ]
            except Exception as exc:
                logger.debug(f"Failed to get recent history from persistent memory: {exc}")
                return []

    def forget_goal(self, goal: str) -> int:
        """Forget all records matching a user goal (R17)."""
        deleted = 0
        with self._lock:
            try:
                with self._conn:
                    cursor = self._conn.cursor()
                    cursor.execute("DELETE FROM task_history WHERE user_goal = ?", (goal,))
                    deleted = cursor.rowcount
                self._flush_encrypted()
            except Exception as exc:
                logger.debug(f"Failed to forget goal from persistent memory: {exc}")
        return deleted

    def forget_everything(self) -> None:
        """Clear all stored task history and user preferences (R17)."""
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute("DELETE FROM task_history")
                    self._conn.execute("DELETE FROM user_preferences")
                self._flush_encrypted()
            except Exception as exc:
                logger.debug(f"Failed to wipe persistent memory: {exc}")


# How many established facts / ruled-out approaches to carry forward. They go
# into every subsequent prompt, so this is a token budget as much as a memory
# one; past a handful, the useful ones stop standing out.
MAX_REMEMBERED = 8


@dataclass
class StepRecord:
    """Record of a single action step in the agentic loop."""

    step_number: int
    thought: str
    action: str
    params: dict[str, Any]
    result: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    user_update: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "params": self.params,
            "result": self.result,
            "timestamp": self.timestamp,
            "user_update": self.user_update,
        }


class AgentMemory:
    """State memory for an autonomous task session with SQLite persistence."""

    def __init__(self, user_goal: str, max_iterations: int = 0,
                 persistent_store: Optional[PersistentMemoryStore] = None,
                 max_seconds: float = 0):
        # max_iterations <= 0 means no step limit at all.
        self.user_goal: str = user_goal
        self.max_iterations: int = max_iterations
        # The real ceiling. A step cap stops a task that is making progress
        # (and the shipped default is deliberately unlimited); a clock stops one
        # that is not, and it is the only bound the twelve-minute run would have
        # crossed - it stayed under any plausible step cap for its first several
        # minutes while achieving nothing.
        self.max_seconds: float = max_seconds
        self._started: float = time.monotonic()
        self._paused_at: Optional[float] = None
        self.steps_taken: list[StepRecord] = []
        self.scratchpad: dict[str, Any] = {}
        self.current_iteration: int = 0
        self.is_completed: bool = False
        self.final_response: Optional[str] = None
        self.safety_pending: Optional[dict[str, Any]] = None
        self.persistent_store: Optional[PersistentMemoryStore] = persistent_store or PersistentMemoryStore()

    def add_step(
        self,
        thought: str,
        action: str,
        params: dict[str, Any],
        result: dict[str, Any],
        user_update: Optional[str] = None,
    ) -> StepRecord:
        """Record an executed step and persist to SQLite disk store."""
        self.current_iteration += 1
        record = StepRecord(
            step_number=self.current_iteration,
            thought=thought,
            action=action,
            params=params,
            result=result,
            user_update=user_update,
        )
        self.steps_taken.append(record)
        if self.persistent_store:
            self.persistent_store.save_step(self.user_goal, action, params, result)
        return record

    def establish(self, fact: str) -> None:
        """Record something now known to be true, outside the 3-step window."""
        self._remember("established", fact)

    def rule_out(self, approach: str, why: str = "") -> None:
        """Record an approach that has been tried and did not work."""
        self._remember("ruled_out", f"{approach} - {why}" if why else approach)

    def _remember(self, key: str, entry: str) -> None:
        """Append to a rolling list, most recent last, without duplicates.

        `format_history_markdown` keeps the last three steps verbatim and
        discards everything before them with no summary, so a goal that has
        tried and rejected four approaches cannot see that it did - it re-plans
        from a window that has already forgotten. This is the minimum that
        survives the window, and it costs no model call: every entry is written
        from facts the loop has already computed.
        """
        entries = self.scratchpad.setdefault(key, [])
        if entry in entries:
            return
        entries.append(entry)
        # Bounded, because it goes into every subsequent prompt.
        del entries[:-MAX_REMEMBERED]

    def set_scratchpad(self, key: str, value: Any):
        """Store intermediate data (e.g. extracted text, search results)."""
        self.scratchpad[key] = value

    def get_scratchpad(self, key: str, default: Any = None) -> Any:
        """Retrieve stored intermediate data."""
        return self.scratchpad.get(key, default)

    @property
    def is_exceeded(self) -> bool:
        """Whether the maximum allowed iteration limit has been reached.

        Always False when max_iterations is 0 or negative - the loop then runs
        until the goal is genuinely done.
        """
        if not self.max_iterations or self.max_iterations <= 0:
            return False
        return self.current_iteration >= self.max_iterations

    @property
    def elapsed_seconds(self) -> float:
        """Time spent working on this goal, excluding time spent waiting on us.

        A safety confirmation is answered by voice, which can take as long as
        the user takes. Counting that against the budget would mean the slower
        someone speaks, the less of their task gets done.
        """
        end = self._paused_at if self._paused_at is not None else time.monotonic()
        return end - self._started

    @property
    def is_out_of_time(self) -> bool:
        if not self.max_seconds or self.max_seconds <= 0:
            return False
        return self.elapsed_seconds >= self.max_seconds

    def pause_clock(self) -> None:
        """Stop the budget clock while waiting for the user to answer."""
        if self._paused_at is None:
            self._paused_at = time.monotonic()

    def resume_clock(self) -> None:
        if self._paused_at is not None:
            self._started += time.monotonic() - self._paused_at
            self._paused_at = None

    def format_history_markdown(self) -> str:
        """Format recent action history as clean markdown for LLM context (last 3 steps)."""
        if not self.steps_taken:
            return "No previous steps executed."

        lines = []
        recent_steps = self.steps_taken[-3:]  # Keep prefill prompt context light
        for step in recent_steps:
            status = step.result.get("status", "ok")
            lines.append(
                f"Step {step.step_number}: Thought: '{step.thought}' -> Action: `{step.action}` (Params: {step.params}) -> Status: {status}"
            )
            if "error" in step.result:
                lines.append(f"  Error: {step.result['error']}")
            else:
                res = step.result.get("result", step.result)
                if isinstance(res, dict):
                    if "windows" in res:
                        titles = [w.get("title", "") for w in res["windows"] if w.get("title")]
                        lines.append(f"  Windows found ({len(titles)}): {', '.join(titles[:8])}")
                    elif "apps" in res:
                        app_names = [a.get("name", "") for a in res["apps"] if a.get("name")]
                        lines.append(f"  Apps found ({len(app_names)}): {', '.join(app_names[:8])}")
                    elif "text" in res and isinstance(res["text"], str):
                        lines.append(f"  Result text: {res['text'][:300]}")
                    elif "message" in res:
                        lines.append(f"  Result message: {res['message']}")
                    else:
                        lines.append(f"  Result data: {str(res)[:300]}")
                elif isinstance(res, str):
                    lines.append(f"  Result: {res[:300]}")
        return "\n".join(lines)

    def format_scratchpad_markdown(self) -> str:
        """Format scratchpad data keys as markdown summary."""
        if not self.scratchpad:
            return "Scratchpad is empty."
        lines = []
        for key, val in self.scratchpad.items():
            if isinstance(val, str):
                val_snippet = val[:300] + ("..." if len(val) > 300 else "")
                lines.append(f"- **{key}**: {val_snippet}")
            elif isinstance(val, list):
                val_str = ", ".join(str(v) for v in val[:10])
                lines.append(f"- **{key}**: [{val_str}]")
            else:
                lines.append(f"- **{key}**: {type(val).__name__} ({str(val)[:200]})")
        return "\n".join(lines)

    def format_established_ruled_out_markdown(self) -> str:
        """Format only established facts and ruled-out approaches (plus data) for model context.

        Never sends the whole 3-step history so the model only ever makes small,
        single-screen decisions based on the current goal step and known bounds.
        """
        lines = []
        established = self.scratchpad.get("established", [])
        if established:
            lines.append("Established facts:")
            for item in established:
                lines.append(f"- {item}")
        ruled_out = self.scratchpad.get("ruled_out", [])
        if ruled_out:
            lines.append("Ruled out approaches:")
            for item in ruled_out:
                lines.append(f"- {item}")
        other_keys = {k: v for k, v in self.scratchpad.items() if k not in ("established", "ruled_out")}
        if other_keys:
            lines.append("Scratchpad data:")
            for key, val in other_keys.items():
                if isinstance(val, str):
                    val_snippet = val[:300] + ("..." if len(val) > 300 else "")
                    lines.append(f"- **{key}**: {val_snippet}")
                elif isinstance(val, list):
                    val_str = ", ".join(str(v) for v in val[:10])
                    lines.append(f"- **{key}**: [{val_str}]")
                else:
                    lines.append(f"- **{key}**: {type(val).__name__} ({str(val)[:200]})")
        if not lines:
            return "No previous facts or ruled out approaches."
        return "\n".join(lines)
