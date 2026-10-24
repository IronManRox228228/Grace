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

from grace.agent.facts import looks_like_secret, redact_value

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
                    self._conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS facts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            topic TEXT NOT NULL,
                            group_key TEXT NOT NULL,
                            kind TEXT NOT NULL,
                            value TEXT NOT NULL,
                            provenance TEXT NOT NULL,
                            status TEXT NOT NULL,
                            active INTEGER NOT NULL DEFAULT 1,
                            created_turn INTEGER NOT NULL DEFAULT 0,
                            last_used_turn INTEGER NOT NULL DEFAULT 0,
                            superseded_by INTEGER,
                            supersede_reason TEXT,
                            created REAL,
                            confirmed INTEGER NOT NULL DEFAULT 0
                        )
                        """
                    )
                    self._conn.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS facts_one_active_per_topic ON facts(topic) WHERE active = 1"
                    )
                    self._conn.execute(
                        "CREATE INDEX IF NOT EXISTS facts_group_active ON facts(group_key) WHERE active = 1"
                    )
                    try:
                        self._conn.execute(
                            "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(value, content='facts', content_rowid='id')"
                        )
                        self._conn.execute(
                            """
                            CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                                INSERT INTO facts_fts(rowid, value) VALUES (new.id, new.value);
                            END;
                            """
                        )
                        self._conn.execute(
                            """
                            CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
                                INSERT INTO facts_fts(facts_fts, rowid, value) VALUES('delete', old.id, old.value);
                            END;
                            """
                        )
                        self._conn.execute(
                            """
                            CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
                                INSERT INTO facts_fts(facts_fts, rowid, value) VALUES('delete', old.id, old.value);
                                INSERT INTO facts_fts(rowid, value) VALUES (new.id, new.value);
                            END;
                            """
                        )
                    except Exception as fts_exc:
                        logger.debug("FTS5 table/triggers skipped or not available: %s", fts_exc)
                self._flush_encrypted()
            except Exception as exc:
                logger.warning(f"Persistent memory DB init failed: {exc}")

    def save_step(self, user_goal: str, action: str, params: dict, result: dict, app: str = "unknown") -> None:
        """Persist a completed action step to SQLite disk database with typed text redacted."""
        with self._lock:
            try:
                params_to_save = params
                if isinstance(params, dict) and "text" in params and isinstance(params["text"], str):
                    import hashlib
                    raw_text = params["text"]
                    h = hashlib.sha256(f"{self.salt}:{raw_text}".encode("utf-8")).hexdigest()[:16]
                    params_to_save = dict(params)
                    params_to_save["text"] = f"[REDACTED len={len(raw_text)} hash={h}]"

                with self._conn:
                    self._conn.execute(
                        "INSERT INTO task_history (user_goal, action, params, result, timestamp) VALUES (?, ?, ?, ?, ?)",
                        (user_goal, action, json.dumps(params_to_save), json.dumps(result), time.time()),
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
                    deleted += cursor.rowcount
                    cursor.execute("DELETE FROM user_preferences WHERE key LIKE ?", (f"%{goal}%",))
                    deleted += cursor.rowcount
                self._flush_encrypted()
            except Exception as exc:
                logger.debug(f"Failed to forget goal from persistent memory: {exc}")
        return deleted

    def forget_today(self, start_time: Optional[float] = None) -> int:
        """Forget all records logged today."""
        if start_time is None:
            t = time.localtime()
            start_time = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, t.tm_isdst))
        deleted = 0
        with self._lock:
            try:
                with self._conn:
                    cursor = self._conn.cursor()
                    cursor.execute("DELETE FROM task_history WHERE timestamp >= ?", (start_time,))
                    deleted = cursor.rowcount
                self._flush_encrypted()
            except Exception as exc:
                logger.debug(f"Failed to forget today from persistent memory: {exc}")
        return deleted

    def forget_app(self, app_name: str) -> int:
        """Forget all records matching an application name."""
        deleted = 0
        pattern = f"%{app_name.lower()}%"
        with self._lock:
            try:
                with self._conn:
                    cursor = self._conn.cursor()
                    cursor.execute(
                        "DELETE FROM task_history WHERE lower(user_goal) LIKE ? OR lower(params) LIKE ?",
                        (pattern, pattern),
                    )
                    deleted = cursor.rowcount
                self._flush_encrypted()
            except Exception as exc:
                logger.debug(f"Failed to forget app {app_name} from persistent memory: {exc}")
        return deleted

    def forget_everything(self) -> None:
        """Clear all stored task history, user preferences, and facts (PLAN.md §12.1, R17)."""
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute("DELETE FROM task_history")
                    self._conn.execute("DELETE FROM user_preferences")
                    self._conn.execute("DELETE FROM facts")
                    try:
                        self._conn.execute("DELETE FROM facts_fts")
                    except Exception:
                        pass
                self._flush_encrypted()
                logger.info("Wiped persistent memory: task_history, user_preferences, and facts cleared")
            except Exception as exc:
                logger.debug(f"Failed to wipe persistent memory: {exc}")

    def store_fact(
        self,
        topic: str,
        value: str,
        provenance: str = "user_heard",
        group_key: Optional[str] = None,
        kind: str = "fact",
        current_turn: int = 0,
        is_password_field: bool = False,
    ) -> int:
        """Store a new fact about the user (PLAN.md §12.1).

        Only 'user_heard', 'model', or 'screen' can be passed as initial provenance.
        'user_confirmed' is reachable solely via confirm_fact().
        Refuses secret-shaped topics or values.
        """
        if is_password_field or looks_like_secret(topic, value):
            raise ValueError(f"Secret-shaped value refused for topic '{topic}'")
        if provenance == "user_confirmed":
            raise ValueError("user_confirmed provenance cannot be set directly; only confirm_fact() can produce it")
        if provenance not in ("user_heard", "model", "screen"):
            raise ValueError(f"Invalid provenance '{provenance}'")

        group_key = group_key or topic
        status = "pending" if provenance == "user_heard" else "usable"
        now = time.time()

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT id FROM facts WHERE topic = ? AND active = 1", (topic,))
            if cursor.fetchone():
                raise ValueError(f"Topic '{topic}' is already active; use supersede_fact instead")

            with self._conn:
                cursor.execute(
                    """
                    INSERT INTO facts (topic, group_key, kind, value, provenance, status, active, created_turn, last_used_turn, created, confirmed)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, 0)
                    """,
                    (topic, group_key, kind, value, provenance, status, current_turn, current_turn, now),
                )
                fact_id = cursor.lastrowid
            self._flush_encrypted()
            logger.info("Stored fact id=%d topic=%s provenance=%s value=%s", fact_id, topic, provenance, redact_value(value, self.salt))
            return fact_id

    def supersede_fact(
        self,
        topic: str,
        value: str,
        reason: str = "user correction",
        provenance: str = "user_heard",
        kind: str = "fact",
        current_turn: int = 0,
        is_password_field: bool = False,
    ) -> int:
        """Replace the active fact for a topic in one transaction, recording why (PLAN.md §12.1)."""
        if is_password_field or looks_like_secret(topic, value):
            raise ValueError(f"Secret-shaped value refused for topic '{topic}'")
        if provenance == "user_confirmed":
            raise ValueError("user_confirmed provenance cannot be set directly; use confirm_fact()")
        if provenance not in ("user_heard", "model", "screen"):
            raise ValueError(f"Invalid provenance '{provenance}'")

        status = "pending" if provenance == "user_heard" else "usable"
        now = time.time()

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT id, group_key FROM facts WHERE topic = ? AND active = 1", (topic,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"No active fact for topic '{topic}' to supersede")
            old_id, group_key = row[0], row[1]

            with self._conn:
                cursor.execute("UPDATE facts SET active = 0, supersede_reason = ? WHERE id = ?", (reason, old_id))
                cursor.execute(
                    """
                    INSERT INTO facts (topic, group_key, kind, value, provenance, status, active, created_turn, last_used_turn, created, confirmed)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, 0)
                    """,
                    (topic, group_key, kind, value, provenance, status, current_turn, current_turn, now),
                )
                new_id = cursor.lastrowid
                cursor.execute("UPDATE facts SET superseded_by = ? WHERE id = ?", (new_id, old_id))
            self._flush_encrypted()
            logger.info("Superseded fact old_id=%d new_id=%d topic=%s value=%s", old_id, new_id, topic, redact_value(value, self.salt))
            return new_id

    def confirm_fact(self, topic: str, current_turn: int = 0) -> None:
        """The dedicated confirmation path (PLAN.md §12.1).

        Promotes a pending fact to user_confirmed and usable.
        Only called when the user explicitly confirmed with yes.
        """
        with self._lock:
            cursor = self._conn.cursor()
            with self._conn:
                cursor.execute(
                    """
                    UPDATE facts
                    SET provenance = 'user_confirmed', status = 'usable', confirmed = 1, last_used_turn = ?
                    WHERE topic = ? AND active = 1
                    """,
                    (current_turn, topic),
                )
                if cursor.rowcount == 0:
                    raise ValueError(f"No active fact for topic '{topic}' to confirm")
            self._flush_encrypted()
            logger.info("Confirmed fact topic=%s", topic)

    def get_active_fact(self, topic: str) -> Optional[dict[str, Any]]:
        """Retrieve the single active fact for a topic if one exists."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, topic, group_key, kind, value, provenance, status, active,
                       created_turn, last_used_turn, superseded_by, supersede_reason, created, confirmed
                FROM facts WHERE topic = ? AND active = 1
                """,
                (topic,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "topic": row[1],
                "group_key": row[2],
                "kind": row[3],
                "value": row[4],
                "provenance": row[5],
                "status": row[6],
                "active": bool(row[7]),
                "created_turn": row[8],
                "last_used_turn": row[9],
                "superseded_by": row[10],
                "supersede_reason": row[11],
                "created": row[12],
                "confirmed": bool(row[13]),
            }

    def get_all_confirmed_facts(self) -> list[dict[str, Any]]:
        """All active, user-confirmed facts, ordered by recency of use."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, topic, group_key, kind, value, provenance, status, active,
                       created_turn, last_used_turn, superseded_by, supersede_reason, created, confirmed
                FROM facts
                WHERE active = 1 AND provenance = 'user_confirmed'
                ORDER BY last_used_turn DESC, id DESC
                """
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "topic": r[1],
                    "group_key": r[2],
                    "kind": r[3],
                    "value": r[4],
                    "provenance": r[5],
                    "status": r[6],
                    "active": bool(r[7]),
                    "created_turn": r[8],
                    "last_used_turn": r[9],
                    "superseded_by": r[10],
                    "supersede_reason": r[11],
                    "created": r[12],
                    "confirmed": bool(r[13]),
                }
                for r in rows
            ]

    def get_all_facts(self) -> list[str]:
        """List of confirmed fact strings for word biasing."""
        facts = self.get_all_confirmed_facts()
        return [f"{f['topic']}: {f['value']}" for f in facts]

    def forget_fact(self, topic: str) -> int:
        """Hard delete a topic's entire history (PLAN.md §12.1)."""
        with self._lock:
            cursor = self._conn.cursor()
            with self._conn:
                cursor.execute("DELETE FROM facts WHERE topic = ?", (topic,))
                deleted = cursor.rowcount
            self._flush_encrypted()
            logger.info("Forgot fact topic=%s (deleted %d rows)", topic, deleted)
            return deleted

    def render_context(self, budget_chars: int = 600, max_facts: int = 8) -> str:
        """Render bounded confirmed facts for prompt injection (PLAN.md §12.1).

        Capped at max_facts (default 8) and budget_chars (default 600).
        """
        facts = self.get_all_confirmed_facts()[:max_facts]
        lines = []
        total_len = 0
        for f in facts:
            line = f"- {f['topic']}: {f['value']}"
            if total_len + len(line) + 1 > budget_chars:
                break
            lines.append(line)
            total_len += len(line) + 1
        return "\n".join(lines)


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
