"""Tests for memory security: DPAPI encryption (R20), external salt (R23), and forget (R17)."""

import os
import shutil
import sqlite3
import tempfile
import pytest

from grace.agent.memory import PersistentMemoryStore


class TestMemorySecurity:
    def test_dpapi_encrypts_file_at_rest_and_persists_data(self, tmp_path):
        """A persistent store writes DPAPI encrypted bytes to disk, not raw SQLite."""
        db_path = str(tmp_path / "memory.db")
        store = PersistentMemoryStore(db_path=db_path)

        store.save_step("open notepad", "open_app", {"name": "Notepad"}, {"status": "ok"})
        store.set_preference("editor", "notepad")
        store.close()

        # Check raw bytes on disk
        with open(db_path, "rb") as f:
            raw_bytes = f.read()

        assert not raw_bytes.startswith(b"SQLite format 3\x00"), "Database on disk must NOT be plaintext SQLite header"

        # Attempting standard sqlite3 connection on the file directly should fail
        with pytest.raises(sqlite3.DatabaseError):
            with sqlite3.connect(db_path) as conn:
                conn.execute("SELECT * FROM user_preferences")

        # Re-opening with PersistentMemoryStore decrypts and loads successfully
        store2 = PersistentMemoryStore(db_path=db_path)
        assert store2.get_preference("editor") == "notepad"
        history = store2.get_recent_history(5)
        assert len(history) == 1
        assert history[0]["user_goal"] == "open notepad"
        store2.close()

    def test_plaintext_migration_creates_backup_and_encrypts(self, tmp_path):
        """Existing plaintext database is migrated in-place, creating .bak backup."""
        db_path = str(tmp_path / "legacy_memory.db")
        bak_path = db_path + ".bak"

        # 1. Create a legacy plaintext sqlite database
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE task_history (id INTEGER PRIMARY KEY, user_goal TEXT, action TEXT, params TEXT, result TEXT, timestamp REAL)")
            conn.execute("CREATE TABLE user_preferences (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)")
            conn.execute("INSERT INTO user_preferences VALUES ('browser', 'chrome', 12345.0)")
            conn.execute("INSERT INTO task_history VALUES (1, 'search web', 'browser_search', '{}', '{}', 12345.0)")
            conn.commit()

        with open(db_path, "rb") as f:
            assert f.read().startswith(b"SQLite format 3\x00")

        # 2. Open with PersistentMemoryStore -> triggers migration
        store = PersistentMemoryStore(db_path=db_path)
        assert os.path.exists(bak_path), "Backup file .bak must be created on migration"
        with open(bak_path, "rb") as f:
            assert f.read().startswith(b"SQLite format 3\x00"), "Backup must preserve original plaintext database"

        with open(db_path, "rb") as f:
            assert not f.read().startswith(b"SQLite format 3\x00"), "Original file must now be DPAPI encrypted"

        assert store.get_preference("browser") == "chrome"
        history = store.get_recent_history(5)
        assert len(history) == 1
        assert history[0]["user_goal"] == "search web"
        store.close()

    def test_external_salt_storage_and_persistence(self, tmp_path):
        """Salt is stored outside the database in memory.salt and persists across restarts."""
        db_path = str(tmp_path / "memory.db")
        salt_path = str(tmp_path / "memory.salt")

        store = PersistentMemoryStore(db_path=db_path)
        assert os.path.exists(salt_path), "memory.salt file must be created"
        salt1 = store.salt
        assert len(salt1) >= 32
        store.close()

        store2 = PersistentMemoryStore(db_path=db_path)
        assert store2.salt == salt1, "Salt must match previously generated salt in file"
        store2.close()

    def test_forget_goal_and_forget_everything(self, tmp_path):
        """forget_goal purges matching goal records and forget_everything clears store (R17)."""
        db_path = str(tmp_path / "memory.db")
        store = PersistentMemoryStore(db_path=db_path)

        store.save_step("open calculator", "open_app", {"name": "calc"}, {"status": "ok"})
        store.save_step("secret password check", "check_pwd", {}, {"status": "ok"})
        store.set_preference("key", "val")

        assert len(store.get_recent_history(5)) == 2

        deleted = store.forget_goal("secret password check")
        assert deleted == 1
        history = store.get_recent_history(5)
        assert len(history) == 1
        assert history[0]["user_goal"] == "open calculator"

        store.forget_everything()
        assert len(store.get_recent_history(5)) == 0
        assert store.get_preference("key") is None
        store.close()

    def test_dpapi_unavailable_falls_back_to_logged_plaintext(self, tmp_path, monkeypatch, caplog):
        """When DPAPI/win32crypt is unavailable, logs clearly and runs in plaintext mode."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "win32crypt":
                raise ImportError("win32crypt not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        db_path = str(tmp_path / "fallback_memory.db")
        with caplog.at_level("WARNING"):
            store = PersistentMemoryStore(db_path=db_path)
            assert "operating memory.db in plaintext mode" in caplog.text

            store.save_step("open terminal", "open_app", {}, {"status": "ok"})
            store.set_preference("mode", "plain")
            assert store.get_preference("mode") == "plain"
            store.close()

        # In fallback mode, disk file is plaintext SQLite
        with open(db_path, "rb") as f:
            assert f.read().startswith(b"SQLite format 3\x00")
