import os
import pytest
from grace.agent.memory import PersistentMemoryStore


def test_default_persistent_memory_store_does_not_touch_home_under_pytest():
    # Calling PersistentMemoryStore without db_path under pytest must default to in-memory
    store = PersistentMemoryStore()
    assert store._db_path == ":memory:"
    assert store._salt_path is None


def test_persistent_memory_store_refuses_real_home_path_under_pytest():
    real_home_db = os.path.join(os.path.expanduser("~/.grace"), "memory.db")
    with pytest.raises(RuntimeError, match="Under pytest, PersistentMemoryStore refused to open real home path"):
        PersistentMemoryStore(db_path=real_home_db)
