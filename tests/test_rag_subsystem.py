"""Unit tests for Grace local RAG engine and PersistentMemoryStore."""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.rag.retriever import DocumentChunker, LocalRagIndex
from grace.agent.memory import PersistentMemoryStore, AgentMemory


class TestRagSubsystem:
    """Test suite for local RAG retriever and persistent memory."""

    def test_document_chunker(self):
        """Test semantic document chunking."""
        text = "Sample sentence 1. " * 50
        chunks = DocumentChunker.chunk_text(text, chunk_size=200, overlap=40)
        assert len(chunks) > 1
        assert all(len(c) <= 250 for c in chunks)

    def test_local_rag_index_query(self):
        """Test sub-ms vector indexing and querying."""
        index = LocalRagIndex()
        doc = (
            "Grace is an offline voice accessibility assistant. "
            "Microsoft Edge is a web browser. "
            "Python PyAutoGUI performs desktop mouse clicks and keyboard key presses."
        )
        count = index.index_text("doc1", doc)
        assert count > 0
        assert index.is_indexed

        results = index.query("mouse clicks and key presses", top_k=2)
        assert len(results) > 0
        assert "PyAutoGUI" in results[0]["chunk"]

    def test_persistent_memory_store(self):
        """Test SQLite cross-session persistent memory store."""
        tmp_dir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmp_dir, "test_memory.db")
            store = PersistentMemoryStore(db_path=db_path)

            store.save_step("Open YouTube", "open_app", {"name": "YouTube"}, {"status": "ok"})
            store.set_preference("default_browser", "msedge")

            assert store.get_preference("default_browser") == "msedge"

            history = store.get_recent_history(limit=5)
            assert len(history) == 1
            assert history[0]["user_goal"] == "Open YouTube"
            assert history[0]["action"] == "open_app"
            store.close()
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_agent_memory_with_persistent_store(self):
        """Test AgentMemory integration with PersistentMemoryStore."""
        tmp_dir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmp_dir, "test_memory.db")
            store = PersistentMemoryStore(db_path=db_path)
            memory = AgentMemory(user_goal="Check volume", persistent_store=store)

            memory.add_step("Adjusting volume", "adjust_volume", {"amount": 50}, {"status": "ok"})
            assert len(memory.steps_taken) == 1

            recent = store.get_recent_history(limit=1)
            assert len(recent) == 1
            assert recent[0]["action"] == "adjust_volume"
            store.close()
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
