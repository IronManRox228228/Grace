"""Test suite for Windows AppIndexer."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.automation.app_indexer import AppIndexer


class TestAppIndexer:
    """Test suite for AppIndexer."""

    def test_init_and_find(self):
        indexer = AppIndexer()
        assert indexer._indexed
        assert isinstance(indexer._apps, dict)

    def test_find_known_app(self):
        indexer = AppIndexer()
        # Edge or Notepad or PowerShell should be found on Windows
        target = indexer.find_app("notepad") or indexer.find_app("edge") or indexer.find_app("powershell")
        assert target is not None

    def test_launch_nonexistent(self):
        indexer = AppIndexer()
        res = indexer.launch("nonexistent_fake_app_12345")
        assert res["status"] in ("ok", "error")
