"""Test suite for Windows AppIndexer."""

import json
import sys
import os
import time

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.automation import app_indexer as app_indexer_module
from grace.automation.app_indexer import AppIndexer


@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    """Point the on-disk index at a throwaway file, not the user's ~/.grace.

    Also replaces the recursive Program Files glob with a stub: these tests are
    about cache mechanics, and a real scan takes ~12 seconds each.
    """
    path = str(tmp_path / "app_index.json")
    monkeypatch.setattr(app_indexer_module, "CACHE_PATH", path)

    scans = []

    def fake_scan(self):
        scans.append(1)
        self._apps = {"stubapp": r"C:\stub\stubapp.exe"}
        self._uwp_apps = {"stubuwp": "Stub.App_8wekyb!App"}
        self._indexed = True

    monkeypatch.setattr(AppIndexer, "_index_apps", fake_scan)
    return path, scans


class TestIndexCache:
    """The index globs Program Files recursively, so it is cached on disk.

    Before this, every 'open X' with no indexer yet paid the full scan on the
    event loop, freezing Grace for tens of seconds.
    """

    def test_cache_is_written_then_reused_without_rescanning(self, temp_cache):
        path, scans = temp_cache
        AppIndexer()
        assert os.path.exists(path)
        assert len(scans) == 1

        cached = AppIndexer()
        assert len(scans) == 1, "second construction must not rescan"
        assert cached._indexed
        assert cached._apps["stubapp"].endswith("stubapp.exe")
        assert cached._uwp_apps["stubuwp"] == "Stub.App_8wekyb!App"

    def test_use_cache_false_always_scans(self, temp_cache):
        _path, scans = temp_cache
        AppIndexer()
        AppIndexer(use_cache=False)
        assert len(scans) == 2

    def test_stale_cache_is_rebuilt(self, temp_cache):
        path, scans = temp_cache
        AppIndexer()
        old = time.time() - (app_indexer_module.CACHE_MAX_AGE_SECONDS + 60)
        os.utime(path, (old, old))

        AppIndexer()
        assert len(scans) == 2

    def test_version_mismatch_is_rebuilt(self, temp_cache):
        path, scans = temp_cache
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": 999, "apps": {"fake": "C:/fake.exe"}, "uwp": {}}, fh)

        indexer = AppIndexer()
        assert len(scans) == 1
        assert "fake" not in indexer._apps

    def test_corrupt_cache_falls_back_to_scanning(self, temp_cache):
        path, scans = temp_cache
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{not json at all")

        indexer = AppIndexer()
        assert indexer._indexed
        assert len(scans) == 1

    def test_empty_cache_is_not_trusted(self, temp_cache):
        path, scans = temp_cache
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": app_indexer_module.CACHE_VERSION, "apps": {}, "uwp": {}}, fh)

        AppIndexer()
        assert len(scans) == 1

    def test_missing_cache_directory_is_created(self, tmp_path, monkeypatch, temp_cache):
        nested = str(tmp_path / "does" / "not" / "exist" / "app_index.json")
        monkeypatch.setattr(app_indexer_module, "CACHE_PATH", nested)
        AppIndexer()
        assert os.path.exists(nested)

    def test_refresh_rescans_and_rewrites(self, temp_cache):
        path, scans = temp_cache
        indexer = AppIndexer()
        indexer._apps["sentinel_entry"] = "C:/sentinel.exe"
        indexer.refresh()
        assert len(scans) == 2
        assert "sentinel_entry" not in indexer._apps
        with open(path, "r", encoding="utf-8") as fh:
            assert "sentinel_entry" not in json.load(fh)["apps"]


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
