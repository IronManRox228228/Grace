"""Cockpit 4 item 3c: search_files searches the index, then the disk, and finds folders.

Nothing here queries the real Windows Search index or walks a real folder:
the OLE DB call is patched and every walk runs inside a temp dir.
"""

import asyncio
import os

import pytest

from grace.intent.parser import Intent
from grace.tools import file_search
from grace.tools.dispatcher import Dispatcher

QUERY = "qwen 3.6 35b a3b model"


@pytest.fixture(autouse=True)
def no_real_index(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the real Windows Search index was queried")

    monkeypatch.setattr(file_search, "_index_query", refuse)
    monkeypatch.setattr(file_search, "_fixed_drive_roots", lambda: [])


class TestWordMatching:
    def test_words_split_on_separators(self):
        assert file_search.query_words("Qwen3.6-35B_A3B model") == ["qwen3", "6", "35b", "a3b", "model"]

    def test_spoken_query_matches_the_folder_through_the_relaxed_rule(self, tmp_path):
        (tmp_path / "Qwen3.6-35B-A3B-GGUF").mkdir()
        (tmp_path / "Qwen2-7B").mkdir()
        (tmp_path / "notes.txt").write_text("x")
        words = file_search.query_words(QUERY)
        results, stopped = file_search.walk_search([(str(tmp_path), None)], words)
        assert stopped is None
        assert [(os.path.basename(p), f) for p, f in results] == [("Qwen3.6-35B-A3B-GGUF", True)]

    def test_all_words_in_any_order_case_insensitive(self, tmp_path):
        (tmp_path / "Quarterly_Budget-2026.xlsx").write_text("x")
        results, _ = file_search.walk_search([(str(tmp_path), None)], file_search.query_words("BUDGET quarterly"))
        assert len(results) == 1 and results[0][1] is False

    def test_one_matching_word_is_not_enough_to_relax(self, tmp_path):
        (tmp_path / "Qwen2-7B").mkdir()
        results, _ = file_search.walk_search([(str(tmp_path), None)], file_search.query_words(QUERY))
        assert results == []


class TestSkipListAndDepth:
    def test_skip_list_is_not_walked(self, tmp_path):
        for skipped in ("venv", "node_modules", ".git", "__pycache__", "AppData", "Windows", "$Recycle.Bin"):
            (tmp_path / skipped / "deep").mkdir(parents=True)
            (tmp_path / skipped / "deep" / "target-file.txt").write_text("x")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "target-file.txt").write_text("x")
        results, _ = file_search.walk_search([(str(tmp_path), None)], ["target", "file"])
        assert [os.path.relpath(p, tmp_path) for p, _ in results] == [os.path.join("docs", "target-file.txt")]

    def test_other_drives_are_only_two_levels_deep(self, tmp_path):
        (tmp_path / "a" / "b" / "c").mkdir(parents=True)
        (tmp_path / "a" / "b" / "needle-two").mkdir()
        (tmp_path / "a" / "b" / "c" / "needle-three").mkdir()
        results, _ = file_search.walk_search([(str(tmp_path), 2)], ["needle"])
        assert [os.path.basename(p) for p, _ in results] == []
        (tmp_path / "needle-one").mkdir()
        (tmp_path / "a" / "needle-two").mkdir()
        results, _ = file_search.walk_search([(str(tmp_path), 2)], ["needle"])
        assert sorted(os.path.basename(p) for p, _ in results) == ["needle-one", "needle-two"]


class TestCaps:
    def test_time_cap_stops_a_slow_walk_and_says_so(self, tmp_path):
        for i in range(30):
            (tmp_path / f"d{i}").mkdir()
        ticks = iter(range(0, 10_000))
        slow_clock = lambda: next(ticks) * 1.0  # one fake second per look at the clock
        results, stopped = file_search.walk_search(
            [(str(tmp_path), None)], ["zzz"], time_budget=5.0, clock=slow_clock
        )
        assert results == []
        assert stopped == "stopped after 5 seconds"

    def test_result_cap(self, tmp_path):
        for i in range(40):
            (tmp_path / f"report-{i}").mkdir()
        results, stopped = file_search.walk_search([(str(tmp_path), None)], ["report"], max_results=20)
        assert len(results) == 20 and stopped == "stopped at 20 results"


class TestIndexRoute:
    def test_index_is_asked_first_and_filters_to_every_word(self, monkeypatch):
        asked = []

        def fake(words, strict, limit):
            asked.append(strict)
            return [(r"D:\models\Qwen3.6-35B-A3B-GGUF", True), (r"D:\models\other thing", True)]

        monkeypatch.setattr(file_search, "_index_query", fake)
        results = file_search.search_index(["qwen3", "6", "35b"])
        assert asked == [True]
        assert results == [(r"D:\models\Qwen3.6-35B-A3B-GGUF", True)]

    def test_relaxed_index_pass_for_a_long_spoken_query(self, monkeypatch):
        def fake(words, strict, limit):
            return [] if strict else [(r"D:\m\Qwen3.6-35B-A3B-GGUF", True)]

        monkeypatch.setattr(file_search, "_index_query", fake)
        assert file_search.search_index(file_search.query_words(QUERY)) == [(r"D:\m\Qwen3.6-35B-A3B-GGUF", True)]


class TestDispatcherSearch:
    def _run(self, query):
        return asyncio.run(Dispatcher(computer_use=None).execute(Intent(tool="search_files", params={"query": query})))

    def test_index_hit_reports_paths_and_folder_flag(self, monkeypatch):
        monkeypatch.setattr(file_search, "search_index", lambda words, *a, **k: [(r"D:\models\Qwen3.6-35B-A3B-GGUF", True)])
        res = self._run(QUERY)
        assert res["status"] == "ok"
        assert res["results"] == [{"path": r"D:\models\Qwen3.6-35B-A3B-GGUF", "is_folder": True}]
        assert "(folder)" in res["text"] and "Windows Search index" in res["text"]

    def test_index_failure_falls_back_to_the_walk(self, monkeypatch, tmp_path):
        (tmp_path / "Qwen3.6-35B-A3B-GGUF").mkdir()

        def broken(words, *a, **k):
            raise OSError("Windows Search service is stopped")

        monkeypatch.setattr(file_search, "search_index", broken)
        monkeypatch.setattr(file_search, "default_walk_roots", lambda: [(str(tmp_path), None)])
        res = self._run(QUERY)
        assert [os.path.basename(r["path"]) for r in res["results"]] == ["Qwen3.6-35B-A3B-GGUF"]
        assert "scan of your folders" in res["text"]

    def test_index_finding_nothing_also_falls_back(self, monkeypatch, tmp_path):
        (tmp_path / "budget.xlsx").write_text("x")
        monkeypatch.setattr(file_search, "search_index", lambda words, *a, **k: [])
        monkeypatch.setattr(file_search, "default_walk_roots", lambda: [(str(tmp_path), None)])
        res = self._run("budget")
        assert res["results"][0]["is_folder"] is False

    def test_nothing_found_says_so(self, monkeypatch, tmp_path):
        monkeypatch.setattr(file_search, "search_index", lambda words, *a, **k: [])
        monkeypatch.setattr(file_search, "default_walk_roots", lambda: [(str(tmp_path), None)])
        res = self._run("nothing here")
        assert res["files"] == [] and "couldn't find" in res["text"]

    def test_default_walk_roots_never_leave_the_profile_without_other_drives(self, monkeypatch, tmp_path):
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setattr(file_search, "_fixed_drive_roots", lambda: ["E:\\"])
        roots = file_search.default_walk_roots()
        assert roots[0] == (str(tmp_path / "Desktop"), None)
        assert roots[-1] == ("E:\\", 2)
