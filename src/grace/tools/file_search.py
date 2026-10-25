"""Find files and folders by name.

Two routes, both offline:

  1. The Windows Search index (OLE DB, `SystemIndex`): instant, and it already
     knows every indexed folder on the machine.
  2. An `os.scandir` walk, used only when the index fails or finds nothing.

Matching rule, the same for both routes: the query is split into words on any
non-alphanumeric character (spaces, `-`, `_`, `.`, punctuation), and a name
matches when every word appears in it, in any order, case-insensitive. So
`qwen 3.6 35b a3b` matches the folder `Qwen3.6-35B-A3B-GGUF`.

Relaxed rule: a long, spoken query often has a word the name never contains
(`... model`). If nothing contains every word, words that match no candidate at
all are dropped, but only while at least two words remain, and the names
containing all of the remaining words are returned.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Callable, Iterable, Optional

logger = logging.getLogger("grace.tools.file_search")

MAX_RESULTS = 20
TIME_BUDGET_SECONDS = 15.0
RELAXED_POOL = 500

SKIP_DIRS = frozenset({
    "venv", ".venv", "node_modules", ".git", "__pycache__", "appdata",
    "$recycle.bin", "windows",
})
PROFILE_FOLDERS = ("Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

SearchResult = tuple[str, bool]  # (path, is_folder)


def query_words(query: str) -> list[str]:
    """Distinct lower-case alphanumeric words of `query`, in order."""
    seen: list[str] = []
    for word in _NON_ALNUM.split((query or "").lower()):
        if word and word not in seen:
            seen.append(word)
    return seen


def _norm(name: str) -> str:
    return _NON_ALNUM.sub(" ", name.lower()).strip()


def _words_in(name: str, words: Iterable[str]) -> set[str]:
    hay = _norm(name)
    return {w for w in words if w in hay}


def matches_all(name: str, words: list[str]) -> bool:
    return bool(words) and len(_words_in(name, words)) == len(words)


def relax(pool: list[SearchResult], words: list[str]) -> list[SearchResult]:
    """The relaxed rule: drop words nothing matched, if at least two remain."""
    useful: set[str] = set()
    for path, _ in pool:
        useful |= _words_in(os.path.basename(path), words)
    kept = [w for w in words if w in useful]
    if len(kept) < 2:
        return []
    return [(p, f) for p, f in pool if matches_all(os.path.basename(p), kept)]


# -- Windows Search index -----------------------------------------------------

def _index_query(words: list[str], strict: bool, limit: int) -> list[SearchResult]:
    """Ask the Windows Search index. Raises if the index is unavailable.

    The seam the tests replace: nothing else here touches the index.
    """
    import pythoncom
    import win32com.client

    joiner = " AND " if strict else " OR "
    conds = joiner.join(
        f"(System.ItemName LIKE '%{w}%' OR System.FileName LIKE '%{w}%')" for w in words
    )
    sql = (
        f"SELECT TOP {int(limit)} System.ItemPathDisplay, System.ItemType "
        f"FROM SystemIndex WHERE {conds}"
    )
    pythoncom.CoInitialize()
    try:
        conn = win32com.client.Dispatch("ADODB.Connection")
        conn.Open("Provider=Search.CollatorDSO;Extended Properties='Application=Windows'")
        try:
            rs = conn.Execute(sql)[0]
            found: list[SearchResult] = []
            while not rs.EOF:
                path = rs.Fields.Item(0).Value
                kind = str(rs.Fields.Item(1).Value or "").lower()
                if path:
                    found.append((str(path), kind in ("directory", "folder")))
                rs.MoveNext()
            rs.Close()
            return found
        finally:
            conn.Close()
    finally:
        pythoncom.CoUninitialize()


def search_index(words: list[str], max_results: int = MAX_RESULTS) -> list[SearchResult]:
    """Index route. Raises if the index is unavailable; [] means it found nothing."""
    strict = [
        r for r in _index_query(words, True, max_results * 4)
        if matches_all(os.path.basename(r[0]), words)
    ]
    if strict:
        return strict[:max_results]
    if len(words) < 3:
        return []
    pool = _index_query(words, False, RELAXED_POOL)
    return relax(pool, words)[:max_results]


# -- scandir walk -------------------------------------------------------------

def _fixed_drive_roots() -> list[str]:
    """Fixed drives other than the system drive, e.g. ['D:\\']. Seam for tests."""
    try:
        import win32api
        import win32file

        system = os.environ.get("SystemDrive", "C:").upper()
        roots = []
        for drive in win32api.GetLogicalDriveStrings().split("\x00"):
            if drive and win32file.GetDriveType(drive) == win32file.DRIVE_FIXED:
                if drive[:2].upper() != system:
                    roots.append(drive)
        return roots
    except Exception:
        return []


def default_walk_roots() -> list[tuple[str, Optional[int]]]:
    """(root, max_depth) pairs: profile folders in full, then other drives two levels deep."""
    profile = os.environ.get("USERPROFILE", os.path.expanduser("~"))
    roots: list[tuple[str, Optional[int]]] = [
        (os.path.join(profile, name), None) for name in PROFILE_FOLDERS
    ]
    roots += [(drive, 2) for drive in _fixed_drive_roots()]
    return roots


def walk_search(
    roots: list[tuple[str, Optional[int]]],
    words: list[str],
    *,
    max_results: int = MAX_RESULTS,
    time_budget: float = TIME_BUDGET_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[list[SearchResult], Optional[str]]:
    """Breadth-first scandir over `roots`; returns (results, why it stopped early or None).

    Folders count as matches too. Stops at `max_results` all-word matches or
    when `time_budget` seconds have passed; names holding at least two query
    words are kept as a pool for the relaxed rule if no all-word match turns up.
    """
    start = clock()
    strict: list[SearchResult] = []
    pool: list[SearchResult] = []
    stopped: Optional[str] = None

    def out_of_time() -> bool:
        return clock() - start >= time_budget

    for root, max_depth in roots:
        if stopped or not os.path.isdir(root):
            continue
        queue: list[tuple[str, int]] = [(root, 0)]
        while queue and not stopped:
            current, depth = queue.pop(0)
            if out_of_time():
                stopped = f"stopped after {time_budget:g} seconds"
                break
            try:
                with os.scandir(current) as it:
                    entries = list(it)
            except OSError:
                continue
            for entry in entries:
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                hit = _words_in(entry.name, words)
                if len(hit) == len(words):
                    strict.append((entry.path, is_dir))
                    if len(strict) >= max_results:
                        stopped = f"stopped at {max_results} results"
                        break
                elif len(hit) >= 2 and len(pool) < RELAXED_POOL:
                    pool.append((entry.path, is_dir))
                if (
                    is_dir
                    and entry.name.lower() not in SKIP_DIRS
                    and (max_depth is None or depth + 1 < max_depth)
                ):
                    queue.append((entry.path, depth + 1))
            if not stopped and out_of_time():
                stopped = f"stopped after {time_budget:g} seconds"

    if strict:
        return strict, stopped
    return relax(pool, words)[:max_results], stopped
