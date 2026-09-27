"""Shared "download, then verify" helper for Grace's setup scripts.

The red-team review's theme 3: three separate setup scripts fetched
binaries/JARs over plain HTTP(S) with no hash or signature check anywhere,
and never re-verified on subsequent runs - most seriously
`setup_llama_cpp.py`, whose binary is used both as the local planner LLM and
as the vision model that decides where Grace clicks on screen (R4). The
structural fix is one shared "verified download" helper that every setup
script is required to call, so none of them can quietly roll its own fetch
loop and skip the check.

Fails closed by construction: `verify_sha256` never treats a missing or
placeholder expected hash as "skip the check" - it only ever compares, and an
unset/wrong hash simply never matches, so nothing downloaded under one is
ever trusted.
"""

import hashlib
import os
import urllib.request


class HashMismatch(RuntimeError):
    """A file's SHA-256 did not match what was pinned for it."""


def sha256_of(path: str) -> str:
    """The lowercase hex SHA-256 digest of a file, streamed a chunk at a time."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str, expected_sha256: str) -> None:
    """Raise `HashMismatch` (and delete `path`) unless it hashes to `expected_sha256`.

    Deleting on failure matters as much as raising: a file that fails
    verification must never be left sitting where a later, less careful
    check (or a human) might use it anyway.
    """
    actual = sha256_of(path)
    expected = (expected_sha256 or "").strip().lower()
    if actual.lower() != expected:
        try:
            os.remove(path)
        except OSError:
            pass
        raise HashMismatch(
            f"{path} does not match its pinned SHA-256 - refusing to use it.\n"
            f"  expected: {expected or '(not set)'}\n"
            f"  actual:   {actual}\n"
            "This could mean the file was swapped at the source, or the pinned "
            "hash needs updating for a new release."
        )


def download_and_verify(
    url: str,
    dest_path: str,
    expected_sha256: str,
    *,
    chunk_size: int = 1024 * 1024,
    progress: bool = True,
) -> None:
    """Download `url` to `dest_path`, then verify it. Raises and cleans up on any failure.

    One shared fetch loop, so "downloaded but never checked" cannot happen by
    a script simply not calling the check afterward.
    """
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    try:
        with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if progress and total_size > 0:
                    pct = downloaded / total_size * 100
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(
                        f"\r  Progress: {pct:.1f}% ({mb_down:.1f} MB / {mb_total:.1f} MB)",
                        end="", flush=True,
                    )
            if progress:
                print()
    except BaseException:
        # A partial file is worse than no file: it can look "already
        # downloaded" to a naive existence check on the next run.
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        raise

    verify_sha256(dest_path, expected_sha256)
