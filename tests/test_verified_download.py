"""Tests for scripts/verified_download.py - the shared "download, then
verify" helper (R4 and its structural fix, red-team theme 3): a setup script
that fetches a binary must hash-check it before anything touches it, and
fail closed on a mismatch rather than silently trusting whatever the source
currently serves.

Nothing here touches the network: `urllib.request.urlopen` is mocked in
every test that exercises `download_and_verify`.
"""

import hashlib
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import verified_download as vd  # noqa: E402


class _FakeResponse:
    """Stands in for the object `urllib.request.urlopen` returns."""

    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)
        self.headers = {"content-length": str(len(data))}

    def read(self, n):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestSha256Of:
    def test_matches_hashlib_for_a_small_file(self, tmp_path):
        path = tmp_path / "hello.txt"
        path.write_bytes(b"hello world")
        assert vd.sha256_of(str(path)) == hashlib.sha256(b"hello world").hexdigest()

    def test_streams_a_larger_file_correctly(self, tmp_path):
        data = os.urandom(1024 * 1024 * 3 + 17)  # not a clean multiple of the chunk size
        path = tmp_path / "big.bin"
        path.write_bytes(data)
        assert vd.sha256_of(str(path)) == hashlib.sha256(data).hexdigest()


class TestVerifySha256:
    def test_a_matching_hash_passes_silently(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(b"trusted content")
        expected = hashlib.sha256(b"trusted content").hexdigest()
        vd.verify_sha256(str(path), expected)  # must not raise
        assert path.exists()

    def test_matching_is_case_insensitive(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(b"trusted content")
        expected = hashlib.sha256(b"trusted content").hexdigest().upper()
        vd.verify_sha256(str(path), expected)

    def test_a_mismatch_raises_and_deletes_the_file(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(b"swapped content")
        with pytest.raises(vd.HashMismatch):
            vd.verify_sha256(str(path), "0" * 64)
        assert not path.exists(), "a file that failed verification must not be left on disk"

    def test_an_unset_placeholder_hash_never_matches_real_content(self, tmp_path):
        """The exact fail-closed shape scripts/setup_llama_cpp.py relies on:
        an all-zero placeholder must never be treated as "skip the check".
        """
        path = tmp_path / "f.bin"
        path.write_bytes(b"anything at all")
        with pytest.raises(vd.HashMismatch):
            vd.verify_sha256(str(path), "0" * 64)


class TestDownloadAndVerify:
    def test_a_verified_download_writes_the_file(self, tmp_path, monkeypatch):
        data = b"a perfectly good release asset"
        monkeypatch.setattr(vd.urllib.request, "urlopen", lambda req: _FakeResponse(data))
        dest = tmp_path / "asset.zip"

        vd.download_and_verify(
            "https://example.test/asset.zip", str(dest),
            hashlib.sha256(data).hexdigest(), progress=False,
        )

        assert dest.read_bytes() == data

    def test_a_hash_mismatch_deletes_the_downloaded_file(self, tmp_path, monkeypatch):
        data = b"a swapped release asset"
        monkeypatch.setattr(vd.urllib.request, "urlopen", lambda req: _FakeResponse(data))
        dest = tmp_path / "asset.zip"

        with pytest.raises(vd.HashMismatch):
            vd.download_and_verify(
                "https://example.test/asset.zip", str(dest), "0" * 64, progress=False,
            )

        assert not dest.exists(), "a mismatched download must never be left in place"

    def test_a_transport_failure_cleans_up_the_partial_file(self, tmp_path, monkeypatch):
        class _BrokenResponse(_FakeResponse):
            def read(self, n):
                raise ConnectionError("connection reset")

        monkeypatch.setattr(vd.urllib.request, "urlopen", lambda req: _BrokenResponse(b"x"))
        dest = tmp_path / "asset.zip"

        with pytest.raises(ConnectionError):
            vd.download_and_verify("https://example.test/asset.zip", str(dest), "0" * 64, progress=False)

        assert not dest.exists(), "a partial download must not look like a completed one next run"
