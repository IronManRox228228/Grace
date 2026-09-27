"""Tests for scripts/setup_llama_cpp.py (R4): the download must be pinned
and verified before extraction ever runs.

Nothing here downloads anything - `download_and_verify` is mocked in every
test that would otherwise reach the network, and the "already installed"
tests never construct a real download at all.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import setup_llama_cpp  # noqa: E402
from verified_download import HashMismatch  # noqa: E402


class TestPinnedHashIsFailClosed:
    def test_the_pinned_hash_is_a_well_formed_sha256_shape(self):
        # 64 hex characters, whether or not it has been filled in with a
        # real one yet - a malformed constant would silently never match
        # anything, which is fail-closed for the wrong reason (a typo, not
        # a real check).
        assert len(setup_llama_cpp.LLAMA_CPP_SHA256) == 64
        int(setup_llama_cpp.LLAMA_CPP_SHA256, 16)  # raises ValueError if not hex

    def test_the_placeholder_hash_does_not_match_arbitrary_content(self, tmp_path):
        # Documents the fail-closed default: until a human fills in the real
        # hash, nothing downloaded under it is ever accepted.
        from verified_download import verify_sha256

        path = tmp_path / "whatever.zip"
        path.write_bytes(b"some plausible-looking release asset")
        with pytest.raises(HashMismatch):
            verify_sha256(str(path), setup_llama_cpp.LLAMA_CPP_SHA256)


class TestAlreadyInstalledSkipsDownload:
    def test_an_existing_exe_is_left_alone_and_nothing_is_downloaded(self, tmp_path, monkeypatch):
        fake_dir = tmp_path / "llama cpp"
        fake_dir.mkdir()
        (fake_dir / "llama-server.exe").write_bytes(b"already here")
        monkeypatch.setattr(setup_llama_cpp, "LLAMA_CPP_DIR", str(fake_dir))
        monkeypatch.setattr(setup_llama_cpp, "EXE_PATH", str(fake_dir / "llama-server.exe"))

        calls = []
        monkeypatch.setattr(
            setup_llama_cpp, "download_and_verify",
            lambda *a, **k: calls.append((a, k)),
        )

        assert setup_llama_cpp.download_and_setup() is True
        assert calls == [], "an already-installed exe must never trigger a download"


class TestHashMismatchRefusesToExtract:
    def test_a_failed_verification_stops_before_any_extraction(self, tmp_path, monkeypatch):
        fake_dir = tmp_path / "llama cpp"
        monkeypatch.setattr(setup_llama_cpp, "PROJECT_ROOT", str(tmp_path))
        monkeypatch.setattr(setup_llama_cpp, "LLAMA_CPP_DIR", str(fake_dir))
        monkeypatch.setattr(setup_llama_cpp, "EXE_PATH", str(fake_dir / "llama-server.exe"))

        def _fake_download_and_verify(url, dest_path, expected_sha256, **kwargs):
            raise HashMismatch("the asset does not match the pinned hash")

        monkeypatch.setattr(setup_llama_cpp, "download_and_verify", _fake_download_and_verify)

        extract_calls = []
        monkeypatch.setattr(
            setup_llama_cpp.zipfile, "ZipFile",
            lambda *a, **k: extract_calls.append((a, k)) or pytest.fail("must not reach extraction"),
        )

        assert setup_llama_cpp.download_and_setup() is False
        assert extract_calls == []
        assert not os.path.isfile(str(fake_dir / "llama-server.exe"))
