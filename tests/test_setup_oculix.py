"""Tests for scripts/setup_oculix.py's JAR integrity checks (R32, the same
"verified download" helper R4 introduced): a JAR pulled from Maven Central -
or already sitting in libs/ from an earlier, unverified run - must be
hash-checked before OculiX loads it into the process that issues real
clicks. Nothing here downloads anything or loads a real JAR.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import setup_oculix  # noqa: E402
from verified_download import HashMismatch  # noqa: E402


class TestEnsureJar:
    def test_an_existing_verified_jar_is_accepted_without_downloading(self, tmp_path, monkeypatch):
        jar = tmp_path / "lib.jar"
        jar.write_bytes(b"a genuine jar")
        expected = hashlib.sha256(b"a genuine jar").hexdigest()

        calls = []
        monkeypatch.setattr(setup_oculix, "download_and_verify", lambda *a, **k: calls.append(a))

        assert setup_oculix._ensure_jar("Test", "https://example.test/x.jar", str(jar), expected) is True
        assert calls == []

    def test_an_existing_jar_that_fails_verification_is_refused(self, tmp_path, monkeypatch):
        jar = tmp_path / "lib.jar"
        jar.write_bytes(b"a tampered jar")

        calls = []
        monkeypatch.setattr(setup_oculix, "download_and_verify", lambda *a, **k: calls.append(a))

        assert setup_oculix._ensure_jar("Test", "https://example.test/x.jar", str(jar), "0" * 64) is False
        assert calls == [], "a bad on-disk jar must be refused, not silently re-downloaded"

    def test_a_missing_jar_is_downloaded_and_verified(self, tmp_path, monkeypatch):
        jar = tmp_path / "lib.jar"

        def _fake_download_and_verify(url, dest_path, expected_sha256, **kwargs):
            with open(dest_path, "wb") as fh:
                fh.write(b"freshly downloaded jar")

        monkeypatch.setattr(setup_oculix, "download_and_verify", _fake_download_and_verify)

        assert setup_oculix._ensure_jar("Test", "https://example.test/x.jar", str(jar), "irrelevant-here") is True
        assert jar.read_bytes() == b"freshly downloaded jar"

    def test_a_download_that_fails_verification_is_refused(self, tmp_path, monkeypatch):
        jar = tmp_path / "lib.jar"

        def _fake_download_and_verify(url, dest_path, expected_sha256, **kwargs):
            raise HashMismatch("nope")

        monkeypatch.setattr(setup_oculix, "download_and_verify", _fake_download_and_verify)

        assert setup_oculix._ensure_jar("Test", "https://example.test/x.jar", str(jar), "0" * 64) is False

    def test_the_pinned_hashes_match_the_jars_already_in_this_repo(self):
        # These are computed from libs/oculixapi-3.0.3.jar and
        # libs/opencv-4.7.0-0.jar as they exist in this repo right now, not
        # invented - if this ever fails, the jars on disk changed and the
        # pinned hashes need updating alongside them, not the other way round.
        if not (os.path.isfile(setup_oculix.OCULIX_JAR_PATH) and os.path.isfile(setup_oculix.OPENCV_JAR_PATH)):
            import pytest
            pytest.skip("libs/ JARs are not present in this checkout")

        from verified_download import sha256_of

        assert sha256_of(setup_oculix.OCULIX_JAR_PATH) == setup_oculix.OCULIX_JAR_SHA256
        assert sha256_of(setup_oculix.OPENCV_JAR_PATH) == setup_oculix.OPENCV_JAR_SHA256
