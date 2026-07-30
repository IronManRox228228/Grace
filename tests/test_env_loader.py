"""Tests for .env loading.

The two rules under test exist because of real breakage: the project .env has
historically carried empty keys that would blank working config.py defaults,
and the real environment must stay authoritative over the file.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.env_loader import find_env_file, load_env, parse_env_file


class TestParseEnvFile:
    def test_basic_key_values(self):
        assert parse_env_file("A=1\nB=two\n") == {"A": "1", "B": "two"}

    def test_comments_and_blanks_ignored(self):
        text = "# a comment\n\nA=1\n   \n# another\nB=2\n"
        assert parse_env_file(text) == {"A": "1", "B": "2"}

    def test_empty_values_are_dropped(self):
        # The whole point: WHISPER_MODEL_PATH= must not blank the default.
        assert parse_env_file("A=\nB=   \nC=3\n") == {"C": "3"}

    def test_windows_paths_survive_intact(self):
        text = r"KOKORO_VOICES_PATH=C:\Users\Someone\.cache\voices\af_bella.pt"
        parsed = parse_env_file(text)
        assert parsed["KOKORO_VOICES_PATH"] == r"C:\Users\Someone\.cache\voices\af_bella.pt"

    def test_value_may_contain_equals(self):
        assert parse_env_file("URL=http://x/?a=1&b=2")["URL"] == "http://x/?a=1&b=2"

    def test_surrounding_quotes_stripped(self):
        assert parse_env_file("A='one'\nB=\"two\"\n") == {"A": "one", "B": "two"}

    def test_mismatched_quotes_left_alone(self):
        assert parse_env_file("A='one\"")["A"] == "'one\""

    def test_export_prefix_supported(self):
        assert parse_env_file("export A=1\n") == {"A": "1"}

    def test_lines_without_equals_ignored(self):
        assert parse_env_file("garbage line\nA=1\n") == {"A": "1"}

    def test_blank_key_ignored(self):
        assert parse_env_file("=novalue\nA=1\n") == {"A": "1"}

    def test_whitespace_around_key_and_value_trimmed(self):
        assert parse_env_file("  A  =  1  \n") == {"A": "1"}


class TestLoadEnv:
    def test_applies_values_to_environ(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("GRACE_TEST_KEY=hello\n", encoding="utf-8")
        monkeypatch.delenv("GRACE_TEST_KEY", raising=False)

        applied = load_env(str(env))

        assert applied == {"GRACE_TEST_KEY": "hello"}
        assert os.environ["GRACE_TEST_KEY"] == "hello"

    def test_existing_environ_wins(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("GRACE_TEST_KEY=from_file\n", encoding="utf-8")
        monkeypatch.setenv("GRACE_TEST_KEY", "from_shell")

        applied = load_env(str(env))

        assert applied == {}
        assert os.environ["GRACE_TEST_KEY"] == "from_shell"

    def test_empty_value_does_not_clobber_environ(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("GRACE_TEST_KEY=\n", encoding="utf-8")
        monkeypatch.delenv("GRACE_TEST_KEY", raising=False)

        load_env(str(env))

        assert "GRACE_TEST_KEY" not in os.environ

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_env(str(tmp_path / "nope.env")) == {}

    def test_unreadable_file_is_not_an_error(self, tmp_path):
        # A directory where a file is expected: open() raises, we degrade.
        d = tmp_path / "weird.env"
        d.mkdir()
        assert load_env(str(d)) == {}


class TestFindEnvFile:
    def test_finds_env_by_walking_up(self, tmp_path):
        (tmp_path / ".env").write_text("A=1\n", encoding="utf-8")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)

        assert find_env_file(str(nested)) == str(tmp_path / ".env")

    def test_returns_none_when_absent(self, tmp_path):
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert find_env_file(str(nested)) is None


class TestRealProjectEnv:
    def test_project_env_has_no_empty_values(self):
        """The shipped .env.example must not reintroduce blank-value landmines
        for keys that have a meaningful config.py default."""
        root = os.path.join(os.path.dirname(__file__), "..")
        example = os.path.join(root, ".env.example")
        with open(example, "r", encoding="utf-8") as fh:
            text = fh.read()

        parsed = parse_env_file(text)
        # GEMINI_API_KEY is intentionally blank in the template and therefore
        # must be filtered out rather than exported as "".
        assert "GEMINI_API_KEY" not in parsed
        assert parsed["KOKORO_WORKERS"] == "2"
        assert parsed["GEMINI_MODEL_NAME"] == "gemini-3.1-flash-lite"

    def test_example_carries_no_secret(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        with open(os.path.join(root, ".env.example"), "r", encoding="utf-8") as fh:
            text = fh.read()
        # .env.example is tracked by git; a real key must never land in it.
        assert "AIza" not in text
