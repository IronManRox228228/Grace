"""Unit tests for PersonalCorrectionList (Task 10, PLAN §6)."""

from pathlib import Path
import pytest

from grace.stt.corrections import PersonalCorrectionList


class TestPersonalCorrectionList:
    def test_add_and_apply_correction(self, tmp_path: Path):
        corr_file = tmp_path / "corrections.json"
        corr = PersonalCorrectionList(filepath=corr_file)

        corr.add_correction("crow", "Chrome")
        assert corr.get_all() == {"crow": "Chrome"}

        res = corr.apply("open crow now")
        assert res == "open Chrome now"

    def test_persistence_across_instances(self, tmp_path: Path):
        corr_file = tmp_path / "corrections.json"
        corr1 = PersonalCorrectionList(filepath=corr_file)
        corr1.add_correction("fire fox", "Firefox")

        # Load in a fresh instance
        corr2 = PersonalCorrectionList(filepath=corr_file)
        assert corr2.get_all() == {"fire fox": "Firefox"}
        assert corr2.apply("launch fire fox") == "launch Firefox"

    def test_explicit_correction_detection(self):
        corr = PersonalCorrectionList()

        # Pattern 1: "no, Chrome" following "open crow"
        det1 = corr.detect_explicit_correction("no, Chrome", previous_transcript="open crow")
        assert det1 == ("crow", "Chrome")

        # Pattern 2: "not crow, Chrome"
        det2 = corr.detect_explicit_correction("not crow, Chrome")
        assert det2 == ("crow", "Chrome")

        # Pattern 3: "not crow but Chrome"
        det3 = corr.detect_explicit_correction("not crow but Chrome")
        assert det3 == ("crow", "Chrome")

        # Pattern 4: "I meant Chrome" following "open crow"
        det4 = corr.detect_explicit_correction("I meant Chrome", previous_transcript="open crow")
        assert det4 == ("crow", "Chrome")

        # Ordinary command -> None
        assert corr.detect_explicit_correction("open Chrome") is None

    def test_voice_reversal(self, tmp_path: Path):
        corr_file = tmp_path / "corrections.json"
        corr = PersonalCorrectionList(filepath=corr_file)
        corr.add_correction("crow", "Chrome")
        assert corr.apply("open crow") == "open Chrome"

        # User asks: "forget that correction"
        target = corr.detect_forget_request("forget that correction")
        assert target == "*"

        removed = corr.remove_last_correction()
        assert removed == "crow"
        assert corr.get_all() == {}
        assert corr.apply("open crow") == "open crow"

    def test_voice_reversal_by_name(self, tmp_path: Path):
        corr_file = tmp_path / "corrections.json"
        corr = PersonalCorrectionList(filepath=corr_file)
        corr.add_correction("spot", "Spotify")
        corr.add_correction("noted", "Notepad")

        target = corr.detect_forget_request("remove correction for spot")
        assert target == "spot"

        assert corr.remove_correction("spot") is True
        assert "spot" not in corr.get_all()
        assert "noted" in corr.get_all()

    def test_whole_word_matching_boundary(self):
        corr = PersonalCorrectionList()
        corr.add_correction("cat", "Chrome")
        # Should NOT match "concatenate" or "scat"
        assert corr.apply("scat concatenate cat") == "scat concatenate Chrome"
