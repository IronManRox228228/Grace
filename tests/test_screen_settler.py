"""Tests for screen settling before observation after page-changing actions.

From real-desktop-4:
- After any action that can change the page (press_key Enter/Return, click,
  type plus submit, open_app, activate), wait for the screen to settle before
  the next observation.
- Poll a cheap signal every 250 ms: a downscaled grayscale screenshot hash,
  plus the UIA element count if cheap.
- Proceed when two consecutive samples are stable (pixel diff under 1%).
- Cap at 3 s, then proceed anyway and note "screen may still be loading"
  in the planner's last-step result.
- Configurable settle time (OBSERVE_SETTLE_MAX_MS, default 3000).
- Log settle time in turn trace.
- Tests:
  - Mocked frames that change 3 times then stabilise give 2 stable reads before observing.
  - Never-stable frames hit the cap and add the note.
  - Zero overhead when the first two samples already match.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from grace.agent.screen_settler import (
    ScreenSettler,
    SettleResult,
    action_can_change_page,
    pixel_diff_ratio,
)
from grace.agent.loop import AgentLoop
from grace.agent.planner import PlannedStep
from grace.util.timing import start_turn, end_turn, current_trace


def make_frame(byte_val: int, size: int = 4096) -> bytes:
    """Helper to generate dummy grayscale pixel buffer."""
    return bytes([byte_val] * size)


class TestActionCanChangePage:
    def test_press_key_enter_and_return(self):
        assert action_can_change_page("press_key", {"key": "Return"}) is True
        assert action_can_change_page("cua_press_key", {"key": "Enter"}) is True
        assert action_can_change_page("cua_press_key", {"key": "enter_key"}) is True
        assert action_can_change_page("press_key", {"key": "Tab"}) is False
        assert action_can_change_page("press_key", {"key": "Down"}) is False

    def test_click_actions(self):
        assert action_can_change_page("click", {}) is True
        assert action_can_change_page("cua_click", {"target_name": "Submit"}) is True

    def test_type_text_with_submit(self):
        assert action_can_change_page("type_text", {"text": "hello", "press_enter": True}) is True
        assert action_can_change_page("cua_type_text", {"text": "hello", "submit": True}) is True
        assert action_can_change_page("type_text", {"text": "hello"}) is False

    def test_open_app_and_activate(self):
        assert action_can_change_page("open_app", {"name": "YouTube Music"}) is True
        assert action_can_change_page("launch", {"app": "Chrome"}) is True
        assert action_can_change_page("activate", {"window": "Edge"}) is True
        assert action_can_change_page("cua_activate", {"window": "Edge"}) is True

    def test_non_page_actions(self):
        assert action_can_change_page("search_files", {"query": "test"}) is False
        assert action_can_change_page("adjust_volume", {"level": 50}) is False
        assert action_can_change_page("remember", {"fact": "sister"}) is False
        assert action_can_change_page(None, {}) is False


class TestPixelDiffRatio:
    def test_identical_buffers(self):
        f = make_frame(100)
        assert pixel_diff_ratio(f, f) == 0.0

    def test_empty_buffers(self):
        assert pixel_diff_ratio(b"", b"") == 0.0

    def test_small_difference_under_1_percent(self):
        # 1 value difference out of 255 on a fraction of pixels
        b1 = bytes([100] * 1000)
        b2 = bytes([101] * 1000)
        # diff is 1 / 255 ~= 0.0039 (< 0.01)
        assert pixel_diff_ratio(b1, b2) < 0.01

    def test_large_difference_over_1_percent(self):
        b1 = bytes([0] * 1000)
        b2 = bytes([255] * 1000)
        assert pixel_diff_ratio(b1, b2) == 1.0


class TestScreenSettlerPolling:
    @pytest.mark.asyncio
    async def test_frames_change_3_times_then_stabilise_gives_2_stable_reads(self):
        """Mocked frames that change 3 times then stabilise give 2 stable reads before observing."""
        frame1 = make_frame(10)
        frame2 = make_frame(50)
        frame3 = make_frame(100)
        frame4 = make_frame(200)
        # Sequence: frame1 -> frame2 (change 1) -> frame3 (change 2) -> frame4 (change 3) -> frame4 (stable read 2)
        sequence = [
            (frame1, 10),
            (frame2, 12),
            (frame3, 15),
            (frame4, 20),
            (frame4, 20),
        ]
        call_idx = 0

        def sample_mock():
            nonlocal call_idx
            frame = sequence[min(call_idx, len(sequence) - 1)]
            call_idx += 1
            return frame

        # Virtual clock sleep
        sim_time = 0.0

        async def fast_sleep(s):
            nonlocal sim_time
            sim_time += s

        settler = ScreenSettler(
            max_ms=3000,
            poll_interval_ms=250,
            sample_fn=sample_mock,
            sleep_fn=fast_sleep,
        )

        with patch("time.monotonic", side_effect=lambda: sim_time):
            res = await settler.wait_for_settle()

        assert res.settled is True
        assert res.samples_taken == 5
        assert res.note == ""

    @pytest.mark.asyncio
    async def test_never_stable_frames_hit_cap_and_add_note(self):
        """Never-stable frames hit the cap and add the note."""
        sim_time = 0.0
        frame_val = 0

        def sample_changing():
            nonlocal frame_val
            frame_val = (frame_val + 50) % 255
            return make_frame(frame_val), frame_val

        async def fast_sleep(s):
            nonlocal sim_time
            sim_time += s

        settler = ScreenSettler(
            max_ms=3000,
            poll_interval_ms=250,
            sample_fn=sample_changing,
            sleep_fn=fast_sleep,
        )

        with patch("time.monotonic", side_effect=lambda: sim_time):
            res = await settler.wait_for_settle()

        assert res.settled is False
        assert res.elapsed_ms >= 3000
        assert res.note == "screen may still be loading"

    @pytest.mark.asyncio
    async def test_zero_overhead_when_first_two_samples_match(self):
        """Zero overhead when the first two samples already match."""
        frame = make_frame(128)
        reads = 0

        def sample_stable():
            nonlocal reads
            reads += 1
            return frame, 50

        sim_time = 0.0

        async def fast_sleep(s):
            nonlocal sim_time
            sim_time += s

        settler = ScreenSettler(
            max_ms=3000,
            poll_interval_ms=250,
            sample_fn=sample_stable,
            sleep_fn=fast_sleep,
        )

        with patch("time.monotonic", side_effect=lambda: sim_time):
            res = await settler.wait_for_settle()

        assert res.settled is True
        assert res.samples_taken == 2
        assert reads == 2
        assert res.note == ""

    @pytest.mark.asyncio
    async def test_settle_time_logged_in_turn_trace(self):
        """Log the settle time in the turn trace."""
        frame = make_frame(50)
        settler = ScreenSettler(
            max_ms=1000,
            poll_interval_ms=10,
            sample_fn=lambda: (frame, 10),
            sleep_fn=lambda s: asyncio.sleep(0.001),
        )

        with start_turn("test_turn") as trace:
            res = await settler.wait_for_settle()
            stages = [s for s in trace.stages if s.name == "settle_screen"]
            assert len(stages) == 1
            assert "settled=True" in (stages[0].detail or "")


class TestAgentLoopExpectationNoteScreenLoading:
    def test_expectation_note_adds_screen_loading_when_cap_hit(self):
        loop = AgentLoop.__new__(AgentLoop)
        step = PlannedStep(action="cua_press_key", params={"key": "Return"}, expect="Search results appear")
        last_result = {"status": "ok", "result": {"sent": True, "verified": True, "evidence": "pressed Return"}}
        snapshot_mock = MagicMock()
        snapshot_mock.active_window.title = "YouTube Music"
        snapshot_mock.graph = None

        note_stable = loop._expectation_note(
            step, last_result, snapshot_mock, screen_still_loading=False
        )
        assert "screen may still be loading" not in note_stable

        note_loading = loop._expectation_note(
            step, last_result, snapshot_mock, screen_still_loading=True
        )
        assert "screen may still be loading" in note_loading
