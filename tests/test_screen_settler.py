"""Tests for screen settling before observation after page-changing actions.

From real-desktop-4:
- After any action that can change the page (press_key Enter/Return, click,
  type plus submit, open_app, activate), wait for the screen to settle before
  the next observation.
- Poll a cheap signal every 250 ms: a downscaled grayscale screenshot hash,
  plus the UIA element count if cheap.
- Proceed only when the samples stayed stable for a continuous window of at
  least 1000 ms (several consecutive polls) - one matching pair is not
  enough: a loading YouTube watch page shows a black player that looks
  identical on two quick samples.
- Wait at least 400 ms before the first sample after a page-changing action.
- Cap at 3 s, then proceed anyway and note "screen may still be loading"
  in the planner's last-step result.
- Configurable settle time (OBSERVE_SETTLE_MAX_MS, default 3000).
- Log settle time in turn trace.
- Tests:
  - Frames that change 3 times then stabilise settle after the 1000 ms window.
  - Never-stable frames hit the cap and add the note.
  - Stable from the first sample settles at the 1000 ms window, not at the 2nd poll.
  - Stable for 2 polls then a change must not count as settled at the 2nd poll.
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
    async def test_frames_change_3_times_then_stabilise_settles_after_the_window(self):
        """Mocked frames that change 3 times then stabilise settle once they
        stayed unchanged for a continuous 1000 ms window, not on the first
        matching pair."""
        frame1 = make_frame(10)
        frame2 = make_frame(50)
        frame3 = make_frame(100)
        frame4 = make_frame(200)
        # Sequence: frame1 -> frame2 -> frame3 -> frame4 (3 changes), then
        # frame4 stays on screen for the whole stability window.
        sequence = [frame1, frame2, frame3, frame4]
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

        # Samples: frame1@400, frame2@650, frame3@900, frame4@1150, frame4@1400,
        # frame4@1650, frame4@1900, frame4@2150 -> the 1000 ms window (1150..2150)
        # completes at the 8th sample.
        assert res.settled is True
        assert res.samples_taken == 8
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
    async def test_stable_from_the_first_sample_settles_at_the_1000ms_window(self):
        """Identical frames from the start settle at the 1000 ms window, not
        when the first two samples match (~250 ms)."""
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

        # Samples at 400, 650, 900, 1150, 1400: the window 400..1400 ms is
        # 1000 ms of stability, so the 5th sample settles.
        assert res.settled is True
        assert res.samples_taken == 5
        assert reads == 5
        assert res.note == ""

    @pytest.mark.asyncio
    async def test_stable_for_two_polls_then_a_change_is_not_settled_at_poll_2(self):
        """A fake sampler stable for 2 polls and then changing must not count
        as settled at the 2nd poll (the old one-pair rule settled here)."""
        stable_frame = make_frame(50)
        other_frame = make_frame(10)
        sequence = [stable_frame, stable_frame, other_frame]
        call_idx = 0

        def sample_mock():
            nonlocal call_idx
            if call_idx < len(sequence):
                frame = sequence[call_idx]
            else:
                # Keep changing after the short stable run so it cannot settle
                # on a later pair either.
                frame = make_frame((call_idx * 37) % 255)
            call_idx += 1
            return frame

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

        # The first pair (poll 1 and poll 2) is stable, but 250 ms of stability
        # is short of the 1000 ms window, and the 3rd sample changes anyway.
        assert res.settled is False
        assert res.note == "screen may still be loading"
        assert res.samples_taken > 2

    @pytest.mark.asyncio
    async def test_settler_disabled_max_ms_zero_no_sleep_no_sample(self):
        """max_ms=0 returns immediately without sleeping or sampling."""
        samples_called = 0
        sleep_called = False

        def sample_mock():
            nonlocal samples_called
            samples_called += 1
            return make_frame(50)

        async def fast_sleep(s):
            nonlocal sleep_called
            sleep_called = True

        settler = ScreenSettler(
            max_ms=0,
            sample_fn=sample_mock,
            sleep_fn=fast_sleep,
        )

        res = await settler.wait_for_settle()
        assert res.settled is True
        assert res.elapsed_ms == 0.0
        assert res.samples_taken == 0
        assert res.note == ""
        assert samples_called == 0
        assert sleep_called is False

    @pytest.mark.asyncio
    async def test_settler_cap_under_400ms_total_wait_bounded(self):
        """max_ms=200 initial wait is capped so total wait <= 200 ms."""
        sim_time = 0.0
        samples_called = 0

        def sample_mock():
            nonlocal samples_called
            samples_called += 1
            return make_frame(samples_called * 20)

        async def fast_sleep(s):
            nonlocal sim_time
            sim_time += s

        settler = ScreenSettler(
            max_ms=200,
            poll_interval_ms=250,
            sample_fn=sample_mock,
            sleep_fn=fast_sleep,
        )

        with patch("time.monotonic", side_effect=lambda: sim_time):
            res = await settler.wait_for_settle()

        assert sim_time <= 0.200001
        assert res.elapsed_ms <= 200.000001
        assert res.settled is False
        assert res.samples_taken == 1
        assert res.note == "screen may still be loading"

    @pytest.mark.asyncio
    async def test_settle_time_logged_in_turn_trace(self):
        """Log the settle time in the turn trace."""
        frame = make_frame(50)
        settler = ScreenSettler(
            max_ms=3000,
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
