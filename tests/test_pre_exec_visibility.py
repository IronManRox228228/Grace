"""Tests for pre-exec visibility and post-launch browser activation (real-desktop-3.md Item 1)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grace.agent.loop import AgentLoop
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.intent.parser import Intent


class TestPreExecVisibility:
    """Test suite for Item 1 of real-desktop-3.md."""

    @pytest.mark.asyncio
    async def test_pre_exec_step_0_appears_in_first_planner_prompt(self):
        """Pre-exec action and result must appear as step 0 in the planner's history on step 1."""
        dummy_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa74e\xed\x00\x00\x00\x00IEND\xaeB`\x82"
        mock_graph = MagicMock()
        mock_graph.actionable_count = 10
        mock_graph.elements = [MagicMock(source="uia", id=1)]
        mock_graph.to_prompt = MagicMock(return_value="[1] button 'OK'")
        mock_graph.__len__ = MagicMock(return_value=1)

        snap = ScreenSnapshot(
            active_window=WindowInfo(hwnd=10, title="Terminal", class_name="W", rect=(0, 0, 800, 600)),
            ocr_lines=[],
            width=1920,
            height=1080,
            image_width=1280,
            image_height=720,
            png_bytes=dummy_png,
            graph=mock_graph,
        )

        recorded_histories = []

        gemma = AsyncMock()

        async def fake_generate_text(prompt, *args, **kwargs):
            recorded_histories.append(prompt)
            return '{"thought": "done", "action": "converse", "params": {}, "is_completed": true, "user_update": "done"}'

        gemma.generate_text.side_effect = fake_generate_text

        dispatcher = AsyncMock()
        dispatcher.execute.return_value = {"status": "ok", "result": {"ok": True}}
        perception = MagicMock()
        perception.capture_snapshot_async.return_value = snap
        perception.capture_snapshot.return_value = snap

        loop = AgentLoop(
            gemma=gemma,
            dispatcher=dispatcher,
            perception=perception,
        )

        pre_exec = {
            "action": "open_app",
            "params": {"name": "YouTube Music", "url": "music.youtube.com"},
            "result": {"status": "ok", "text": "Launched YouTube Music"},
        }

        res = await loop.run(
            user_goal="open youtube music and play signal",
            pre_exec_step=pre_exec,
        )
        assert res["status"] == "ok"
        assert len(recorded_histories) >= 1
        first_prompt = recorded_histories[0]
        # Step 0 must appear in the first prompt
        assert "Step 0: open_app" in first_prompt, f"Step 0 missing from first prompt:\n{first_prompt}"
        assert "music.youtube.com" in first_prompt

    @pytest.mark.asyncio
    async def test_activation_attempted_after_pre_exec(self):
        """After successful pre-exec open_app with a url, browser window activation is attempted."""
        from grace.main import GraceApp

        app = GraceApp(tts_off=True)

        pre_intent = Intent(
            tool="open_app",
            params={"name": "YouTube Music", "url": "music.youtube.com"},
        )

        executed_intents = []

        async def fake_dispatcher_execute(intent):
            executed_intents.append(intent)
            if intent.tool == "open_app":
                return {"status": "ok", "text": "Launched Edge"}
            elif intent.tool == "cua_activate":
                return {"status": "ok", "result": {"ok": True, "verified": True}}
            elif intent.tool == "cua_list_windows":
                return {
                    "ok": True,
                    "windows": [
                        {"id": 100, "title": "Terminal", "pid": 1},
                        {"id": 200, "title": "YouTube Music - Personal - Microsoft Edge", "pid": 2},
                    ],
                }
            return {"status": "ok"}

        app.dispatcher.execute = AsyncMock(side_effect=fake_dispatcher_execute)
        app.gemma.generate_intent = AsyncMock(return_value='{"tool": "open_app", "name": "YouTube Music", "url": "music.youtube.com"}')
        app._run_agent_goal = AsyncMock(return_value={"status": "ok", "steps": []})

        trace = MagicMock()
        trace.stage.return_value.__aenter__ = AsyncMock()
        trace.stage.return_value.__aexit__ = AsyncMock()

        await app._route_transcript(trace, "open youtube music in my browser and play signal")

        # Verify that cua_activate was called for the browser window before or during setup
        activate_calls = [i for i in executed_intents if getattr(i, "tool", None) == "cua_activate"]
        assert len(activate_calls) >= 1, f"Expected cua_activate to be called after pre-exec, got: {executed_intents}"
        assert any(
            c.params.get("id") == 200 or "YouTube Music" in str(c.params.get("window_title", ""))
            for c in activate_calls
        )
