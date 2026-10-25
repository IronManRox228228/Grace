"""Tests for in-process run capture (grace-prompts/run-capture.md)."""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grace.harness.run_capture import RunCapture


class TestRunCapture:
    """Test suite for RunCapture object and capture hooks."""

    def test_run_capture_disabled_writes_nothing(self, tmp_path):
        """When capture is disabled, no files or directories are created."""
        capture_dir = tmp_path / "disabled_run"
        rc = RunCapture(run_dir=capture_dir, enabled=False)
        assert not rc.is_enabled

        # Calling hooks does nothing
        rc.start_logging()
        rc.record_intent(raw_output='{"tool": "test"}', intent={"tool": "test"}, route="fast_path")
        rc.record_screenshot(step_no=1, snapshot=MagicMock(), view=MagicMock())
        rc.record_planner(step_no=1, prompt="Test prompt", raw='{"action": "test"}')
        rc.record_step(
            step_no=1,
            action="test_action",
            params={"a": 1},
            grounded_point=(100, 200),
            result={"status": "ok"},
            snapshot=MagicMock(),
            timings={"duration": 0.5},
        )
        rc.record_summary(goal="test goal", status="complete", step_count=1, total_time=1.0)
        rc.close()

        assert not capture_dir.exists()

    def test_2_step_mocked_agent_run_writes_all_files(self, tmp_path):
        """A 2-step mocked agent run writes every specified file with correct content."""
        run_dir = tmp_path / "test_run"
        rc = RunCapture(run_dir=run_dir, enabled=True)
        assert rc.is_enabled

        rc.start_logging()
        logger = logging.getLogger("grace.test")
        logger.info("Test log line for run capture")

        rc.record_intent(
            raw_output='{"tool": "open_app", "name": "Calculator"}',
            intent={"tool": "open_app", "params": {"name": "Calculator"}},
            route="agentic_goal",
        )

        dummy_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa74e\xed\x00\x00\x00\x00IEND\xaeB`\x82"

        # Step 1
        snap1 = MagicMock()
        snap1.width = 1920
        snap1.height = 1080
        snap1.image_width = 1280
        snap1.image_height = 720
        snap1.png_bytes = dummy_png
        snap1.active_window = MagicMock(hwnd=101, title="Calculator")
        snap1.graph = MagicMock(elements=[
            MagicMock(source="uia"),
            MagicMock(source="uia"),
            MagicMock(source="ocr"),
        ])
        view1 = MagicMock(image_b64=None)

        rc.record_screenshot(step_no=1, snapshot=snap1, view=view1)
        rc.record_planner(step_no=1, prompt="Step 1 prompt", raw='{"action": "cua_click", "params": {"element_id": 1}}')
        rc.record_step(
            step_no=1,
            action="cua_click",
            params={"element_id": 1},
            grounded_point=(300, 400),
            result={"status": "ok", "clicked": True},
            snapshot=snap1,
            timings={"duration": 0.42},
        )

        # Step 2
        snap2 = MagicMock()
        snap2.width = 1920
        snap2.height = 1080
        snap2.image_width = 1280
        snap2.image_height = 720
        snap2.png_bytes = dummy_png
        snap2.active_window = MagicMock(hwnd=101, title="Calculator")
        snap2.graph = MagicMock(elements=[
            MagicMock(source="uia"),
            MagicMock(source="uia"),
        ])
        view2 = MagicMock(image_b64=None)

        rc.record_screenshot(step_no=2, snapshot=snap2, view=view2)
        rc.record_planner(step_no=2, prompt="Step 2 prompt", raw='{"action": "done", "params": {}}')
        rc.record_step(
            step_no=2,
            action="done",
            params={},
            grounded_point=None,
            result={"status": "complete"},
            snapshot=snap2,
            timings={"duration": 0.35},
        )

        rc.record_summary(
            goal="Open calculator and calculate",
            status="complete",
            step_count=2,
            total_time=1.85,
            model="holo3.1-4b",
            tier="dgpu",
        )
        rc.close()

        # Verify all files exist
        assert (run_dir / "log.txt").is_file()
        assert (run_dir / "intent.json").is_file()
        assert (run_dir / "summary.json").is_file()

        for step_idx in (1, 2):
            step_dir = run_dir / f"step_{step_idx:03d}"
            assert step_dir.is_dir()
            assert (step_dir / "screenshot.png").is_file()
            assert (step_dir / "screenshot_meta.json").is_file()
            assert (step_dir / "planner_raw.txt").is_file()
            assert (step_dir / "planner_prompt.txt").is_file()
            assert (step_dir / "step.json").is_file()

        # Check step_001 content
        meta1 = json.loads((run_dir / "step_001" / "screenshot_meta.json").read_text(encoding="utf-8"))
        assert meta1["original_size"] == [1920, 1080]
        assert meta1["sent_size"] == [1280, 720]
        assert abs(meta1["scale"] - (1280 / 1920)) < 1e-4

        raw1 = (run_dir / "step_001" / "planner_raw.txt").read_text(encoding="utf-8")
        assert "cua_click" in raw1

        prompt1 = (run_dir / "step_001" / "planner_prompt.txt").read_text(encoding="utf-8")
        assert "Step 1 prompt" in prompt1

        step1 = json.loads((run_dir / "step_001" / "step.json").read_text(encoding="utf-8"))
        assert step1["action"] == "cua_click"
        assert step1["params"] == {"element_id": 1}
        assert step1["grounded_point"] == [300, 400]
        assert step1["result"] == {"status": "ok", "clicked": True}
        assert step1["element_count"] == 3
        assert step1["element_sources"] == {"uia": 2, "ocr": 1}
        assert step1["window_title"] == "Calculator"
        assert step1["hwnd"] == 101
        assert "duration" in step1["timings"]

        # Check intent.json
        intent_data = json.loads((run_dir / "intent.json").read_text(encoding="utf-8"))
        assert "open_app" in intent_data["raw_output"]
        assert intent_data["route"] == "agentic_goal"

        # Check summary.json
        summary_data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary_data["goal"] == "Open calculator and calculate"
        assert summary_data["status"] == "complete"
        assert summary_data["step_count"] == 2
        assert summary_data["total_time"] == 1.85
        assert summary_data["model"] == "holo3.1-4b"
        assert summary_data["tier"] == "dgpu"
        assert "git_commit" in summary_data

        # Check log.txt
        log_content = (run_dir / "log.txt").read_text(encoding="utf-8")
        assert "Test log line for run capture" in log_content

    def test_redaction_holds_and_vault_secret_never_written(self, tmp_path):
        """Redaction applies to all files; fake vault secret typed through mocks never appears."""
        run_dir = tmp_path / "secret_run"
        fake_secret_val = "SuperSecretVaultVal987!"

        rc = RunCapture(run_dir=run_dir, enabled=True, vault_secrets=[fake_secret_val])
        rc.start_logging()

        logger = logging.getLogger("grace.test")
        logger.info(f"Typing secret for bank: {fake_secret_val}")

        # Intent contains password field indication
        rc.record_intent(
            raw_output=f'{{"tool": "type_saved_secret", "name": "bank", "leak": "{fake_secret_val}"}}',
            intent={"tool": "type_saved_secret", "params": {"name": "bank"}},
            route="agentic_goal",
        )

        dummy_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa74e\xed\x00\x00\x00\x00IEND\xaeB`\x82"
        snap = MagicMock()
        snap.width = 1920
        snap.height = 1080
        snap.image_width = 1280
        snap.image_height = 720
        snap.png_bytes = dummy_png
        snap.active_window = MagicMock(hwnd=102, title="Bank Login")
        snap.graph = MagicMock(elements=[
            MagicMock(source="uia", role="password", value=fake_secret_val),
        ])
        view = MagicMock(image_b64=None)

        rc.record_screenshot(step_no=1, snapshot=snap, view=view)
        rc.record_planner(
            step_no=1,
            prompt=f"Prompt with secret: {fake_secret_val}",
            raw=f'{{"action": "type_saved_secret", "params": {{"name": "bank", "text": "{fake_secret_val}"}}}}',
        )
        rc.record_step(
            step_no=1,
            action="type_saved_secret",
            params={"name": "bank", "text": fake_secret_val},
            grounded_point=None,
            result={"status": "ok", "message": f"secret typed: {fake_secret_val}"},
            snapshot=snap,
            timings={"duration": 0.1},
        )
        rc.record_summary(
            goal=f"Log in with {fake_secret_val}",
            status="complete",
            step_count=1,
            total_time=1.0,
        )
        rc.close()

        # Grep every file in the captured directory recursively for the secret value
        for root, _, files in os.walk(run_dir):
            for file in files:
                file_path = Path(root) / file
                if file.endswith(".png"):
                    continue
                content = file_path.read_text(encoding="utf-8")
                assert fake_secret_val not in content, f"Secret leaked in {file_path}:\n{content}"

    @pytest.mark.asyncio
    async def test_agent_loop_integration_with_run_capture(self, tmp_path):
        """AgentLoop.run with RunCapture writes full 2-step capture structure."""
        from grace.agent.loop import AgentLoop
        from grace.agent.perception import ScreenSnapshot, WindowInfo

        run_dir = tmp_path / "agent_loop_run"
        rc = RunCapture(run_dir=run_dir, enabled=True)
        rc.start_logging()

        # Step 1: click, Step 2: converse/done
        step1_json = '{"thought": "click 1", "action": "cua_click", "params": {"element_id": 1}, "expect": "next", "is_completed": false, "user_update": "Clicking"}'
        step2_json = '{"thought": "all done", "action": "converse", "params": {}, "expect": "done", "is_completed": true, "user_update": "Finished", "final_response": "All set"}'

        dummy_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa74e\xed\x00\x00\x00\x00IEND\xaeB`\x82"
        mock_graph = MagicMock()
        mock_graph.actionable_count = 10
        mock_graph.elements = [MagicMock(source="uia", id=1)]
        mock_graph.to_prompt = MagicMock(return_value="[1] button 'OK'")
        mock_graph.__len__ = MagicMock(return_value=1)

        snap1 = ScreenSnapshot(
            active_window=WindowInfo(hwnd=10, title="TestApp", class_name="W", rect=(0, 0, 800, 600)),
            ocr_lines=[],
            width=1920,
            height=1080,
            image_width=1280,
            image_height=720,
            png_bytes=dummy_png,
            graph=mock_graph,
        )

        gemma = AsyncMock()
        gemma.generate_text.side_effect = [step1_json, step2_json]
        dispatcher = AsyncMock()
        dispatcher.execute.return_value = {"status": "ok", "result": {"ok": True, "verified": True}}
        perception = MagicMock()
        perception.capture_snapshot_async.return_value = snap1
        perception.capture_snapshot.return_value = snap1

        loop = AgentLoop(
            gemma=gemma,
            dispatcher=dispatcher,
            perception=perception,
            run_capture=rc,
        )

        res = await loop.run(user_goal="Do the 2-step test task")
        rc.close()

        assert res["status"] == "ok"
        assert (run_dir / "summary.json").is_file()
        assert (run_dir / "step_001" / "screenshot.png").is_file()
        assert (run_dir / "step_001" / "screenshot_meta.json").is_file()
        assert (run_dir / "step_001" / "step.json").is_file()
        assert (run_dir / "step_001" / "planner_raw.txt").is_file()
        assert (run_dir / "step_001" / "planner_prompt.txt").is_file()

        assert (run_dir / "step_002" / "screenshot.png").is_file()
        assert (run_dir / "step_002" / "step.json").is_file()

        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["goal"] == "Do the 2-step test task"
        assert summary["step_count"] == 2

    def test_save_secret_flow_never_captured(self, tmp_path, monkeypatch):
        """--save-secret flow exits before initializing capture or creating capture dirs."""
        import sys
        from grace.main import main

        fake_run_dir = tmp_path / "never_created"
        test_args = [
            "grace.exe",
            "--save-secret", "bank",
            "--site", "bank.com",
            "--capture-run", str(fake_run_dir),
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        with patch("getpass.getpass", side_effect=["pass123", "pass123"]):
            with patch("grace.agent.memory.PersistentMemoryStore.save_secret", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0

        # Assert capture directory was NEVER created
        assert not fake_run_dir.exists()
