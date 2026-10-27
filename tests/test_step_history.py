import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from grace.agent.loop import AgentLoop
from grace.agent.memory import AgentMemory, StepRecord
from grace.agent.planner import Planner, wrap_untrusted
from grace.agent.perception import ScreenSnapshot, WindowInfo


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.prompts = []

    async def generate_text(self, prompt, system_prompt, **kwargs):
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return '{"action": "converse", "params": {"response": "done"}, "is_completed": true}'


def test_read_pdf_result_reaches_next_planner_prompt():
    """After a read_pdf step whose result has '14 November', the next planner prompt contains '14 November'."""
    llm = FakeLLM([
        '{"thought": "read pdf", "action": "read_pdf", "params": {"path": "report.pdf"}, "expect": "pdf read"}',
        '{"thought": "write note", "action": "converse", "params": {"response": "done"}, "is_completed": true}',
    ])
    planner = Planner(llm)
    dispatcher = AsyncMock()
    perception = MagicMock()

    loop = AgentLoop(
        gemma=llm,
        dispatcher=dispatcher,
        perception=perception,
        planner=planner,
    )

    from grace.perception.element_graph import ElementGraph
    win = WindowInfo(0, "Notepad", "Notepad", (0, 0, 100, 100))
    eg = ElementGraph(elements=[], window=win)

    async def fake_observe():
        return ScreenSnapshot(
            active_window=win,
            ocr_lines=[],
            width=1920,
            height=1080,
            image_width=1920,
            image_height=1080,
            dpi_scale=1.0,
            graph=eg,
        )

    loop._observe = fake_observe

    async def fake_dispatch(step, mem, confirmed=False):
        if step.action == "read_pdf":
            return {
                "status": "ok",
                "text": "Project Report\nFinal submission deadline: 14 November\nPlease submit deliverables on time.",
            }
        return {"status": "ok"}

    loop._dispatch = fake_dispatch

    res = asyncio.run(loop.run(user_goal="find deadline in report PDF"))

    assert len(llm.prompts) >= 2
    step_2_prompt = llm.prompts[1]
    assert "14 November" in step_2_prompt
    # Cockpit 8, single mode: the three old sections collapse into one
    # `### Your last steps` block.
    assert "### Your last steps" in step_2_prompt


def test_history_is_capped_at_8_steps_and_1200_chars():
    """The history is capped at the last 8 steps and about 1200 characters, older steps dropping off first."""
    memory = AgentMemory(user_goal="test capping")

    for i in range(1, 15):
        memory.add_step(
            thought=f"thought {i}",
            action=f"action_{i}",
            params={"path": f"C:\\path\\to\\file_{i}.txt", "text": f"long text content for step {i} " * 5},
            result={"status": "ok", "text": f"Extracted output from tool step {i} with lots of detailed information " * 4},
        )

    history = memory.format_history_markdown()

    # Oldest steps (1-6) must have dropped off:
    assert "action_1(" not in history
    assert "action_2(" not in history
    assert "action_6(" not in history

    # Recent steps must be present:
    assert "action_14" in history

    # Max 8 steps:
    step_lines = [line for line in history.splitlines() if line.strip().startswith("Step ")]
    assert len(step_lines) <= 8

    # Length capped at about 1200 chars:
    assert len(history) <= 1250


def test_history_is_wrapped_as_untrusted():
    """The history in the planner prompt is wrapped in wrap_untrusted markers."""
    memory = AgentMemory(user_goal="test wrapping")
    memory.add_step(
        thought="read file",
        action="read_pdf",
        params={"path": "report.pdf"},
        result={"status": "ok", "text": "Deadline: 14 November"},
    )
    history = memory.format_history_markdown()

    llm = FakeLLM()
    planner = Planner(llm)
    prompt = planner.build_prompt(
        goal="test",
        elements_prompt="none",
        history=history,
    )

    # Cockpit 8 (single mode, the default prompt) renders history under
    # `### Your last steps`, still wrapped as untrusted data.
    assert "### Your last steps" in prompt
    assert "### What you have already done" not in prompt
    assert "<<<UNTRUSTED_DATA_START" in prompt
    assert "<<<UNTRUSTED_DATA_END>>>" in prompt
    assert "Deadline: 14 November" in prompt


def test_history_redacts_secrets_and_omits_confirmation_answers():
    """History must never contain secrets and never include confirmation answers."""
    memory = AgentMemory(user_goal="test security")
    # Step with password
    memory.add_step(
        thought="enter password",
        action="cua_type_text",
        params={"text": "SuperSecretPass123!", "role": "password"},
        result={"status": "ok", "message": "typed"},
    )
    # Step with API key in result
    memory.add_step(
        thought="check key",
        action="search_files",
        params={"query": "keys.env"},
        result={"status": "ok", "text": "Found key: sk-abcdef12345678901234567890"},
    )
    # Step with confirmation answer
    memory.add_step(
        thought="confirm action",
        action="safety_confirm",
        params={"confirmed": True, "text": "yes"},
        result={"status": "ok"},
    )

    history = memory.format_history_markdown()

    assert "SuperSecretPass123!" not in history
    assert "<redacted>" in history
    assert "sk-abcdef12345678901234567890" not in history
    assert "confirmed=True" not in history
    assert "text='yes'" not in history
    assert 'text="yes"' not in history


def test_history_click_line_includes_both_message_and_evidence():
    """A click whose result has both message and evidence renders both, joined with ' - '."""
    memory = AgentMemory(user_goal="send a message")
    memory.add_step(
        thought="click mic button",
        action="cua_click",
        params={"x": 1774, "y": 1049},
        result={"status": "ok",
                "result": {"message": "Clicked 1 time(s) at (1774, 1049) via explicit",
                           "evidence": "keyboard focus moved to [32] 'Voice message'"}},
    )
    history = memory.format_history_markdown()
    line = [l for l in history.splitlines() if "cua_click" in l][0]
    assert "Clicked 1 time(s) at (1774, 1049) via explicit" in line
    assert "keyboard focus moved to [32] 'Voice message'" in line
    assert " - " in line


def test_history_trim_shortens_message_before_evidence():
    """With a long message, the 160-char trim eats the message first; the evidence survives."""
    memory = AgentMemory(user_goal="send a message")
    long_message = "Clicked 1 time(s) at (1774, 1049) via explicit after settling and retrying with a great deal of additional padding text to push the length up"
    evidence = "keyboard focus moved to [32] 'Voice message'"
    assert len(long_message) + len(evidence) > 160
    memory.add_step(
        thought="click mic button",
        action="cua_click",
        params={"x": 1774, "y": 1049},
        result={"status": "ok", "result": {"message": long_message, "evidence": evidence}},
    )
    history = memory.format_history_markdown()
    line = [l for l in history.splitlines() if "cua_click" in l][0]
    assert evidence in line
    assert len(line) <= 160 + len(f"Step 1: cua_click(x=1774, y=1049) -> ")


def test_history_click_line_message_only_still_works():
    """A result with a message but no evidence still renders the message alone."""
    memory = AgentMemory(user_goal="send a message")
    memory.add_step(
        thought="click",
        action="cua_click",
        params={"x": 1},
        result={"status": "ok", "result": {"message": "Clicked 1 time(s) at (1, 0) via explicit"}},
    )
    history = memory.format_history_markdown()
    assert "Clicked 1 time(s) at (1, 0) via explicit" in history
