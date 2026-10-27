import logging
import pytest
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.agent.planner import Planner, wrap_untrusted
from grace.perception.element_graph import ElementGraph
from grace.perception.elements import (
    ElementNode,
    elements_to_prompt,
    is_password_element,
    redact_element_value_if_secret,
)
from grace.harness.redact import redact_data


class FakeLLM:
    def __init__(self):
        self.last_prompt = ""

    async def generate_text(self, prompt, system_prompt, **kwargs):
        self.last_prompt = prompt
        return '{"action": "converse", "params": {"response": "done"}, "is_completed": true}'


def test_ordinary_search_box_value_reaches_planner_prompt():
    """An ordinary search box value reaches the planner prompt."""
    node = ElementNode(
        id=2,
        role="edit",
        name="Search songs, albums, artists",
        rect=(650, 60, 1200, 100),
        center=(925, 80),
        value="Signal by Home",
        placeholder="Search songs, albums, artists",
    )
    elements_text = elements_to_prompt([node])
    llm = FakeLLM()
    planner = Planner(llm, is_single=False)
    prompt = planner.build_prompt(
        goal="search for the song Signal by Home",
        elements_prompt=elements_text,
    )
    assert "Signal by Home" in prompt
    assert "<<<UNTRUSTED_DATA_START" in prompt


def test_address_bar_domain_reaches_planner_prompt_and_not_redacted_on_tape():
    """Ordinary domain names in address bar stay visible to planner and in snapshots."""
    node = ElementNode(
        id=0,
        role="edit",
        name="Address and search bar",
        rect=(120, 60, 1700, 96),
        center=(910, 78),
        value="music.youtube.com",
        placeholder="Search or enter web address",
    )
    elements_text = elements_to_prompt([node])
    llm = FakeLLM()
    planner = Planner(llm, is_single=False)
    prompt = planner.build_prompt(
        goal="open youtube music",
        elements_prompt=elements_text,
    )
    assert "music.youtube.com" in prompt

    # Also verified on tape snapshot redaction:
    snap_data = {"id": 0, "role": "edit", "name": "Address and search bar", "rect": [120, 60, 1700, 96], "value": "music.youtube.com"}
    redacted = redact_data(snap_data)
    assert redacted["value"] == "music.youtube.com"


def test_password_field_value_never_reaches_planner_prompt_and_never_reaches_logs(caplog):
    """A password field value never reaches the planner prompt, and never reaches the logs."""
    caplog.set_level(logging.DEBUG)
    logger = logging.getLogger("grace.test_redact")
    secret_pass = "SuperSecretPassword123!"

    # 1. Element by role='password'
    node_pwd = ElementNode(
        id=5,
        role="password",
        name="Account Password",
        rect=(100, 100, 300, 140),
        center=(200, 120),
        value=secret_pass,
    )
    logger.debug("Captured element: %s", node_pwd.to_dict())

    # 2. Element by name hint 'pin' or 'password'
    node_pin = ElementNode(
        id=6,
        role="edit",
        name="Security PIN",
        rect=(100, 150, 300, 190),
        center=(200, 170),
        value="9876",
    )
    logger.debug("Captured element: %s", node_pin.to_dict())

    # 3. Element with secret-shaped value (e.g. API key)
    api_key = "sk-1234567890abcdef1234567890"
    node_key = ElementNode(
        id=7,
        role="edit",
        name="API Key",
        rect=(100, 200, 300, 240),
        center=(200, 220),
        value=api_key,
    )
    logger.debug("Captured element: %s", node_key.to_dict())

    elements_text = elements_to_prompt([node_pwd, node_pin, node_key])
    llm = FakeLLM()
    planner = Planner(llm, is_single=False)
    prompt = planner.build_prompt(
        goal="log in to account",
        elements_prompt=elements_text,
    )

    # Prompt check: None of the secrets reach the prompt
    assert secret_pass not in prompt
    assert "9876" not in prompt
    assert api_key not in prompt
    assert "<redacted>" in prompt

    # Log check: None of the secrets reach the logs
    assert secret_pass not in caplog.text
    assert "9876" not in caplog.text
    assert api_key not in caplog.text
