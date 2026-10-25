import logging
import pytest
from grace.agent.perception import ScreenSnapshot, WindowInfo
from grace.agent.planner import Planner
from grace.perception.elements import (
    ElementNode,
    elements_to_prompt,
    is_password_element,
)


class FakeLLM:
    def __init__(self):
        self.last_prompt = ""

    async def generate_text(self, prompt, system_prompt, **kwargs):
        self.last_prompt = prompt
        return '{"action": "converse", "params": {"response": "done"}, "is_completed": true}'


def test_filled_and_empty_password_field_render_state():
    """A filled password field renders as filled, and an empty one renders as empty."""
    filled_node = ElementNode(
        id=1,
        role="edit",
        name="Password",
        rect=(100, 100, 300, 140),
        center=(200, 120),
        value="SecretP@ssw0rd!",
        is_password=True,
    )
    empty_node = ElementNode(
        id=2,
        role="password",
        name="Confirm Password",
        rect=(100, 150, 300, 190),
        center=(200, 170),
        value="",
    )

    # In to_dict (compact and full)
    filled_dict = filled_node.to_dict(compact=True)
    empty_dict = empty_node.to_dict(compact=True)

    assert filled_dict.get("value") == "[password field, filled]"
    assert empty_dict.get("value") == "[password field, empty]"

    # In elements_to_prompt
    rendered = elements_to_prompt([filled_node, empty_node])
    assert "[password field, filled]" in rendered
    assert "[password field, empty]" in rendered
    assert "SecretP@ssw0rd!" not in rendered


def test_password_value_never_appears_in_prompt_or_logs(caplog):
    """The password value never appears in the planner prompt or the logs."""
    caplog.set_level(logging.DEBUG)
    logger = logging.getLogger("grace.test_password_redaction")
    raw_secret = "UltraSecretHunter2!"

    node = ElementNode(
        id=10,
        role="password",
        name="User Password",
        rect=(50, 50, 250, 90),
        center=(150, 70),
        value=raw_secret,
    )

    logger.debug("Captured password node: %s", node.to_dict())
    elements_text = elements_to_prompt([node])
    llm = FakeLLM()
    planner = Planner(llm)
    prompt = planner.build_prompt(
        goal="sign in to website",
        elements_prompt=elements_text,
    )
    logger.debug("Built planner prompt: %s", prompt)

    assert raw_secret not in prompt
    assert raw_secret not in caplog.text
    assert "[password field, filled]" in prompt


def test_words_containing_pin_substring_are_not_passwords():
    for text in ["mapping", "shopping", "spinner", "Pinned tabs"]:
        assert not is_password_element(name=text), f"{text} should not be a password element"
        node = ElementNode(
            id=1,
            role="text",
            name=text,
            rect=(0, 0, 10, 10),
            center=(5, 5),
            value=text,
        )
        assert not node.is_password, f"ElementNode with name '{text}' should not have is_password=True"
        assert node.value == text


def test_whole_word_and_tokenized_password_hints():
    # True positives: "PIN", "Enter PIN", "Password"
    for text in ["PIN", "Enter PIN", "Password", "pin code"]:
        assert is_password_element(name=text), f"'{text}' should be recognized as password"
    # Automation IDs: pinInput, txtPassword
    assert is_password_element(automation_id="pinInput"), "pinInput should be recognized as password"
    assert is_password_element(automation_id="txtPassword"), "txtPassword should be recognized as password"


def test_ocr_element_is_never_password_field():
    ocr_node = ElementNode(
        id=99,
        role="text",
        name="Password",
        rect=(0, 0, 10, 10),
        center=(5, 5),
        value="Password",
        source="ocr",
    )
    assert not ocr_node.is_password, "OCR element named Password must not be marked as a password field"
    assert ocr_node.value == "Password"
