"""Cockpit 6 item 5: deterministic JSON repair before parsing the planner output."""

import json

from grace.agent.planner import parse_planned_step_with_reason, repair_json

STEP_002_RAW = (
    '{"thought": "I need to open the chat with \'Mamma\' to send the message about being late. '
    "I can see 'Mamma' in the chat list - it's the second entry with a green checkmark and "
    "'Photo' label. I'll click on this chat to open the conversation.\", "
    '"action": {"tool": "cua_click", "element_id": 4, "x": 100, "y": 230}, '
    '"expect": "The chat with Mamma opens, showing the conversation history and text input field at the bottom.", '
    '"is_completed": false, "user_update": "Opening chat with Mamma", "final_response": ""}}'
)


def test_exact_step_002_raw_parses():
    step, reason = parse_planned_step_with_reason(STEP_002_RAW)
    assert step is not None, reason
    assert step.action == "cua_click"
    assert step.params["x"] == 100 and step.params["element_id"] == 4


def test_missing_final_brace_parses():
    text = '{"action": "converse", "params": {"response": "hi"}, "is_completed": true'
    assert json.loads(repair_json(text))["is_completed"] is True
    step, reason = parse_planned_step_with_reason(text)
    assert step is not None, reason


def test_missing_nested_closers_in_order():
    assert json.loads(repair_json('{"a": [1, {"b": 2')) == {"a": [1, {"b": 2}]}


def test_trailing_comma_parses():
    assert json.loads(repair_json('{"a": [1, 2,], "b": {"c": 1,},}')) == {"a": [1, 2], "b": {"c": 1}}


def test_brace_inside_string_is_not_miscounted():
    text = '{"a": "x } y { z", "b": 1'
    assert json.loads(repair_json(text)) == {"a": "x } y { z", "b": 1}
    assert json.loads(repair_json('{"a": "she said \\"}\\" ok",}')) == {"a": 'she said "}" ok'}


def test_valid_json_is_unchanged():
    text = '{"a": {"b": [1, 2]}, "c": "d"}'
    assert repair_json(text) == text


def test_never_invents_fields():
    assert json.loads(repair_json('{"a": 1, "b": 2')) == {"a": 1, "b": 2}
    assert repair_json("no braces here") == "no braces here"
