"""Tests for Benchmark 3 tool calling evaluators and runner."""

import pytest

from bench.dataset_builder import ToolCallItem
from bench.run_tool_calling_benchmark import (
    FunctionGemmaToolCaller,
    HoloToolCaller,
    LFM350MToolCaller,
    XLAMToolCaller,
    evaluate_tool_caller,
)


def test_tool_callers_system_tools():
    callers = [
        FunctionGemmaToolCaller(),
        LFM350MToolCaller(),
        XLAMToolCaller(),
        HoloToolCaller(),
    ]

    samples = [
        ("open chrome", "open_app", {"app_name": "chrome"}),
        ("close notepad", "close_app", {"app_name": "notepad"}),
        ("turn the volume up", "adjust_volume", {"direction": "up", "step": 10}),
        ("lock the screen", "lock_computer", {}),
        ("find invoice pdf", "search_files", {"query": "invoice pdf"}),
        ("undo that", "undo", {}),
    ]

    for caller in callers:
        for utt, expected_tool, expected_args in samples:
            res = caller.predict_tool_call(utt)
            assert res["tool"] == expected_tool
            assert res["valid_json"] is True
            for k, v in expected_args.items():
                assert res["args"].get(k) == v


def test_tool_callers_cua_tools():
    caller = FunctionGemmaToolCaller()

    samples = [
        ("click the submit button", "cua_click", {"element_id": 3}),
        ("type hello world", "cua_type_text", {"text": "hello world"}),
        ("press enter", "cua_press_key", {"key": "enter"}),
        ("scroll down", "cua_scroll", {"direction": "down"}),
    ]

    for utt, expected_tool, expected_args in samples:
        res = caller.predict_tool_call(utt)
        assert res["tool"] == expected_tool
        for k, v in expected_args.items():
            assert res["args"].get(k) == v


def test_evaluate_tool_caller_pipeline():
    caller = FunctionGemmaToolCaller()
    items = [
        ToolCallItem(sample_id="t1", utterance="open edge", expected_tool="open_app", expected_args={"app_name": "edge"}, is_multi_turn=False),
        ToolCallItem(sample_id="t2", utterance="lock screen", expected_tool="lock_computer", expected_args={}, is_multi_turn=True),
    ]

    results = evaluate_tool_caller(caller, items)
    assert results["overall"]["exact_tool_accuracy"] == 1.0
    assert results["overall"]["args_match_accuracy"] == 1.0
    assert results["overall"]["json_validity_rate"] == 1.0
    assert results["single_turn"]["total_samples"] == 1
    assert results["multi_turn"]["total_samples"] == 1
    assert "p50" in results["latency_ms"]
