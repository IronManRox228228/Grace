"""The intent prompt must carry the rules the router backstops.

The multi-step rule and the search-vs-read distinction were added after the
live Holo model answered "click the search box and type hello" with
`converse` (routing the turn to chat) and "read my report pdf to me" with
`search_files` (truncating it down the fast path). These tests only pin that
the rules are stated where the model reads them, in the single prompt file
that is the source of truth; the deterministic half is tested in
test_router.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.intent.prompt import get_system_prompt


def test_the_prompt_states_the_multi_step_rule():
    prompt = get_system_prompt()
    lowered = prompt.lower()
    assert "multi-step" in lowered
    # It must say what to do with one, and forbid the conversational escape.
    assert "converse" in lowered
    assert "first action" in lowered


def test_the_prompt_draws_the_read_vs_search_line():
    prompt = get_system_prompt()
    assert "search_files" in prompt
    assert "read_pdf" in prompt and "summarize_pdf" in prompt
