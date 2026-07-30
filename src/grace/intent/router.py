"""Dual-Path Capability Router for Grace.

Classifies incoming user voice requests into either:
1. Fast-Path (Deterministic Windows API execution)
2. Agentic Goal (Autonomous multi-step ReAct loop)
3. Direct Verbal Conversation

Two things used to send almost everything down the slow path. Keywords were
matched as substrings, so "explain" contained "plain"... and more damagingly
"open" matched "opening", "happen" and "reopened", while "x" matched any word
containing an x. And the keyword sweep ran *before* the parsed intent was
consulted, so a request the intent model had already resolved to a single
deterministic tool still took the full screenshot-and-vision loop.
"""

import logging
import re
from enum import Enum
from typing import Optional
from grace.intent.parser import Intent

logger = logging.getLogger("grace.intent.router")


class TaskComplexity(Enum):
    FAST_PATH = "fast_path"
    AGENTIC_GOAL = "agentic_goal"
    CONVERSATION = "conversation"


class CapabilityRouter:
    """Routes voice intents to either the fast deterministic executor or the agentic loop."""

    # Tools that complete in one deterministic pass. No screen state is read,
    # so there is nothing for the agentic loop to add.
    FAST_PATH_TOOLS = {
        "adjust_volume",
        "lock_computer",
        "open_calculator",
        "open_app",
        "close_app",
        "open_file",
        "search_files",
        "delete_file",
        "cua_launch",
        "cua_list_windows",
        "cua_list_apps",
    }

    # Tools that inherently need to look at the screen and iterate.
    AGENTIC_TOOLS = {
        "cua_click",
        "cua_type_text",
        "cua_press_key",
        "cua_scroll",
        "cua_drag",
        "cua_set_value",
        "cua_secondary_action",
        "cua_activate",
        "read_pdf",
        "summarize_pdf",
    }

    # Multi-step phrasings that genuinely need the agentic loop even when the
    # intent model produced a single tool call. Deliberately much smaller than
    # the old ~80-keyword list, which caught the word "app" in "happy".
    AGENTIC_KEYWORDS = {
        "click", "press", "type", "select", "choose", "tap",
        "scroll", "drag", "hover", "play", "pause", "watch",
        "navigate", "fill", "summarize", "extract", "organize",
        "then", "after that", "and then",
    }

    CONVERSATION_KEYWORDS = {
        "hello", "hi", "hey", "thanks", "thank you", "goodbye", "bye",
        "who are you", "what can you do", "how are you",
    }

    @classmethod
    def _mentions(cls, text: str, phrases) -> bool:
        """Whole-word / whole-phrase matching, not substring."""
        for phrase in phrases:
            pattern = r"\b" + r"\s+".join(re.escape(w) for w in phrase.split()) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    @classmethod
    def classify(cls, prompt_text: str, parsed_intent: Optional[Intent] = None) -> TaskComplexity:
        """Determine task complexity path."""
        prompt_lower = (prompt_text or "").lower().strip()

        # 1. Trust the parsed intent first. It is the most specific signal we
        #    have, and re-deciding from raw keywords discards that work.
        if parsed_intent is not None:
            tool = parsed_intent.tool or ""

            if tool in cls.AGENTIC_TOOLS:
                logger.info(f"CapabilityRouter: Route -> AGENTIC_GOAL (tool '{tool}')")
                return TaskComplexity.AGENTIC_GOAL

            if tool in cls.FAST_PATH_TOOLS:
                # A single-tool intent still goes agentic if the utterance
                # clearly chains more work onto it ("open YouTube and play...").
                if cls._mentions(prompt_lower, cls.AGENTIC_KEYWORDS):
                    logger.info(f"CapabilityRouter: Route -> AGENTIC_GOAL (multi-step phrasing around '{tool}')")
                    return TaskComplexity.AGENTIC_GOAL
                logger.info(f"CapabilityRouter: Route -> FAST_PATH (tool '{tool}')")
                return TaskComplexity.FAST_PATH

            if parsed_intent.is_conversation:
                logger.info("CapabilityRouter: Route -> CONVERSATION (intent)")
                return TaskComplexity.CONVERSATION

        # 2. No usable intent. Greetings and small talk should never take the
        #    agentic path - that was a screenshot and a vision call to say hi.
        if cls._mentions(prompt_lower, cls.CONVERSATION_KEYWORDS):
            logger.info("CapabilityRouter: Route -> CONVERSATION (phrasing)")
            return TaskComplexity.CONVERSATION

        if cls._mentions(prompt_lower, cls.AGENTIC_KEYWORDS):
            logger.info(f"CapabilityRouter: Route -> AGENTIC_GOAL (keyword match: '{prompt_text}')")
            return TaskComplexity.AGENTIC_GOAL

        logger.info(f"CapabilityRouter: Route -> AGENTIC_GOAL (default for '{prompt_text}')")
        return TaskComplexity.AGENTIC_GOAL
