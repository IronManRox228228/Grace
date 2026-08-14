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
    """Routes voice intents to either the fast deterministic executor or the agentic loop.

    The agentic loop is the default and the fast path is the exception, because
    the two ways of being wrong do not cost the same. Sending a one-step request
    to the loop costs a screenshot and a planner call - a few seconds, and the
    right outcome, since the loop can call the same deterministic tools as its
    first step. Sending a multi-step request down the fast path runs the first
    tool, discards the rest of the sentence, and reports success: the user is
    told "I've opened WhatsApp" when they asked for a file inside it.

    This used to be the other way round, with a whitelist of verbs ("click",
    "play", "scroll") promoting a request to the loop. Any verb missing from
    that list - "search" was - silently truncated the request, and no list of
    verbs is ever finished.
    """

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

    # Words that join a second clause onto the first. A second clause is a
    # second action, whatever verb it happens to use.
    CLAUSE_SEPARATORS = {"and", "then", "also", "next", "after that", "plus"}

    # Longest utterance still treated as a single deterministic command.
    #
    # A backstop for the case the separators miss: speech often drops the
    # conjunction ("open whatsapp search for the pdf"), and the intent model
    # answers such a sentence with the first tool it recognises, silently
    # discarding the rest. Length is a crude proxy for "there is more here than
    # one tool call", but it fails in the safe direction - the longest genuinely
    # atomic command in the recorded corpus is six words.
    MAX_FAST_PATH_WORDS = 8

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
    def _not_atomic(cls, text: str) -> Optional[str]:
        """Why this utterance is more than one command, or None if it isn't.

        Returns a reason string so the routing decision is legible in the log;
        a bare bool makes "why did that go agentic?" unanswerable after the
        fact.
        """
        if cls._mentions(text, cls.CLAUSE_SEPARATORS):
            return "the request has a second clause"

        words = len(text.split())
        if words > cls.MAX_FAST_PATH_WORDS:
            return f"the request is {words} words, longer than one command"

        return None

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
                # The fast path is opt-in, and the utterance has to earn it by
                # being one short command. Anything else goes to the loop, which
                # can call this very tool as its first step if that is all the
                # request needed.
                reason = cls._not_atomic(prompt_lower)
                if reason is None:
                    logger.info(f"CapabilityRouter: Route -> FAST_PATH (tool '{tool}')")
                    return TaskComplexity.FAST_PATH
                logger.info(f"CapabilityRouter: Route -> AGENTIC_GOAL ('{tool}' but {reason})")
                return TaskComplexity.AGENTIC_GOAL

            if parsed_intent.is_conversation:
                logger.info("CapabilityRouter: Route -> CONVERSATION (intent)")
                return TaskComplexity.CONVERSATION

        # 2. No usable intent. Greetings and small talk should never take the
        #    agentic path - that was a screenshot and a vision call to say hi.
        if cls._mentions(prompt_lower, cls.CONVERSATION_KEYWORDS):
            logger.info("CapabilityRouter: Route -> CONVERSATION (phrasing)")
            return TaskComplexity.CONVERSATION

        logger.info(f"CapabilityRouter: Route -> AGENTIC_GOAL (default for '{prompt_text}')")
        return TaskComplexity.AGENTIC_GOAL
