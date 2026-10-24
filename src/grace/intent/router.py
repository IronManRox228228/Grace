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
        "cua_press_key",
        "cua_scroll",
        "cua_activate",
        "undo",
        "describe_screen",
        "set_speech_rate",
        "remember",
    }

    # Tools that inherently need to look at the screen and iterate.
    AGENTIC_TOOLS = {
        "cua_click",
        "cua_type_text",
        "cua_drag",
        "cua_set_value",
        "cua_secondary_action",
        "read_pdf",
        "summarize_pdf",
    }

    # Words that join a second clause onto the first. A second clause is a
    # second action, whatever verb it happens to use.
    CLAUSE_SEPARATORS = {"and", "then", "also", "next", "after that", "plus"}

    # Splits a request into clauses for the multi-step backstop below. Covers
    # the separators above plus comma/semicolon chains, which speech uses just
    # as often ("switch to notepad, copy my notes, paste it into the email").
    CLAUSE_SPLIT_RE = re.compile(r"\s*(?:[,;]|\b(?:after\s+that|and|also|next|plus|then)\b)\s*")

    # Verbs that name a desktop action, used only to count action clauses in
    # the multi-step backstop. Deliberately NOT the old verb-whitelist that
    # used to promote single requests to the loop - that approach was removed
    # because no list of verbs is ever finished. Here nothing is promoted on a
    # verb alone: promotion requires two or more clauses, each carrying one,
    # so the list only has to catch verbs the user actually said again, not
    # every verb a user might say once.
    ACTION_VERBS = (
        "open", "close", "launch", "switch", "start", "click", "press", "type",
        "scroll", "drag", "search", "find", "read", "write", "delete", "remove",
        "copy", "paste", "play", "mute", "unmute", "undo", "lock", "unlock",
        "download", "upload", "install", "print", "save", "send", "email",
        "reply", "summarize", "minimize", "maximize", "check", "go",
        "take", "capture", "screenshot",
    )

    # Longest utterance still treated as a single deterministic command.
    #
    # A backstop for the case the separators miss: speech often drops the
    # conjunction ("open whatsapp search for the pdf"), and the intent model
    # answers such a sentence with the first tool it recognises, silently
    # discarding the rest. Length is a crude proxy for "there is more here than
    # one tool call", but it fails in the safe direction - the longest genuinely
    # atomic command in the recorded corpus is six words.
    MAX_FAST_PATH_WORDS = 8

    # Words that ask for a file to be found. `search_files` claims the
    # utterance asked for something to be found; when it never did, the tool
    # was the model's mis-pick - "read my report pdf to me" was answered
    # "search_files" (query "report.pdf") on the live model, dispatched once
    # down the fast path, and the ask was discarded.
    FILE_FIND_PHRASES = ("find", "search", "locate", "list", "look up", "scan")

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
    def _action_clauses(cls, text: str) -> int:
        """How many clauses of `text` name a desktop action."""
        clauses = [c.strip() for c in cls.CLAUSE_SPLIT_RE.split(text) if c.strip()]
        return sum(1 for c in clauses if cls._mentions(c, cls.ACTION_VERBS))

    @classmethod
    def _multi_action(cls, text: str) -> Optional[str]:
        """Why a request reads as two or more chained actions, or None.

        The deterministic backstop for the intent model's most harmful
        shortcut: answering a multi-step request with `converse` ("I'll click
        the search box and then type hello...") or with the first tool it
        recognises, which runs one action and strands the rest of the
        sentence. A clause counts only when it carries a verb that names a
        desktop action, so chit-chat with a conjunction in it ("thanks and
        goodbye") stays conversation.
        """
        if cls._action_clauses(text) >= 2:
            return "the request asks for two or more actions"
        return None

    @classmethod
    def _search_without_a_search_ask(cls, tool: str, text: str) -> Optional[str]:
        """Why a `search_files` dispatch has no right to the fast path.

        The tool's premise is that the utterance asked for a file to be found.
        When it never did, the pick was the model's truncation of a bigger ask,
        and the fast path would run the search and discard the request.
        """
        if tool != "search_files":
            return None
        if cls._mentions(text, cls.FILE_FIND_PHRASES):
            return None
        return "'search_files' was chosen but the request never asks for a file to be found"

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
                reason = (
                    cls._not_atomic(prompt_lower)
                    or cls._multi_action(prompt_lower)
                    or cls._search_without_a_search_ask(tool, prompt_lower)
                )
                if reason is None:
                    logger.info(f"CapabilityRouter: Route -> FAST_PATH (tool '{tool}')")
                    return TaskComplexity.FAST_PATH
                logger.info(f"CapabilityRouter: Route -> AGENTIC_GOAL ('{tool}' but {reason})")
                return TaskComplexity.AGENTIC_GOAL

            if parsed_intent.is_conversation:
                # The intent model folds multi-step requests into a spoken
                # reply ("I'll click the search box and then type hello..."),
                # which strands every action after the first. Two or more
                # action clauses outrank the conversational label.
                reason = cls._multi_action(prompt_lower)
                if reason is not None:
                    logger.info(f"CapabilityRouter: Route -> AGENTIC_GOAL "
                                f"(converse intent, but {reason})")
                    return TaskComplexity.AGENTIC_GOAL
                logger.info("CapabilityRouter: Route -> CONVERSATION (intent)")
                return TaskComplexity.CONVERSATION

        # 2. No usable intent. Greetings and small talk should never take the
        #    agentic path - that was a screenshot and a vision call to say hi.
        #    Two chained actions, however, need the loop whichever way the
        #    sentence reads, and before the parser produced anything usable.
        reason = cls._multi_action(prompt_lower)
        if reason is not None:
            logger.info(f"CapabilityRouter: Route -> AGENTIC_GOAL ({reason})")
            return TaskComplexity.AGENTIC_GOAL

        if cls._mentions(prompt_lower, cls.CONVERSATION_KEYWORDS):
            logger.info("CapabilityRouter: Route -> CONVERSATION (phrasing)")
            return TaskComplexity.CONVERSATION

        logger.info(f"CapabilityRouter: Route -> AGENTIC_GOAL (default for '{prompt_text}')")
        return TaskComplexity.AGENTIC_GOAL
