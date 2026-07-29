"""Dual-Path Capability Router for Grace.

Classifies incoming user voice requests into either:
1. Fast-Path (Deterministic Windows API execution)
2. Agentic Goal (Autonomous multi-step ReAct loop)
3. Direct Verbal Conversation
"""

import logging
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

    # Tools that execute in a single deterministic fast-path pass (pure hardware/OS locks)
    FAST_PATH_TOOLS = {
        "adjust_volume",
        "lock_computer",
    }

    # Comprehensive action verbs, UI elements, media terms, and workflow keywords
    AGENTIC_KEYWORDS = {
        # Action verbs & UI interactions
        "click", "press", "type", "input", "select", "choose", "tap", "hit",
        "scroll", "drag", "move", "hover", "focus", "play", "pause", "watch",
        "open", "launch", "close", "exit", "stop", "search", "find", "lookup",
        "navigate", "go to", "visit", "fill", "check", "read", "summarize",
        "create", "write", "notes", "copy", "paste", "extract", "organize",

        # UI components, screen & media elements
        "video", "link", "button", "menu", "icon", "image", "photo", "picture",
        "tab", "window", "page", "screen", "site", "website", "url", "browser",
        "app", "application", "file", "folder", "doc", "pdf", "text", "element",

        # Popular web platforms & desktop tools
        "youtube", "google", "reddit", "github", "twitter", "x.com", "amazon",
        "wikipedia", "netflix", "chatgpt", "edge", "chrome", "notepad", "calculator",
        "explorer", "terminal", "cmd", "word", "excel", "powerpoint", "vlc", "spotify"
    }

    @classmethod
    def classify(cls, prompt_text: str, parsed_intent: Optional[Intent] = None) -> TaskComplexity:
        """Determine task complexity path."""
        prompt_lower = prompt_text.lower().strip()

        # Check for multi-step agentic keywords
        if any(keyword in prompt_lower for keyword in cls.AGENTIC_KEYWORDS):
            logger.info(f"CapabilityRouter: Route -> AGENTIC_GOAL (Keyword match: '{prompt_text}')")
            return TaskComplexity.AGENTIC_GOAL

        if parsed_intent:
            if parsed_intent.tool in cls.FAST_PATH_TOOLS:
                logger.info(f"CapabilityRouter: Route -> FAST_PATH (Tool: '{parsed_intent.tool}')")
                return TaskComplexity.FAST_PATH
            elif parsed_intent.tool.startswith("cua_") or parsed_intent.tool in ("open_app", "close_app", "open_file", "read_pdf", "summarize_pdf"):
                # App workflows and CUA manipulation -> Agentic path
                logger.info(f"CapabilityRouter: Route -> AGENTIC_GOAL (Tool: '{parsed_intent.tool}')")
                return TaskComplexity.AGENTIC_GOAL
            elif parsed_intent.is_conversation:
                logger.info("CapabilityRouter: Route -> CONVERSATION")
                return TaskComplexity.CONVERSATION

        logger.info(f"CapabilityRouter: Route -> AGENTIC_GOAL (Default agentic for workflow prompt '{prompt_text}')")
        return TaskComplexity.AGENTIC_GOAL
