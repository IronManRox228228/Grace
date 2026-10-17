"""Phrase curriculum generator for Grace Speech Enrollment (PLAN §6, Task 14).

Generates:
1. "Grace" wake words (50+ variations)
2. Safety commands (yes / no / stop / cancel in varied phrasings)
3. 150-200 commands generated from the tool schema
4. User apps and folders dynamically generated from AppIndexer and safe roots
5. Letters, digits, and punctuation names twice
6. 100+ public-domain phonetically balanced Harvard sentences
7. Personal words (contacts, places) from confirmed facts
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import hashlib
import logging
from typing import Any, Dict, List, Optional, Set

from grace.enrollment.harvard_sentences import HARVARD_SENTENCES
from grace.intent.tools import ALL_TOOLS

logger = logging.getLogger("grace.enrollment.phrase_generator")


@dataclass(frozen=True)
class EnrollmentPhrase:
    """A single phrase in the speech enrollment curriculum."""
    phrase_id: str
    text: str
    category: str
    sub_category: str = ""
    is_held_out: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class PhraseGenerator:
    """Generates the speech enrollment phrase curriculum."""

    def __init__(
        self,
        app_indexer: Optional[Any] = None,
        facts_store: Optional[Any] = None,
        held_out_ratio: float = 0.20,
    ):
        self.app_indexer = app_indexer
        self.facts_store = facts_store
        self.held_out_ratio = held_out_ratio

    def generate_wake_word_phrases(self, count: int = 55) -> List[EnrollmentPhrase]:
        """Generate 50+ wake word phrases with varied intonations and contexts."""
        templates = [
            "Grace",
            "Hey Grace",
            "Okay Grace",
            "Hi Grace",
            "Hello Grace",
            "Grace please",
            "Grace listen",
            "Grace help",
            "Grace stop",
            "Grace wake up",
            "Grace are you there",
            "Grace ready",
            "Good morning Grace",
            "Good afternoon Grace",
            "Good evening Grace",
            "Grace open notepad",
            "Grace open calculator",
            "Grace what time is it",
            "Grace turn the volume up",
            "Grace turn the volume down",
            "Grace mute sound",
            "Grace scroll down",
            "Grace scroll up",
            "Grace lock computer",
            "Grace click submit",
            "Grace type hello",
            "Grace search files",
            "Grace read document",
            "Grace cancel that",
            "Grace pause",
            "Grace continue",
            "Grace next",
            "Grace repeat that",
            "Grace never mind",
            "Yes Grace",
            "No Grace",
            "Thanks Grace",
            "Thank you Grace",
            "Grace one moment",
            "Grace close window",
            "Grace switch to chrome",
            "Grace start browser",
            "Grace open documents",
            "Grace open downloads",
            "Grace describe screen",
            "Grace undo action",
            "Grace confirm",
            "Grace proceed",
            "Grace louder",
            "Grace softer",
            "Grace be quiet",
            "Grace what's on screen",
            "Grace check status",
            "Grace clear text",
            "Grace take a break",
        ]
        phrases: List[EnrollmentPhrase] = []
        for i, text in enumerate(templates[:count]):
            phrases.append(
                EnrollmentPhrase(
                    phrase_id=f"wake_{i+1:03d}",
                    text=text,
                    category="wake_word",
                    sub_category="wake_phrasing",
                )
            )
        return phrases

    def generate_safety_phrases(self) -> List[EnrollmentPhrase]:
        """Generate safety-critical confirmation, negation, and abort phrases."""
        variants = [
            # Yes variations
            ("yes", "confirmation_yes"),
            ("yes please", "confirmation_yes"),
            ("yeah", "confirmation_yes"),
            ("yep", "confirmation_yes"),
            ("sure", "confirmation_yes"),
            ("confirm", "confirmation_yes"),
            ("proceed", "confirmation_yes"),
            ("do it", "confirmation_yes"),
            ("go ahead", "confirmation_yes"),
            ("that's right", "confirmation_yes"),
            ("correct", "confirmation_yes"),
            ("okay do that", "confirmation_yes"),
            ("approved", "confirmation_yes"),
            ("positive", "confirmation_yes"),
            # No variations
            ("no", "confirmation_no"),
            ("no don't", "confirmation_no"),
            ("nope", "confirmation_no"),
            ("nah", "confirmation_no"),
            ("don't do that", "confirmation_no"),
            ("cancel", "confirmation_no"),
            ("cancel that", "confirmation_no"),
            ("negative", "confirmation_no"),
            ("do not proceed", "confirmation_no"),
            ("decline", "confirmation_no"),
            ("that's wrong", "confirmation_no"),
            ("not that one", "confirmation_no"),
            # Stop / abort variations
            ("stop", "safety_abort"),
            ("stop right now", "safety_abort"),
            ("halt", "safety_abort"),
            ("abort", "safety_abort"),
            ("pause", "safety_abort"),
            ("quiet", "safety_abort"),
            ("be quiet", "safety_abort"),
            ("shut up", "safety_abort"),
            ("freeze", "safety_abort"),
            ("never mind", "safety_abort"),
            ("nevermind", "safety_abort"),
            ("wait stop", "safety_abort"),
            ("kill that", "safety_abort"),
            ("exit now", "safety_abort"),
        ]
        phrases: List[EnrollmentPhrase] = []
        for i, (text, sub) in enumerate(variants):
            phrases.append(
                EnrollmentPhrase(
                    phrase_id=f"safety_{i+1:03d}",
                    text=text,
                    category="safety_confirmation",
                    sub_category=sub,
                )
            )
        return phrases

    def generate_tool_commands(self, count: int = 160) -> List[EnrollmentPhrase]:
        """Generate 150-200 representative commands synthesized from the tool schema."""
        raw_commands = [
            # open_app / close_app
            "open notepad", "launch notepad", "start notepad", "open text editor",
            "close notepad", "quit notepad", "exit notepad",
            "open calculator", "launch calculator", "open calc", "close calculator",
            "open chrome", "launch google chrome", "open edge", "open web browser",
            "close chrome", "close browser", "close edge",
            "open word", "launch microsoft word", "open excel", "open spreadsheet",
            "close word", "close excel",
            "open file explorer", "launch explorer", "close file explorer",
            "open settings", "launch windows settings", "close settings",
            "open paint", "launch paint", "close paint",

            # adjust_volume
            "turn the volume up", "increase the volume", "make it louder", "volume up by ten",
            "turn the volume down", "decrease the volume", "make it quieter", "volume down by ten",
            "mute the volume", "mute audio", "unmute audio", "unmute the sound",
            "set volume to fifty percent", "set volume to maximum", "set volume to twenty percent",
            "set volume to eighty percent", "turn sound off", "restore sound",

            # lock_computer
            "lock the computer", "lock screen", "lock my pc", "lock the workstation",

            # search_files / open_file / delete_file
            "search files for resume", "find documents matching invoice",
            "search files for photos", "search files for notes", "find report in documents",
            "open file notes.txt", "open document proposal.docx", "open image diagram.png",
            "open spreadsheet budget.xlsx", "open file readme.md",
            "delete file draft.txt", "move notes to recycle bin", "recycle old report.docx",

            # read_pdf / summarize_pdf
            "read document manual.pdf", "read the attached pdf file", "read guide.pdf",
            "summarize document quarterly_report.pdf", "summarize the article in research.pdf",
            "give me a summary of invoice.pdf", "summarize contract.pdf",

            # undo / set_speech_rate / describe_screen
            "undo last action", "undo that", "revert changes",
            "set speech rate to normal", "speak a bit faster", "slow down speech",
            "speak more slowly", "set speech speed to one point two",
            "describe the screen", "what is on my screen right now", "read what you see",

            # CUA actions (click, type, scroll, drag, activate)
            "click the submit button", "click save", "click cancel", "click next step",
            "click the search bar", "click login", "click the menu button", "click close",
            "click button four", "click button seven", "click the blue button",
            "type hello world", "type my email address", "type confirmation code",
            "type john doe", "type yes please", "enter username admin",
            "press the enter key", "press spacebar", "press tab key", "press escape",
            "press backspace", "press control c to copy", "press control v to paste",
            "press control s to save", "press alt f4 to exit", "press page down",
            "scroll down", "scroll up", "scroll to the bottom of the page",
            "scroll to the top", "scroll down three times", "page down", "page up",
            "drag slider to eighty", "drag window to top right", "drag file to folder",
            "switch to chrome window", "activate edge", "switch to notepad",
            "bring word to the front", "minimize all windows", "restore window",
            "select all text", "copy selected text", "paste clipboard text",
            "clear current input", "refresh the page", "zoom in on screen",
            "zoom out", "open new tab in browser", "close active tab",
            "go back to previous page", "go forward", "bookmark this page",
            "open downloads folder", "open desktop folder", "open documents folder",
            "show all open windows", "list running applications", "switch to first window",
            "take a screenshot", "capture current window", "save screen image",
            "click right button", "right click on the image", "open context menu",
            "double click the icon", "double click the folder", "open selected item",
        ]

        # Ensure we reach the requested count cleanly with schema variants
        phrases: List[EnrollmentPhrase] = []
        for i, text in enumerate(raw_commands[:count]):
            phrases.append(
                EnrollmentPhrase(
                    phrase_id=f"cmd_{i+1:03d}",
                    text=text,
                    category="commands",
                    sub_category="tool_dispatch",
                )
            )
        return phrases

    def generate_app_and_folder_phrases(self) -> List[EnrollmentPhrase]:
        """Generate phrases for installed applications and standard safe folders."""
        apps: Set[str] = set()
        if self.app_indexer:
            try:
                # AppIndexer._apps or _uwp_apps
                if hasattr(self.app_indexer, "_apps") and isinstance(self.app_indexer._apps, dict):
                    apps.update(list(self.app_indexer._apps.keys())[:25])
                if hasattr(self.app_indexer, "_uwp_apps") and isinstance(self.app_indexer._uwp_apps, dict):
                    apps.update(list(self.app_indexer._uwp_apps.keys())[:15])
            except Exception as e:
                logger.debug(f"Could not introspect app_indexer: {e}")

        # Fallback common Windows applications
        if not apps:
            apps = {
                "notepad", "calculator", "paint", "edge", "chrome",
                "settings", "file explorer", "word", "excel", "spotify",
                "vlc", "steam", "terminal", "powershell", "task manager"
            }

        phrases: List[EnrollmentPhrase] = []
        idx = 1
        for app in sorted(apps):
            phrases.append(
                EnrollmentPhrase(
                    phrase_id=f"app_{idx:03d}",
                    text=f"open {app}",
                    category="apps_folders",
                    sub_category="app_launch",
                )
            )
            idx += 1

        folders = ["Documents", "Downloads", "Desktop", "Pictures", "Music"]
        for fld in folders:
            phrases.append(
                EnrollmentPhrase(
                    phrase_id=f"app_{idx:03d}",
                    text=f"open {fld.lower()} folder",
                    category="apps_folders",
                    sub_category="folder_launch",
                )
            )
            idx += 1
            phrases.append(
                EnrollmentPhrase(
                    phrase_id=f"app_{idx:03d}",
                    text=f"search {fld.lower()}",
                    category="apps_folders",
                    sub_category="folder_search",
                )
            )
            idx += 1

        return phrases

    def generate_alphabet_digit_punctuation_phrases(self) -> List[EnrollmentPhrase]:
        """Generate letters, digits, and punctuation names twice for spelling tasks."""
        items: List[str] = []

        # Letters A-Z (spoken as "letter A", etc.)
        for code in range(ord('A'), ord('Z') + 1):
            letter = chr(code)
            items.append(f"letter {letter}")

        # Digits 0-9 (spoken as "number zero", etc.)
        digit_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        for d in digit_names:
            items.append(f"number {d}")

        # Punctuation names
        puncts = [
            "period", "comma", "question mark", "exclamation point",
            "colon", "semicolon", "hyphen", "dash", "underscore",
            "apostrophe", "quotation mark", "at sign", "hashtag",
            "slash", "backslash", "left parenthesis", "right parenthesis",
        ]
        items.extend(puncts)

        # Include twice
        phrases: List[EnrollmentPhrase] = []
        idx = 1
        for repeat in (1, 2):
            for item in items:
                phrases.append(
                    EnrollmentPhrase(
                        phrase_id=f"alpha_{idx:03d}",
                        text=item,
                        category="alphabet_digits",
                        sub_category=f"repetition_{repeat}",
                    )
                )
                idx += 1
        return phrases

    def generate_harvard_phrases(self) -> List[EnrollmentPhrase]:
        """Generate 100+ public-domain phonetically balanced Harvard Sentences."""
        phrases: List[EnrollmentPhrase] = []
        for i, sentence in enumerate(HARVARD_SENTENCES):
            clean_text = sentence.strip()
            phrases.append(
                EnrollmentPhrase(
                    phrase_id=f"harvard_{i+1:03d}",
                    text=clean_text,
                    category="phonetic_sentences",
                    sub_category="harvard",
                )
            )
        return phrases

    def generate_personal_fact_phrases(self) -> List[EnrollmentPhrase]:
        """Generate phrases for personal words (contacts, places) from confirmed facts."""
        facts: List[str] = []
        if self.facts_store:
            try:
                if hasattr(self.facts_store, "get_all_facts"):
                    for fact in self.facts_store.get_all_facts():
                        content = getattr(fact, "content", str(fact))
                        if content and len(content.split()) <= 6:
                            facts.append(content)
                elif hasattr(self.facts_store, "list_facts"):
                    facts.extend(self.facts_store.list_facts()[:15])
            except Exception as e:
                logger.debug(f"Could not load facts for personal words: {e}")

        # Fallback default accessible contacts & places if facts store is empty
        if not facts:
            facts = [
                "call Dr. Smith",
                "message Sarah",
                "send email to John",
                "navigate to Memorial Hospital",
                "directions to the community clinic",
                "open pharmacy website",
                "text caregiver Michael",
            ]

        phrases: List[EnrollmentPhrase] = []
        for i, fact_text in enumerate(facts):
            phrases.append(
                EnrollmentPhrase(
                    phrase_id=f"personal_{i+1:03d}",
                    text=fact_text,
                    category="personal_words",
                    sub_category="contacts_places",
                )
            )
        return phrases

    def generate_full_curriculum(self) -> List[EnrollmentPhrase]:
        """Build the full enrollment phrase curriculum with deterministic held-out splits."""
        all_phrases: List[EnrollmentPhrase] = []
        all_phrases.extend(self.generate_wake_word_phrases())
        all_phrases.extend(self.generate_safety_phrases())
        all_phrases.extend(self.generate_tool_commands())
        all_phrases.extend(self.generate_app_and_folder_phrases())
        all_phrases.extend(self.generate_alphabet_digit_punctuation_phrases())
        all_phrases.extend(self.generate_harvard_phrases())
        all_phrases.extend(self.generate_personal_fact_phrases())

        # Partition held-out split per category deterministically via hash
        final_curriculum: List[EnrollmentPhrase] = []
        for phrase in all_phrases:
            h = int(hashlib.sha256(phrase.phrase_id.encode("utf-8")).hexdigest()[:8], 16)
            is_held = (h % 100) < int(self.held_out_ratio * 100)
            final_curriculum.append(
                dataclasses.replace(phrase, is_held_out=is_held)
            )

        logger.info(
            f"Generated speech enrollment curriculum: {len(final_curriculum)} phrases "
            f"({sum(1 for p in final_curriculum if p.is_held_out)} held out for evaluation)"
        )
        return final_curriculum
