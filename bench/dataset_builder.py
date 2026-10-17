"""Dataset extraction and builders for Grace decision and tool calling benchmarks (PLAN.md §3, §8, §9, Task 11).

Extracts:
1. Element selection items: (instruction, element summaries, correct_id) from UIA snapshots and corpus tapes.
2. Yes/No confirmation items: clear, mumbled, hedged, greeting-prefixed, repeat, stop, and abstain utterances.
3. Tool calling items: corpus utterances and synthetic tool schema calls to expected tool + arguments.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "datasets"


@dataclass
class ElementSummary:
    """Compact summary of a UI element — never the full tree."""
    id: int
    name: str
    control_type: str
    parent_path: str
    window_title: str
    neighbouring_labels: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Format element summary into a compact string representation."""
        near_str = f" | near: {', '.join(self.neighbouring_labels)}" if self.neighbouring_labels else ""
        parent_str = f" in {self.parent_path}" if self.parent_path else ""
        return f"[#{self.id}] {self.name} ({self.control_type}){parent_str} | window: {self.window_title}{near_str}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ElementSummary:
        return cls(
            id=int(data.get("id", -1)),
            name=str(data.get("name", "")),
            control_type=str(data.get("control_type", "element")),
            parent_path=str(data.get("parent_path", "")),
            window_title=str(data.get("window_title", "")),
            neighbouring_labels=[str(x) for x in data.get("neighbouring_labels", [])],
        )


@dataclass
class ElementSelectionItem:
    """Benchmark item for element selection."""
    sample_id: str
    instruction: str
    elements: List[ElementSummary]
    correct_id: int  # -1 for 'none of these / ask the user'
    app_name: str
    is_held_out: bool = False
    source_tape: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "instruction": self.instruction,
            "elements": [e.to_dict() for e in self.elements],
            "correct_id": self.correct_id,
            "app_name": self.app_name,
            "is_held_out": self.is_held_out,
            "source_tape": self.source_tape,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ElementSelectionItem:
        return cls(
            sample_id=str(data.get("sample_id", "")),
            instruction=str(data.get("instruction", "")),
            elements=[ElementSummary.from_dict(e) for e in data.get("elements", [])],
            correct_id=int(data.get("correct_id", -1)),
            app_name=str(data.get("app_name", "unknown")),
            is_held_out=bool(data.get("is_held_out", False)),
            source_tape=data.get("source_tape"),
        )


@dataclass
class YesNoItem:
    """Benchmark item for confirmation and yes/no decision."""
    sample_id: str
    utterance: str
    category: str  # clear_yes, clear_no, mumbled, hedged, greeting_yes, greeting_no, repeat, stop, abstain
    expected_label: str  # YES, NO, REPEAT, STOP, ABSTAIN

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> YesNoItem:
        return cls(
            sample_id=str(data.get("sample_id", "")),
            utterance=str(data.get("utterance", "")),
            category=str(data.get("category", "")),
            expected_label=str(data.get("expected_label", "ABSTAIN")),
        )


@dataclass
class ToolCallItem:
    """Benchmark item for tool calling."""
    sample_id: str
    utterance: str
    expected_tool: str
    expected_args: Dict[str, Any]
    is_multi_turn: bool = False
    source: str = "corpus"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolCallItem:
        return cls(
            sample_id=str(data.get("sample_id", "")),
            utterance=str(data.get("utterance", "")),
            expected_tool=str(data.get("expected_tool", "")),
            expected_args=dict(data.get("expected_args", {})),
            is_multi_turn=bool(data.get("is_multi_turn", False)),
            source=str(data.get("source", "corpus")),
        )


def build_element_summary(el: Dict[str, Any], all_elements: List[Dict[str, Any]], window_title: str) -> ElementSummary:
    """Build a compact ElementSummary from a raw element dict and context."""
    el_id = int(el.get("id", 0))
    name = str(el.get("name") or el.get("placeholder") or "").strip()
    role = str(el.get("role") or el.get("control_type") or "element").strip().lower()
    container = str(el.get("container") or el.get("frame") or "").strip()

    # Find neighbouring labels (closest elements by list index)
    neighbours: List[str] = []
    idx = -1
    for i, e in enumerate(all_elements):
        if e.get("id") == el_id:
            idx = i
            break

    if idx >= 0:
        for offset in [-1, 1, -2, 2]:
            adj_idx = idx + offset
            if 0 <= adj_idx < len(all_elements) and adj_idx != idx:
                adj_name = str(all_elements[adj_idx].get("name") or "").strip()
                if adj_name and adj_name != name and adj_name not in neighbours:
                    neighbours.append(adj_name)
                    if len(neighbours) >= 2:
                        break

    return ElementSummary(
        id=el_id,
        name=name,
        control_type=role,
        parent_path=container,
        window_title=window_title,
        neighbouring_labels=neighbours,
    )


def extract_app_name(window_title: str, default: str = "unknown") -> str:
    """Extract application name from window title (e.g. 'Grace test page - Edge' -> 'Edge')."""
    if not window_title:
        return default
    if "-" in window_title:
        parts = window_title.split("-")
        return parts[-1].strip()
    if "|" in window_title:
        parts = window_title.split("|")
        return parts[-1].strip()
    return window_title.strip()


def extract_element_selection_dataset(
    corpus_dir: Union[str, Path],
    held_out_apps: Optional[List[str]] = None,
) -> List[ElementSelectionItem]:
    """Extract element selection benchmark items from corpus tape snapshots and dispatch logs."""
    held_out = set(a.lower() for a in (held_out_apps or ["notepad", "settings"]))
    c_path = Path(corpus_dir)
    items: List[ElementSelectionItem] = []

    if not c_path.is_dir():
        return items

    for tape_dir in sorted(c_path.iterdir()):
        if not tape_dir.is_dir():
            continue

        snapshots_file = tape_dir / "snapshots.jsonl"
        dispatch_file = tape_dir / "dispatch.jsonl"
        stt_file = tape_dir / "stt.jsonl"

        if not snapshots_file.is_file() or not dispatch_file.is_file():
            continue

        # Load STT utterances
        instructions: List[str] = []
        if stt_file.is_file():
            with open(stt_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            stt_obj = json.loads(line)
                            txt = stt_obj.get("text", "").strip()
                            if txt:
                                instructions.append(txt)
                        except Exception:
                            pass

        primary_instruction = instructions[0] if instructions else tape_dir.name.replace("_", " ")

        # Load snapshots
        snapshots: List[Dict[str, Any]] = []
        with open(snapshots_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        snapshots.append(json.loads(line))
                    except Exception:
                        pass

        # Load dispatch actions
        dispatch_actions: List[Dict[str, Any]] = []
        with open(dispatch_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        dispatch_actions.append(json.loads(line))
                    except Exception:
                        pass

        # Match cua_click or cua_set_value with snapshot
        step_idx = 0
        for action in dispatch_actions:
            tool = action.get("tool", "")
            params = action.get("params", {})
            target_el_id = params.get("element_id")

            if tool in ("cua_click", "cua_set_value", "cua_type_text") and target_el_id is not None:
                # Find corresponding snapshot
                snap_obj = snapshots[step_idx] if step_idx < len(snapshots) else (snapshots[-1] if snapshots else None)
                if not snap_obj:
                    continue

                snap_data = snap_obj.get("snapshot", snap_obj)
                win = snap_data.get("window", {})
                win_title = win.get("title", "")
                raw_elements = snap_data.get("elements", [])

                if not raw_elements:
                    continue

                app_name = extract_app_name(win_title, default=tape_dir.name)
                is_held_out = app_name.lower() in held_out

                element_summaries: List[ElementSummary] = []
                for el in raw_elements:
                    element_summaries.append(build_element_summary(el, raw_elements, win_title))

                # Always add abstain candidate
                element_summaries.append(
                    ElementSummary(
                        id=-1,
                        name="none of these / ask the user",
                        control_type="action",
                        parent_path="",
                        window_title=win_title,
                        neighbouring_labels=[],
                    )
                )

                instruction_text = primary_instruction
                if tool == "cua_set_value" and "value" in params:
                    instruction_text = f"set value to {params['value']}"
                elif len(instructions) > step_idx:
                    instruction_text = instructions[step_idx]

                sample_id = f"{tape_dir.name}_step_{step_idx}"
                items.append(
                    ElementSelectionItem(
                        sample_id=sample_id,
                        instruction=instruction_text,
                        elements=element_summaries,
                        correct_id=int(target_el_id),
                        app_name=app_name,
                        is_held_out=is_held_out,
                        source_tape=tape_dir.name,
                    )
                )

                # Generate a paired negative item (target element not in candidate list -> correct_id = -1)
                if len(element_summaries) > 2:
                    neg_elements = [e for e in element_summaries if e.id != int(target_el_id)]
                    neg_sample_id = f"{tape_dir.name}_neg_{step_idx}"
                    items.append(
                        ElementSelectionItem(
                            sample_id=neg_sample_id,
                            instruction=instruction_text,
                            elements=neg_elements,
                            correct_id=-1,
                            app_name=app_name,
                            is_held_out=is_held_out,
                            source_tape=tape_dir.name,
                        )
                    )

                step_idx += 1

    return items


def extract_yes_no_dataset(corpus_dir: Optional[Union[str, Path]] = None) -> List[YesNoItem]:
    """Generate and extract diverse confirmation and yes/no benchmark items."""
    items: List[YesNoItem] = []

    # 1. Clear YES items
    clear_yes = [
        "yes", "yep", "yeah", "sure", "confirm", "go ahead", "do it",
        "please", "yes please", "affirmative", "correct", "proceed",
        "that's right", "absolutely", "ok", "okay",
    ]
    for i, u in enumerate(clear_yes):
        items.append(YesNoItem(sample_id=f"clear_yes_{i:02d}", utterance=u, category="clear_yes", expected_label="YES"))

    # 2. Clear NO items
    clear_no = [
        "no", "nope", "cancel", "don't", "abort", "never mind",
        "negative", "no don't do that", "reject", "deny", "nah",
        "cancel that", "please don't",
    ]
    for i, u in enumerate(clear_no):
        items.append(YesNoItem(sample_id=f"clear_no_{i:02d}", utterance=u, category="clear_no", expected_label="NO"))

    # 3. Mumbled & Hedged answers
    hedged_yes = [
        "uh yeah I guess", "um maybe yeah", "well yes", "i think so",
        "i suppose so", "probably", "sure why not", "mhm yeah",
    ]
    for i, u in enumerate(hedged_yes):
        items.append(YesNoItem(sample_id=f"hedged_yes_{i:02d}", utterance=u, category="hedged", expected_label="YES"))

    hedged_no = [
        "maybe later", "not right now", "uh no", "um I guess not",
        "not really", "i don't think so", "nah let's wait",
    ]
    for i, u in enumerate(hedged_no):
        items.append(YesNoItem(sample_id=f"hedged_no_{i:02d}", utterance=u, category="hedged", expected_label="NO"))

    # 4. Greeting prefixed answers (must classify the inner answer, ignoring greeting)
    greeting_yes = [
        "hey grace yes", "hi grace please confirm", "hello grace do it",
        "grace go ahead", "hey grace yep", "okay grace yes",
    ]
    for i, u in enumerate(greeting_yes):
        items.append(YesNoItem(sample_id=f"greeting_yes_{i:02d}", utterance=u, category="greeting_yes", expected_label="YES"))

    greeting_no = [
        "hey grace no", "grace stop", "hey grace don't", "hi grace cancel",
        "hello grace nope", "grace abort",
    ]
    for i, u in enumerate(greeting_no):
        items.append(YesNoItem(sample_id=f"greeting_no_{i:02d}", utterance=u, category="greeting_no", expected_label="NO"))

    # 5. Repeat reflex items
    repeat_items = [
        "what did you say", "repeat that please", "could you repeat",
        "say again", "pardon", "what was that", "can you say that again",
    ]
    for i, u in enumerate(repeat_items):
        items.append(YesNoItem(sample_id=f"repeat_{i:02d}", utterance=u, category="repeat", expected_label="REPEAT"))

    # 6. Stop reflex items
    stop_items = [
        "stop", "stop right now", "shut up", "halt", "quit", "grace stop now",
    ]
    for i, u in enumerate(stop_items):
        items.append(YesNoItem(sample_id=f"stop_{i:02d}", utterance=u, category="stop", expected_label="STOP"))

    # 7. Unrelated / Out-of-domain abstain items
    abstain_items = [
        "open chrome", "what is the weather today", "send an email to john",
        "play some music", "turn up the volume", "how do you make coffee",
        "where is my mouse cursor", "search for tax forms",
    ]
    for i, u in enumerate(abstain_items):
        items.append(YesNoItem(sample_id=f"abstain_{i:02d}", utterance=u, category="abstain", expected_label="ABSTAIN"))

    return items


def extract_tool_calling_dataset(
    corpus_dir: Optional[Union[str, Path]] = None,
) -> List[ToolCallItem]:
    """Build tool calling benchmark items from corpus tapes and synthetic schema calls."""
    items: List[ToolCallItem] = []
    sample_idx = 0

    # 1. Extract from corpus tapes if available
    if corpus_dir:
        c_path = Path(corpus_dir)
        if c_path.is_dir():
            for tape_dir in sorted(c_path.iterdir()):
                if not tape_dir.is_dir():
                    continue

                stt_file = tape_dir / "stt.jsonl"
                dispatch_file = tape_dir / "dispatch.jsonl"
                if not stt_file.is_file() or not dispatch_file.is_file():
                    continue

                stt_lines: List[str] = []
                with open(stt_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                stt_lines.append(json.loads(line).get("text", "").strip())
                            except Exception:
                                pass

                dispatch_lines: List[Dict[str, Any]] = []
                with open(dispatch_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                dispatch_lines.append(json.loads(line))
                            except Exception:
                                pass

                for i, action in enumerate(dispatch_lines):
                    tool = action.get("tool", "")
                    params = action.get("params", {})
                    utt = stt_lines[i] if i < len(stt_lines) and stt_lines[i] else f"run {tool}"
                    is_multi = len(dispatch_lines) > 1

                    items.append(
                        ToolCallItem(
                            sample_id=f"corpus_{tape_dir.name}_{i}",
                            utterance=utt,
                            expected_tool=tool,
                            expected_args=params,
                            is_multi_turn=is_multi,
                            source="corpus",
                        )
                    )
                    sample_idx += 1

    # 2. Rich synthetic tool calls generated from Grace's tool schema
    synthetic_samples = [
        # System tools
        ("open chrome", "open_app", {"app_name": "chrome"}),
        ("launch calculator", "open_app", {"app_name": "calculator"}),
        ("start edge browser", "open_app", {"app_name": "edge"}),
        ("close notepad", "close_app", {"app_name": "notepad"}),
        ("quit spotify", "close_app", {"app_name": "spotify"}),
        ("turn the volume up", "adjust_volume", {"direction": "up", "step": 10}),
        ("mute the computer", "adjust_volume", {"direction": "mute"}),
        ("set volume to 60 percent", "adjust_volume", {"direction": "set", "level": 60}),
        ("lock my screen", "lock_computer", {}),
        ("lock the pc", "lock_computer", {}),
        ("find resume docx", "search_files", {"query": "resume.docx"}),
        ("search files for financial report", "search_files", {"query": "financial report"}),
        ("open notes txt", "open_file", {"path": "notes.txt"}),
        ("view invoice pdf", "open_file", {"path": "invoice.pdf"}),
        ("delete old log file", "delete_file", {"path": "old.log"}),
        ("read doc pdf", "read_pdf", {"path": "doc.pdf"}),
        ("summarize document pdf", "summarize_pdf", {"path": "document.pdf"}),
        ("undo that", "undo", {}),
        ("what is on my screen", "describe_screen", {}),
        ("speak faster", "set_speech_rate", {"rate": 1.2}),
        ("slow down your voice", "set_speech_rate", {"rate": 0.8}),

        # CUA tools
        ("click the submit button", "cua_click", {"element_id": 3}),
        ("click the search box", "cua_click", {"element_id": 2}),
        ("type hello world", "cua_type_text", {"text": "hello world"}),
        ("type my email address test at example dot com", "cua_type_text", {"text": "test@example.com"}),
        ("press enter", "cua_press_key", {"key": "enter"}),
        ("press tab", "cua_press_key", {"key": "tab"}),
        ("press alt f4", "cua_press_key", {"key": "alt+f4"}),
        ("press escape", "cua_press_key", {"key": "escape"}),
        ("scroll down", "cua_scroll", {"direction": "down", "amount": 3}),
        ("scroll up to the top", "cua_scroll", {"direction": "up", "amount": 5}),
        ("drag from button to folder", "cua_drag", {"from_id": 1, "to_id": 4}),
        ("set text value to 100", "cua_set_value", {"element_id": 5, "value": "100"}),
    ]

    for utt, tool, args in synthetic_samples:
        items.append(
            ToolCallItem(
                sample_id=f"synthetic_{sample_idx:03d}",
                utterance=utt,
                expected_tool=tool,
                expected_args=args,
                is_multi_turn=False,
                source="synthetic",
            )
        )
        sample_idx += 1

    return items


def save_dataset_jsonl(items: List[Any], path: Union[str, Path]) -> Path:
    """Save items list to a JSONL file atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        for item in items:
            d = item.to_dict() if hasattr(item, "to_dict") else asdict(item)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    tmp.replace(target)
    logger.info("Saved %d items to %s", len(items), target)
    return target


def load_element_selection_dataset(path: Union[str, Path]) -> List[ElementSelectionItem]:
    """Load ElementSelectionItem records from JSONL."""
    items: List[ElementSelectionItem] = []
    p = Path(path)
    if not p.is_file():
        return items
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(ElementSelectionItem.from_dict(json.loads(line)))
    return items


def load_yes_no_dataset(path: Union[str, Path]) -> List[YesNoItem]:
    """Load YesNoItem records from JSONL."""
    items: List[YesNoItem] = []
    p = Path(path)
    if not p.is_file():
        return items
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(YesNoItem.from_dict(json.loads(line)))
    return items


def load_tool_calling_dataset(path: Union[str, Path]) -> List[ToolCallItem]:
    """Load ToolCallItem records from JSONL."""
    items: List[ToolCallItem] = []
    p = Path(path)
    if not p.is_file():
        return items
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(ToolCallItem.from_dict(json.loads(line)))
    return items
