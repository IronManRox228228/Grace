"""Fast snapshot labelling CLI tool for Grace (PLAN.md §3, §8, Task 11).

Allows the user to rapidly inspect recorded UIA/DOM snapshots, see the candidate
element summaries, specify or refine instructions, and label the target element ID
(including -1 for 'none of these / ask the user').
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional

# Add src and bench to sys.path
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bench.dataset_builder import (
    DEFAULT_DATASET_DIR,
    ElementSelectionItem,
    ElementSummary,
    build_element_summary,
    extract_app_name,
    load_element_selection_dataset,
    save_dataset_jsonl,
)

logger = logging.getLogger(__name__)


def format_element_row(el: ElementSummary, is_target: bool = False) -> str:
    """Format a single element into a terminal-friendly table row."""
    marker = ">>> " if is_target else "    "
    id_str = f"#{el.id:<3}"
    role_str = f"[{el.control_type:<10}]"
    name_str = f'"{el.name}"' if el.name else "<unnamed>"
    container_str = f" in {el.parent_path}" if el.parent_path else ""
    return f"{marker}{id_str} {role_str} {name_str}{container_str}"


def display_snapshot_prompt(
    sample_id: str,
    window_title: str,
    instruction: str,
    elements: List[ElementSummary],
    correct_id: Optional[int] = None,
) -> None:
    """Print the snapshot and element candidate table to stdout."""
    print("=" * 70)
    print(f"Snapshot: {sample_id}")
    print(f"Window:   {window_title}")
    print(f"Goal:     {instruction}")
    print("-" * 70)
    print("Candidates:")
    for el in elements:
        print(format_element_row(el, is_target=(correct_id is not None and el.id == correct_id)))
    print("    #-1   [action    ] \"none of these / ask the user\"")
    print("-" * 70)


def label_snapshot_interactive(
    sample_id: str,
    window_title: str,
    default_instruction: str,
    elements: List[ElementSummary],
    input_fn: Callable[[str], str] = input,
) -> Optional[ElementSelectionItem]:
    """Interactively label a single snapshot.

    Returns ElementSelectionItem, or None if skipped/quit.
    Raises KeyboardInterrupt or EOFError on abort.
    """
    valid_ids = {e.id for e in elements}
    valid_ids.add(-1)

    instruction = default_instruction
    while True:
        display_snapshot_prompt(sample_id, window_title, instruction, elements)
        prompt_msg = (
            f"Enter target element ID (or -1 for none, 's' to skip, 'q' to quit, 'e' to edit goal) "
            f"[{elements[0].id if elements else -1}]: "
        )
        try:
            choice = input_fn(prompt_msg).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return None

        if not choice and elements:
            choice = str(elements[0].id)

        if choice.lower() == "q":
            return None
        if choice.lower() == "s":
            print("Skipped.")
            return None
        if choice.lower() == "e":
            new_goal = input_fn(f"Enter new goal [{instruction}]: ").strip()
            if new_goal:
                instruction = new_goal
            continue

        try:
            chosen_id = int(choice)
            if chosen_id in valid_ids:
                app_name = extract_app_name(window_title)
                # Ensure -1 candidate is included
                all_candidates = list(elements)
                if not any(e.id == -1 for e in all_candidates):
                    all_candidates.append(
                        ElementSummary(
                            id=-1,
                            name="none of these / ask the user",
                            control_type="action",
                            parent_path="",
                            window_title=window_title,
                        )
                    )
                return ElementSelectionItem(
                    sample_id=sample_id,
                    instruction=instruction,
                    elements=all_candidates,
                    correct_id=chosen_id,
                    app_name=app_name,
                    is_held_out=False,
                )
            else:
                print(f"Invalid element ID #{chosen_id}. Valid choices: {sorted(list(valid_ids))}")
        except ValueError:
            print(f"Invalid input '{choice}'. Enter an integer ID, 's', 'e', or 'q'.")


def run_labeller(
    snapshots_path: Path,
    out_path: Path,
    resume: bool = True,
    input_fn: Callable[[str], str] = input,
    max_items: Optional[int] = None,
) -> int:
    """Run labelling session over a snapshots.jsonl file."""
    existing_items: List[ElementSelectionItem] = []
    labeled_sample_ids: set[str] = set()

    if out_path.is_file() and resume:
        existing_items = load_element_selection_dataset(out_path)
        labeled_sample_ids = {it.sample_id for it in existing_items}
        print(f"Loaded {len(existing_items)} existing labelled samples from {out_path}.")

    snapshots: List[Dict[str, Any]] = []
    if snapshots_path.is_file():
        with open(snapshots_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        snapshots.append(json.loads(line))
                    except Exception:
                        pass
    else:
        print(f"Snapshots file not found: {snapshots_path}")
        return 0

    print(f"Found {len(snapshots)} snapshots in {snapshots_path}.")
    count_added = 0

    for i, snap_obj in enumerate(snapshots):
        if max_items is not None and count_added >= max_items:
            break

        sample_id = f"{snapshots_path.stem}_snap_{i:03d}"
        if sample_id in labeled_sample_ids:
            continue

        snap_data = snap_obj.get("snapshot", snap_obj)
        win = snap_data.get("window", {})
        win_title = win.get("title", "Active Window")
        raw_elements = snap_data.get("elements", [])
        if not raw_elements:
            continue

        element_summaries = [build_element_summary(el, raw_elements, win_title) for el in raw_elements]
        default_goal = f"click {element_summaries[0].name}" if element_summaries and element_summaries[0].name else "select target element"

        labeled_item = label_snapshot_interactive(
            sample_id=sample_id,
            window_title=win_title,
            default_instruction=default_goal,
            elements=element_summaries,
            input_fn=input_fn,
        )

        if labeled_item is None:
            # Check if user wanted to quit
            break

        existing_items.append(labeled_item)
        labeled_sample_ids.add(sample_id)
        count_added += 1

        # Save incrementally
        save_dataset_jsonl(existing_items, out_path)
        print(f"Saved sample #{sample_id} (Total labelled: {len(existing_items)})")

    return count_added


def main() -> None:
    parser = argparse.ArgumentParser(description="Grace fast snapshot labelling CLI")
    parser.add_argument("--snapshots", type=Path, required=True, help="Path to snapshots.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_DATASET_DIR / "element_selection.jsonl", help="Output JSONL path")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing output instead of resuming")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of items to label")
    args = parser.parse_args()

    run_labeller(
        snapshots_path=args.snapshots,
        out_path=args.out,
        resume=not args.no_resume,
        max_items=args.max,
    )


if __name__ == "__main__":
    main()
