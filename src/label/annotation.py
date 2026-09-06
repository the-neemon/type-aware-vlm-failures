"""Data structures and I/O for the annotation tool (TASKS P2.2).

Guarantees:
1. Blinding: Model identity, model name, and prior/ground-truth labels are
   explicitly excluded from AnnotationItem so annotators never see them.
2. Immediate persistence: Every annotation record is appended to disk and
   flushed with fsync immediately upon entry to guarantee crash resilience.
3. Resumability: Automatically detects and skips previously annotated items.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
from dataclasses import asdict, dataclass
from typing import Sequence

ALLOWED_LABELS = ("structural", "fabrication", "ambiguous")
KEY_MAP = {
    "s": "structural",
    "f": "fabrication",
    "a": "ambiguous",
}


@dataclass(frozen=True)
class AnnotationItem:
    """An item presented to the annotator.

    Deliberately omits model identity, judge labels, or ground-truth failure types
    to enforce blind annotation.
    """
    item_id: str
    image_path: str
    question: str
    gold_answer: str
    model_answer: str

    def __post_init__(self):
        if not self.item_id:
            raise ValueError("item_id must not be empty")
        if not self.question:
            raise ValueError("question must not be empty")


@dataclass(frozen=True)
class AnnotationRecord:
    """A completed annotation entry written to disk."""
    item_id: str
    question: str
    gold_answer: str
    model_answer: str
    label: str
    rationale: str
    annotator: str
    timestamp: str

    def __post_init__(self):
        if self.label not in ALLOWED_LABELS:
            raise ValueError(
                f"Invalid label {self.label!r}; must be one of {ALLOWED_LABELS}"
            )
        if not self.annotator:
            raise ValueError("annotator name or id must not be empty")


def append_annotation(record: AnnotationRecord, output_path: str | pathlib.Path) -> None:
    """Append a single annotation record to disk and immediately flush/fsync.

    Guarantees that a crash during an annotation session costs at most one item.
    """
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass  # fsync may not be supported on all virtual/memory filesystems


def load_completed_ids(output_path: str | pathlib.Path) -> set[str]:
    """Return the set of item_ids already annotated in output_path."""
    path = pathlib.Path(output_path)
    if not path.is_file():
        return set()

    completed: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "item_id" in data:
                    completed.add(data["item_id"])
            except json.JSONDecodeError:
                continue
    return completed


def load_items(
    input_path: str | pathlib.Path,
    image_base_dir: str | pathlib.Path | None = None,
) -> list[AnnotationItem]:
    """Load items from a JSON Lines file.

    Accepts:
    1. Direct item records with {item_id, image_path, question, gold_answer, model_answer}
    2. Manifest rows with {figure_id, question, gold, prediction}
    3. Synthetic manifest entries with {figure_id, image_path, questions: [{question, gold_answer, ...}]}

    Strips any metadata related to model identity or ground-truth labels.
    """
    input_path = pathlib.Path(input_path)
    base_dir = pathlib.Path(image_base_dir) if image_base_dir else input_path.parent
    items: list[AnnotationItem] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # Case 1: Standard AnnotationItem dict
            if "item_id" in data and "question" in data:
                img_path = data.get("image_path") or data.get("figure_id") or ""
                p = pathlib.Path(img_path)
                if p.is_absolute() or p.is_file():
                    full_img_path = str(p)
                else:
                    full_img_path = str(base_dir / img_path)
                items.append(
                    AnnotationItem(
                        item_id=data["item_id"],
                        image_path=full_img_path,
                        question=str(data["question"]),
                        gold_answer=str(data.get("gold_answer") or data.get("gold") or ""),
                        model_answer=str(data.get("model_answer") or data.get("prediction") or ""),
                    )
                )

            # Case 2: ManifestRow format (src.eval.manifest)
            elif "figure_id" in data and "prediction" in data and "questions" not in data:
                img_path = str(base_dir / data["figure_id"])
                item_id = f"{data['figure_id']}::{data['question']}"
                items.append(
                    AnnotationItem(
                        item_id=item_id,
                        image_path=img_path,
                        question=str(data["question"]),
                        gold_answer=str(data.get("gold", "")),
                        model_answer=str(data.get("prediction", "")),
                    )
                )

            # Case 3: Synthetic dataset manifest format (figure with nested questions)
            elif "figure_id" in data and "questions" in data:
                fig_img = data.get("image_path", f"images/{data['figure_id']}.png")
                full_img_path = str(base_dir / fig_img) if not pathlib.Path(fig_img).is_absolute() else fig_img
                for q_idx, q in enumerate(data["questions"]):
                    q_id = f"{data['figure_id']}_q{q_idx}"
                    q_text = str(q.get("question", ""))
                    gold = str(q.get("gold_answer", ""))
                    model_ans = str(q.get("model_answer", q.get("prediction", "")))
                    items.append(
                        AnnotationItem(
                            item_id=q_id,
                            image_path=full_img_path,
                            question=q_text,
                            gold_answer=gold,
                            model_answer=model_ans,
                        )
                    )

    return items


def make_record(
    item: AnnotationItem,
    label: str,
    rationale: str,
    annotator: str,
) -> AnnotationRecord:
    """Create a validated AnnotationRecord with current UTC timestamp."""
    clean_label = label.strip().lower()
    if clean_label in KEY_MAP:
        clean_label = KEY_MAP[clean_label]

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return AnnotationRecord(
        item_id=item.item_id,
        question=item.question,
        gold_answer=item.gold_answer,
        model_answer=item.model_answer,
        label=clean_label,
        rationale=rationale.strip(),
        annotator=annotator.strip(),
        timestamp=now_iso,
    )
