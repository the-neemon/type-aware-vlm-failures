"""Synthetic items for the caching run, and the scorer for absent-category answers.

Kept free of matplotlib so the GPU environment can import it:
src/extract/cache_chartqa.py reads a synthetic manifest through
`load_manifest_items` when its inference config names one.

A synthetic manifest (src/synth/absent_pairs.py writes one) has one JSON line
per figure, each with an `image_path` relative to the manifest's folder and a
`questions` list. Every question becomes one item.

Absent-category questions have no answer in the figure, so relaxed accuracy
cannot score them: whatever number the model says, `is_correct("not present", ...)`
is false, and so is a correct refusal. `classify_absent` scores them instead,
from the model's actual output (docs/taxonomy.md: label what the model did,
never what the question was designed to induce):

    rejected     the answer says the category is not there ("Not shown", "None")
    fabricated   the answer is a bare value (a number, including "0")
    unclear      anything else, e.g. a category name, or a number with words
                 around it ("45 (not shown)", "45 units");
                 read these by hand before they are used as labels
"""

from __future__ import annotations

import json
import pathlib
import re

from src.eval.relaxed_accuracy import is_correct

SOURCE = "synthetic"      # the `source` field hashed into every synthetic item_id

_REFUSAL = re.compile(
    r"\b(no|not|none|nothing|n/?a|unknown|unavailable|missing|absent|cannot|can't|"
    r"doesn't|isn't|there is no|there are no)\b", re.IGNORECASE)
_NUMBER = re.compile(r"^[-+]?\$?\d[\d,]*(\.\d+)?\s*%?$")


def classify_absent(prediction: str) -> str:
    """rejected / fabricated / unclear, for an answer about a category the figure lacks."""
    text = prediction.strip().rstrip(".").strip()
    if _NUMBER.match(text):
        return "fabricated"
    if _REFUSAL.search(text) and not re.search(r"\d", text):
        return "rejected"
    return "unclear"


def score(item: dict, prediction: str) -> dict:
    """The fields a synthetic prediction row adds, including `correct`.

    A present-category question is scored by relaxed accuracy, like ChartQA. An
    absent one is correct only when the model rejects it.
    """
    fields = {k: item[k] for k in ("template", "absent", "phrasing", "asks_about")}
    if item["absent"]:
        outcome = classify_absent(prediction)
        fields.update(outcome=outcome, correct=outcome == "rejected")
    else:
        fields["correct"] = is_correct(item["gold"], prediction)
    return fields


def load_manifest_items(manifest: pathlib.Path, splits: tuple[str, ...]) -> list[dict]:
    """One item per question, in manifest order, for figures in `splits`.

    Items carry what cache_chartqa's ChartQA items carry (figure_id, question,
    gold, split, source, image), plus the synthetic fields `score` copies.
    """
    manifest = pathlib.Path(manifest)
    if not manifest.is_file():
        raise SystemExit(f"synthetic manifest {manifest} not found; generate it first "
                         "(python -m src.synth.absent_pairs, see its docstring)")
    items = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        figure = json.loads(line)
        if figure["split"] not in splits:
            continue
        image = manifest.parent / figure["image_path"]
        for q in figure["questions"]:
            items.append({
                "figure_id": figure["figure_id"],
                "question": q["question"],
                "gold": str(q["gold_answer"]),
                "split": figure["split"],
                "source": SOURCE,
                "image": image,
                "template": q["template"],
                "absent": q["absent"],
                "phrasing": q["phrasing"],
                "asks_about": q["asks_about"],
            })
    return items
