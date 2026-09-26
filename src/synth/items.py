"""Synthetic items for the caching run, and the scorer for absent-category answers.

Kept free of matplotlib so the GPU environment can import it:
src/extract/cache_chartqa.py reads a synthetic manifest through
`load_manifest_items` when its inference config names one.

A synthetic manifest (src/synth/absent_pairs.py and absent_lookalikes.py write
them) has one JSON line per figure, each with an `image_path` relative to the
manifest's folder and a `questions` list. Every question becomes one item.

Absent-category questions have no answer in the figure, so relaxed accuracy
cannot score them: whatever the model says, `is_correct("not present", ...)` is
false, and so is a correct refusal. They are scored from the model's actual
output instead (docs/taxonomy.md: label what the model did, never what the
question was designed to induce). Outcomes, by question:

  "What is the value of Guava?"  (absent_category, absent_value)
    rejected        says it is not there ("None", "Not shown")
    zero            "0". Kept apart from fabricated: on the first pilot Qwen gave
                    it 276 times in 300, and whether it means "no bar" or asserts
                    a value is an open decision, not something to settle here
    fabricated      any other bare number
  "Which is larger, Guava or Apple?"  (absent_compare)
    rejected        says one of them is not there
    fabricated      names the absent one
    picked_present  names the present one: arguably the same escape as "0"
  "Which bar is immediately to the right of Guava?"  (absent_neighbor)
    rejected        says it is not there
    fabricated      names a bar of the chart
  any question
    unclear         anything else; read these by hand before using them as labels

An absent question counts as `correct` only when rejected.
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
# fields copied from a manifest question onto its item, and from there onto the
# prediction row, when the generator wrote them
_OPTIONAL = ("family", "lookalike_of", "compared_with", "closest_pair", "gap")


def _clean(prediction: str) -> str:
    return prediction.strip().rstrip(".").strip()


def _refuses(text: str) -> bool:
    return bool(_REFUSAL.search(text)) and not re.search(r"\d", text)


def classify_absent(prediction: str) -> str:
    """rejected / zero / fabricated / unclear, for a value asked of a missing category."""
    text = _clean(prediction)
    if _NUMBER.match(text):
        return "zero" if float(text.lstrip("+$").rstrip("%").replace(",", "")) == 0 else "fabricated"
    return "rejected" if _refuses(text) else "unclear"


def classify_compare(prediction: str, absent: str, present: str) -> str:
    """rejected / fabricated / picked_present / unclear, for "Which is larger, absent or present?"."""
    text = _clean(prediction)
    if text.lower() == absent.lower():
        return "fabricated"
    if text.lower() == present.lower():
        return "picked_present"
    return "rejected" if _refuses(text) else "unclear"


def classify_neighbor(prediction: str, categories: list[str]) -> str:
    """rejected / fabricated / unclear, for "Which bar is next to <missing category>?"."""
    text = _clean(prediction)
    if text.lower() in {c.lower() for c in categories}:
        return "fabricated"
    return "rejected" if _refuses(text) else "unclear"


def score(item: dict, prediction: str) -> dict:
    """The fields a synthetic prediction row adds, including `correct`.

    A present-category question is scored by relaxed accuracy, like ChartQA. An
    absent one is correct only when the model rejects it.
    """
    fields = {k: item[k] for k in ("template", "absent", "phrasing", "asks_about")}
    fields.update({k: item[k] for k in _OPTIONAL if k in item})
    if not item["absent"]:
        fields["correct"] = is_correct(item["gold"], prediction)
        return fields
    if item["template"] == "absent_compare":
        outcome = classify_compare(prediction, item["asks_about"], item["compared_with"])
    elif item["template"] == "absent_neighbor":
        outcome = classify_neighbor(prediction, item["categories"])
    else:                                   # absent_category (pilot 1), absent_value
        outcome = classify_absent(prediction)
    fields.update(outcome=outcome, correct=outcome == "rejected")
    return fields


def load_manifest_items(manifest: pathlib.Path, splits: tuple[str, ...]) -> list[dict]:
    """One item per question, in manifest order, for figures in `splits`.

    Items carry what cache_chartqa's ChartQA items carry (figure_id, question,
    gold, split, source, image), plus the synthetic fields `score` uses.
    """
    manifest = pathlib.Path(manifest)
    if not manifest.is_file():
        raise SystemExit(f"synthetic manifest {manifest} not found; generate it first "
                         "(the generator named in the inference config's comments)")
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
                "categories": figure["categories"],
                "template": q["template"],
                "absent": q["absent"],
                "phrasing": q["phrasing"],
                "asks_about": q["asks_about"],
                **{k: q[k] for k in _OPTIONAL if k in q},
            })
    return items
