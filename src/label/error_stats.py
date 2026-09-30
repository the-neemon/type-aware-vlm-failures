"""Numbers behind the error-set description in the report: what the labelled errors are.

Joins one model's predictions to its labels and counts labels and reasons by
question source (human-written vs augmented), how often the gold answer itself
is wrong, and how many "errors" are the right number on the other scale (0.43
for a gold of 43, or the reverse). Relaxed accuracy scores those as wrong, so
they inflate the error set; they are listed so the metric can be fixed upstream.

    python -m src.label.error_stats \\
        --predictions results/predictions/qwen2_5_vl_7b_test.jsonl \\
        --labels results/labels/qwen2_5_vl_7b_test.claude.jsonl \\
        --out results/labels/qwen2_5_vl_7b_test.error_stats.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

from src.eval.relaxed_accuracy import _try_parse_float, is_correct

REAL_ERRORS = ("structural", "fabrication", "computation")


def rescaled_match(gold: str, prediction: str) -> bool:
    """True when the prediction is relaxed-correct after multiplying by 100 or 0.01."""
    p = _try_parse_float(prediction)
    if p is None or _try_parse_float(gold) is None:
        return False
    return any(is_correct(gold, repr(p * k)) for k in (100, 0.01))


def stats(predictions: list[dict], labels: list[dict]) -> dict:
    rows = {r["item_id"]: r for r in predictions}
    # label files are append-only and a later line corrects an earlier one
    # (src/label/multi_rater.py, src/label/llm_label.py correct): keep the last
    labels = list({l["item_id"]: l for l in labels}.values())
    missing = [l["item_id"] for l in labels if l["item_id"] not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} labels have no prediction row, e.g. {missing[0]}")
    errors = [r for r in rows.values() if not r["correct"]]
    if {r["item_id"] for r in errors} != {l["item_id"] for l in labels}:
        raise SystemExit("labels do not cover exactly the incorrect predictions")

    sources = sorted({r["source"] for r in rows.values()})
    by_source = {}
    for s in sources:
        n = sum(r["source"] == s for r in rows.values())
        ls = [l for l in labels if rows[l["item_id"]]["source"] == s]
        by_source[s] = {
            "items": n,
            "errors": len(ls),
            "error_rate": round(len(ls) / n, 4),
            "labels": dict(collections.Counter(l["label"] for l in ls).most_common()),
            "real_errors": sum(l["label"] in REAL_ERRORS for l in ls),
            "gold_wrong": sum(l.get("reason") == "gold_error" for l in ls),
        }

    reasons = collections.defaultdict(collections.Counter)
    for l in labels:
        reasons[l["label"]][l.get("reason")] += 1

    rescaled = [l for l in labels
                if rescaled_match(rows[l["item_id"]]["gold"], rows[l["item_id"]]["prediction"])]
    n = len(rows)
    n_not_error = sum(l["label"] == "not_an_error" for l in labels)
    return {
        "items": n,
        "errors": len(labels),
        "by_source": by_source,
        "reasons": {k: dict(v.most_common()) for k, v in reasons.items()},
        "gold_wrong": sum(l.get("reason") == "gold_error" for l in labels),
        "rescaled": {
            "count": len(rescaled),
            "labels": dict(collections.Counter(
                f"{l['label']}/{l.get('reason')}" for l in rescaled)),
            "items": [{"item_id": l["item_id"], "gold": rows[l["item_id"]]["gold"],
                       "prediction": rows[l["item_id"]]["prediction"],
                       "source": rows[l["item_id"]]["source"]} for l in rescaled],
        },
        "accuracy": {
            "relaxed": round(1 - len(labels) / n, 4),
            "rescaled_counted_correct": round(1 - (len(labels) - len(rescaled)) / n, 4),
            "not_an_error_counted_correct": round(1 - (len(labels) - n_not_error) / n, 4),
        },
    }


def _read(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--predictions", type=pathlib.Path, required=True)
    ap.add_argument("--labels", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    out = stats(_read(args.predictions), _read(args.labels))
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("errors", "gold_wrong", "accuracy")}, indent=2))
    print("rescaled:", out["rescaled"]["count"])
    for s, v in out["by_source"].items():
        print(s, {k: v[k] for k in ("items", "errors", "error_rate", "real_errors", "gold_wrong")})


if __name__ == "__main__":
    main()
