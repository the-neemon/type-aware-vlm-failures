"""Agreement across several annotators, including an LLM labeller (TASKS P2.4, P2.5).

Every top-level `annotations/<rater>.jsonl` is one rater. A human's file is
written by the annotation app; an LLM labeller's file only needs `item_id` and
`label` per line, with IDs taken from the same pool.jsonl, so it drops into the
same comparison with no special casing.

Records are append-only. When a rater relabels an item, the later record wins,
which is how the app implements "change my last answer" without rewriting a
file on disk.

Two statistics, for different questions:

    fleiss_kappa   the headline gate (kappa > 0.6). Overlap items are spread
                   across all six annotator pairs, so "rater 1" is not a fixed
                   person, and Cohen's per-rater marginals are ill-defined.
                   Fleiss pools them correctly and allows 2 or 3 raters per item.
    pairwise       Cohen's kappa for each pair, to find the one annotator whose
                   reading of the rubric has drifted from everyone else's.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import pathlib
from typing import Mapping, Sequence

from src.label.agreement import compute_cohen_kappa
from src.label.annotation import ALLOWED_LABELS

KAPPA_THRESHOLD = 0.6
MIN_PAIR_ITEMS = 30   # below this a pairwise kappa is noise; shown, but flagged

Ratings = Mapping[str, Mapping[str, dict]]   # rater -> item_id -> record


def load_raters(directory: str | pathlib.Path) -> tuple[dict[str, dict[str, dict]], int]:
    """Read every top-level *.jsonl as one rater. Returns (ratings, bad_line_count).

    Subdirectories (annotations/tasks/ holds the inputs) are ignored. Lines that
    are not valid JSON, lack an item_id, or carry a label outside the rubric are
    skipped and counted rather than raised, so one stray line in a teammate's
    file cannot take the whole agreement page down.
    """
    ratings: dict[str, dict[str, dict]] = {}
    bad = 0
    for path in sorted(pathlib.Path(directory).glob("*.jsonl")):
        records: dict[str, dict] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if not rec.get("item_id") or rec.get("label") not in ALLOWED_LABELS:
                    bad += 1
                    continue
                records[rec["item_id"]] = rec          # last record wins
        if records:
            ratings[path.stem] = records
    return ratings, bad


def fleiss_kappa(items: Sequence[Sequence[str]],
                 categories: Sequence[str] = ALLOWED_LABELS) -> float:
    """Fleiss's kappa over items rated by two or more raters (variable counts allowed).

    Returns nan when chance agreement is total, i.e. every rating falls in one
    category. That is not perfect agreement in any useful sense, and reporting
    it as 1.0 would pass the gate on a degenerate set.
    """
    rated = [labels for labels in items if len(labels) >= 2]
    if not rated:
        return float("nan")

    totals = collections.Counter()
    p_bar = 0.0
    for labels in rated:
        counts = collections.Counter(labels)
        n = len(labels)
        p_bar += (sum(c * c for c in counts.values()) - n) / (n * (n - 1))
        totals.update(counts)
    p_bar /= len(rated)

    n_all = sum(totals.values())
    p_e = sum((totals[c] / n_all) ** 2 for c in categories)
    if math.isclose(p_e, 1.0):
        return float("nan")
    return (p_bar - p_e) / (1 - p_e)


def shared_items(ratings: Ratings, raters: Sequence[str]) -> dict[str, list[str]]:
    """item_id -> labels, for items that at least two of `raters` labelled."""
    by_item: dict[str, list[str]] = collections.defaultdict(list)
    for r in raters:
        for iid, rec in ratings[r].items():
            by_item[iid].append(rec["label"])
    return {iid: labels for iid, labels in by_item.items() if len(labels) >= 2}


def pairwise(ratings: Ratings) -> list[dict]:
    """Cohen's kappa for every pair of raters with at least one shared item."""
    rows = []
    for a, b in itertools.combinations(sorted(ratings), 2):
        common = sorted(set(ratings[a]) & set(ratings[b]))
        if not common:
            continue
        la = [ratings[a][i]["label"] for i in common]
        lb = [ratings[b][i]["label"] for i in common]
        rows.append({
            "rater_a": a, "rater_b": b, "n": len(common),
            "agreement": round(sum(x == y for x, y in zip(la, lb)) / len(common), 3),
            "kappa": compute_cohen_kappa(la, lb, ALLOWED_LABELS),
            "enough_items": len(common) >= MIN_PAIR_ITEMS,
        })
    return rows


def disagreements(ratings: Ratings, raters: Sequence[str]) -> list[dict]:
    """Items where the selected raters do not all agree, with each rater's reasoning."""
    out = []
    for iid, labels in sorted(shared_items(ratings, raters).items()):
        if len(set(labels)) == 1:
            continue
        votes = {r: ratings[r][iid] for r in raters if iid in ratings[r]}
        any_rec = next(iter(votes.values()))
        out.append({
            "item_id": iid,
            "question": any_rec.get("question", ""),
            "gold_answer": any_rec.get("gold_answer", ""),
            "model_answer": any_rec.get("model_answer", ""),
            "votes": {r: (v["label"], v.get("rationale", "")) for r, v in votes.items()},
        })
    return out


def label_counts(ratings: Ratings) -> dict[str, dict[str, int]]:
    return {r: {c: sum(rec["label"] == c for rec in recs.values()) for c in ALLOWED_LABELS}
            for r, recs in ratings.items()}


def consensus(ratings: Ratings, raters: Sequence[str],
              adjudicator: str = "adjudicated") -> tuple[list[dict], list[str]]:
    """One label per item, for E4 and the type probes. Returns (labels, unresolved ids).

    An item gets a label when every selected rater who labelled it agrees, or
    when the `adjudicated` rater file has settled it after discussion; that
    file overrides everyone. Disagreements nobody has settled are held back
    rather than broken by vote, because with two raters there is no majority.
    """
    by_item: dict[str, list[str]] = collections.defaultdict(list)
    for r in raters:
        for iid, rec in ratings[r].items():
            by_item[iid].append(rec["label"])
    settled = ratings.get(adjudicator, {})

    labels, unresolved = [], []
    for iid in sorted(set(by_item) | set(settled)):
        if iid in settled:
            labels.append({"item_id": iid, "label": settled[iid]["label"], "how": "adjudicated"})
        elif len(set(by_item[iid])) == 1:
            how = "agreed" if len(by_item[iid]) > 1 else "single"
            labels.append({"item_id": iid, "label": by_item[iid][0], "how": how})
        else:
            unresolved.append(iid)
    return labels, unresolved


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge annotator files into one label per item")
    ap.add_argument("--annotations", type=pathlib.Path, default=pathlib.Path("annotations"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/labels/test.human.jsonl"))
    args = ap.parse_args()

    ratings, bad = load_raters(args.annotations)
    humans = [r for r in ratings if r != "adjudicated"]
    labels, unresolved = consensus(ratings, humans)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in labels:
            f.write(json.dumps(row) + "\n")
    print(f"{len(labels)} labels -> {args.out}; {len(unresolved)} unresolved "
          f"disagreements (settle them in annotations/adjudicated.jsonl); {bad} bad lines")


if __name__ == "__main__":
    main()
