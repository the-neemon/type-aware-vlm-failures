"""Human agreement check for the final submission (AGENTS.md decision 6).

Two people label the same 200 ChartQA errors blind (`pool.py sample`, then
`assign --overlap 200`). This writes `results/human_agreement.md`:

    1. Cohen's kappa between the two humans, per model and overall, against
       the 0.6 gate, with a 95% bootstrap interval over items (seed 42).
    2. Each human, and the adjudicated human consensus, against the LLM labels
       every experiment uses.
    3. Every item the humans disagree on, with both labels, both rationales and
       the adjudicated label (from `annotations/adjudicated.jsonl`, if settled).

The model of each item comes from `pool_key.jsonl`, joined only here, after
labelling. Tables and numbers only; the interpretation is written by hand.

    python -m src.label.human_agreement \\
        --raters shrish sanjith \\
        --labels results/labels/qwen2_5_vl_7b_test.claude.jsonl \\
        --labels results/labels/llava_next_mistral_7b_test.claude.jsonl \\
        --out results/human_agreement.md
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
from typing import Mapping, Sequence

from src.label.agreement import compute_cohen_kappa
from src.label.annotation import ALLOWED_LABELS
from src.label.multi_rater import KAPPA_THRESHOLD, Ratings, load_raters

SEED = 42
N_BOOT = 2000
MODELS = ("qwen2_5_vl_7b", "llava_next_mistral_7b")


def _read(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def kappa_ci(a: Sequence[str], b: Sequence[str]) -> tuple[float, float, float]:
    """Cohen's kappa and a percentile bootstrap 95% interval, resampling items."""
    k = compute_cohen_kappa(a, b, ALLOWED_LABELS)
    rng = random.Random(SEED)
    n, boots = len(a), []
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        bk = compute_cohen_kappa([a[i] for i in idx], [b[i] for i in idx], ALLOWED_LABELS)
        if not math.isnan(bk):
            boots.append(bk)
    if not boots:
        return k, math.nan, math.nan
    boots.sort()
    return k, boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]


def compare(x: Mapping[str, str], y: Mapping[str, str], items: Sequence[str]) -> dict:
    common = [i for i in items if i in x and i in y]
    if not common:
        return {"n": 0}
    a, b = [x[i] for i in common], [y[i] for i in common]
    k, lo, hi = kappa_ci(a, b)
    return {"n": len(common), "agreement": sum(p == q for p, q in zip(a, b)) / len(common),
            "kappa": k, "lo": lo, "hi": hi}


def consensus_labels(ratings: Ratings, raters: Sequence[str]) -> dict[str, str]:
    """Agreed items, plus adjudicated ones; adjudication overrides."""
    out = {}
    first, second = (ratings[r] for r in raters)
    for iid in set(first) & set(second):
        if first[iid]["label"] == second[iid]["label"]:
            out[iid] = first[iid]["label"]
    for iid, rec in ratings.get("adjudicated", {}).items():
        out[iid] = rec["label"]
    return out


def _fmt(row: dict) -> str:
    if not row["n"]:
        return "| 0 | | | |"
    gate = "" if math.isnan(row["kappa"]) else ("pass" if row["kappa"] >= KAPPA_THRESHOLD else "**fail**")
    return (f"| {row['n']} | {row['agreement']:.3f} | {row['kappa']:.3f} "
            f"[{row['lo']:.3f}, {row['hi']:.3f}] | {gate} |")


def report(ratings: Ratings, raters: Sequence[str], key: Sequence[dict],
           llm: Mapping[str, str], command: str) -> str:
    model_of = {k["item_id"]: k["model"] for k in key}
    groups = {m: sorted(i for i, mm in model_of.items() if mm == m) for m in MODELS}
    groups["overall"] = sorted(model_of)
    a, b = raters
    lab = {r: {i: rec["label"] for i, rec in ratings.get(r, {}).items() if i in model_of}
           for r in raters}
    cons = {i: l for i, l in consensus_labels(ratings, raters).items() if i in model_of}

    out = [f"# Human agreement check (AGENTS.md decision 6)", "",
           f"Command: `{command}`", "",
           f"Labels: {', '.join(ALLOWED_LABELS)}. Gate: kappa >= {KAPPA_THRESHOLD}. "
           f"Intervals: 95% percentile bootstrap over items, {N_BOOT} resamples, seed {SEED}.", ""]

    done = {r: len(lab[r]) for r in raters}
    if any(n < len(model_of) for n in done.values()):
        out += [f"**Incomplete:** " + ", ".join(f"{r} {n}/{len(model_of)}" for r, n in done.items())
                + ". Numbers below cover only the items labelled so far.", ""]

    out += [f"## 1. {a} vs {b}", "",
            "| subset | n | raw agreement | kappa [95% CI] | gate |", "|---|---|---|---|---|"]
    for g, items in groups.items():
        out.append(f"| {g} " + _fmt(compare(lab[a], lab[b], items)))

    out += ["", f"Confusion matrix, all items (rows {a}, columns {b}):", "",
            "| | " + " | ".join(ALLOWED_LABELS) + " |", "|---" * (len(ALLOWED_LABELS) + 1) + "|"]
    cm = collections.Counter((lab[a][i], lab[b][i]) for i in set(lab[a]) & set(lab[b]))
    for x in ALLOWED_LABELS:
        out.append(f"| {x} | " + " | ".join(str(cm[(x, y)]) for y in ALLOWED_LABELS) + " |")

    out += ["", "## 2. Humans vs the LLM labels", "",
            "| rater | subset | n | raw agreement | kappa [95% CI] | gate |",
            "|---|---|---|---|---|---|"]
    for name, x in [(a, lab[a]), (b, lab[b]), ("consensus", cons)]:
        for g, items in groups.items():
            out.append(f"| {name} | {g} " + _fmt(compare(x, llm, items)))
    out += ["", "`consensus`: items both humans agree on, plus adjudicated items "
            f"({sum(1 for i in cons if i in ratings.get('adjudicated', {}))} adjudicated, "
            f"{len(cons)} in total).", ""]

    rows = []
    for iid in sorted(set(lab[a]) & set(lab[b])):
        if lab[a][iid] == lab[b][iid]:
            continue
        rec = ratings[a][iid]
        adj = ratings.get("adjudicated", {}).get(iid, {}).get("label", "unresolved")
        why = lambda r: (ratings[r][iid].get("rationale") or "").replace("|", "/")
        rows.append(f"| {iid} | {model_of[iid]} | {rec.get('question', '').replace('|', '/')} "
                    f"| {rec.get('gold_answer', '')} | {rec.get('model_answer', '')} "
                    f"| {lab[a][iid]}: {why(a)} | {lab[b][iid]}: {why(b)} "
                    f"| {llm.get(iid, '')} | {adj} |")
    unresolved = sum(r.endswith("| unresolved |") for r in rows)
    out += [f"## 3. Disagreements ({len(rows)}, {unresolved} unresolved)", "",
            "Settle each in `annotations/adjudicated.jsonl` (one `{\"item_id\", \"label\"}` "
            "per line), then rerun.", "",
            f"| item | model | question | gold | model answer | {a} | {b} | LLM | adjudicated |",
            "|---|---|---|---|---|---|---|---|---|", *rows, ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--annotations", type=pathlib.Path, default=pathlib.Path("annotations"))
    ap.add_argument("--key", type=pathlib.Path, default=pathlib.Path("annotations/tasks/pool_key.jsonl"))
    ap.add_argument("--raters", nargs=2, default=["shrish", "sanjith"])
    ap.add_argument("--labels", action="append", required=True, type=pathlib.Path,
                    help="LLM label file; repeat once per model")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/human_agreement.md"))
    args = ap.parse_args()

    ratings, bad = load_raters(args.annotations)
    missing = [r for r in args.raters if r not in ratings]
    if missing:
        raise SystemExit(f"no labels yet from {missing} in {args.annotations}/")
    llm = {}
    for path in args.labels:
        for rec in _read(path):
            llm[rec["item_id"]] = rec["label"]          # last record wins
    command = ("python -m src.label.human_agreement --raters " + " ".join(args.raters)
               + "".join(f" --labels {p}" for p in args.labels) + f" --out {args.out}")
    text = report(ratings, args.raters, _read(args.key), llm, command)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"-> {args.out} ({bad} bad lines skipped)")


if __name__ == "__main__":
    main()
