"""E5 step 2: the type-aware controller against its baselines (AGENTS.md decision 7).

Universe: every value question of synthetic pilots 1 and 2 ("What is the value
of X?", X shown or missing), classes C, S, F, Z, R as in e5_scores.py. For each
question we know whether the original answer (`I_0`) and each repair is right:
errors from E4 (`results/e4/*.outcomes.jsonl`), right answers from the E5 re-ask
(`*.correct.outcomes.jsonl`), so a re-look that breaks a right answer is counted
against the policy that ordered it.

A policy gives every question a final answer (original or repaired) and a
confidence; sweeping an abstention threshold over the confidence gives a
risk-coverage curve. Risk is the error rate among answered questions. Ties in
confidence are averaged (the expected curve under a random order within a tie),
so no policy gains from an arbitrary ordering. AURC is the mean risk over the
curve; lower is better.

    plain           original answers, no abstention ranking (flat curve)
    binary          original answers, abstain on the wrong-vs-right probe first
    missing         original answers, abstain on the missing-bar probe first
    controller      re-look when the misread probe fires, abstain on the
                    missing-bar probe first (decision 7)
    relook_binary   re-look as the controller, abstain on the binary probe first
    oracle          sees the true type: applies the best repair for that type and
                    abstains first on the types that repair helps least

The ablation the spec asks for (does re-querying add anything beyond
abstention?) is controller vs missing, and relook_binary vs binary: each pair
shares its abstention order and differs only in the re-look.

Re-look threshold and repair are cross-fitted by figure fold: for fold k they
maximise accuracy on the other four folds, and are applied to fold k. Every
question is thus scored by a rule chosen without its chart. Intervals are
cluster bootstraps over figures, seed 42.

    python -m src.controller.e5 --models qwen2_5_vl_7b llava_next_mistral_7b
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

REPAIRS = ("I_reask", "I_upsample", "I_crop", "I_verify")
BASELINE = "I_0"
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
POLICIES = ("plain", "binary", "missing", "controller", "relook_binary", "oracle")
ABLATIONS = {"re-look beyond missing-bar abstention": ("controller", "missing"),
             "re-look beyond binary abstention": ("relook_binary", "binary"),
             "controller vs binary": ("controller", "binary")}
N_FOLDS = 5
NOTES_MARKER = "<!-- notes: written by hand, kept when this file is regenerated -->"


def _read(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load(model: str, root: pathlib.Path = pathlib.Path("results")) -> list[dict]:
    """Every scored question with its class, fold, probe scores and per-repair correctness."""
    scores = _read(root / "e5" / f"{model}_scores.jsonl")
    ok: dict[str, dict[str, bool]] = {}
    for path in sorted((root / "e4").glob(f"{model}_synth_pilot*.outcomes.jsonl")):
        for r in _read(path):
            if r.get("applicable", True):
                ok.setdefault(r["item_id"], {})[r["intervention"]] = bool(r["recovered"])
    items, missing = [], []
    for s in scores:
        o = ok.get(s["item_id"], {})
        if set(o) != {BASELINE, *REPAIRS}:
            missing.append(s["item_id"])
            continue
        items.append({**s, "ok": o})
    if missing:
        raise SystemExit(f"{model}: {len(missing)} scored questions lack E4 or re-ask outcomes, "
                         f"e.g. {missing[0]}")
    return items


def curve(ok: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Expected risk at each coverage k/n, k = 1..n, keeping the most confident first."""
    n = len(ok)
    order = np.argsort(-conf, kind="mergesort")
    c, wrong = conf[order], (~ok[order]).astype(float)
    risk = np.empty(n)
    errors_before, start = 0.0, 0
    while start < n:
        end = start + 1
        while end < n and c[end] == c[start]:
            end += 1
        g, e = end - start, wrong[start:end].sum()
        k = np.arange(start + 1, end + 1)
        risk[start:end] = (errors_before + (k - start) * e / g) / k
        errors_before += e
        start = end
    return risk


def aurc(risk: np.ndarray) -> float:
    return float(risk.mean())


def risk_at(risk: np.ndarray, coverage: float) -> float:
    return float(risk[max(1, round(coverage * len(risk))) - 1])


def crossfit_relook(items: list[dict]) -> tuple[np.ndarray, dict, np.ndarray]:
    """Final correctness under "re-look with repair r when misread score > t", per fold,
    the rule chosen for each fold, and which questions were re-looked."""
    folds = np.array([it["fold"] for it in items])
    misread = np.array([it["misread"] for it in items])
    ok0 = np.array([it["ok"][BASELINE] for it in items])
    okr = {r: np.array([it["ok"][r] for it in items]) for r in REPAIRS}
    final, chosen, relooked = ok0.copy(), {}, np.zeros(len(items), bool)
    for k in range(N_FOLDS):
        tune, apply = folds != k, folds == k
        grid = np.unique(np.quantile(misread[tune], np.linspace(0.5, 1.0, 51)))
        best = (ok0[tune].mean(), None, float("inf"))          # never re-look
        for r in REPAIRS:
            for t in grid:
                acc = np.where(misread[tune] > t, okr[r][tune], ok0[tune]).mean()
                if acc > best[0] + 1e-12:
                    best = (acc, r, float(t))
        _, r, t = best
        if r is not None:
            final[apply] = np.where(misread[apply] > t, okr[r][apply], ok0[apply])
            relooked[apply] = misread[apply] > t
        chosen[k] = {"repair": r, "threshold": None if r is None else t,
                     "relooked": int((misread[apply] > t).sum()) if r is not None else 0}
    return final, chosen, relooked


def oracle(items: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Best repair per true class; confidence is that repair's accuracy on the class."""
    cls = np.array([it["cls"] for it in items])
    final, conf, plan = np.zeros(len(items), bool), np.zeros(len(items)), {}
    for c in np.unique(cls):
        m = cls == c
        acc = {iv: np.mean([items[i]["ok"][iv] for i in np.flatnonzero(m)])
               for iv in (BASELINE, *REPAIRS)}
        iv = max(acc, key=lambda x: (acc[x], x == BASELINE))
        final[m] = [items[i]["ok"][iv] for i in np.flatnonzero(m)]
        conf[m] = acc[iv]
        plan[str(c)] = {"action": iv, "accuracy": acc[iv]}
    return final, conf, plan


def policies(items: list[dict]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    ok0 = np.array([it["ok"][BASELINE] for it in items])
    binary = np.array([it["binary"] for it in items])
    missing = np.array([it["missing"] for it in items])
    relook, chosen, relooked = crossfit_relook(items)
    o_final, o_conf, plan = oracle(items)
    cls = np.array([it["cls"] for it in items])
    by_class = {str(c): {"n": int((cls == c).sum()),
                         "relooked": int(relooked[cls == c].sum()),
                         "fixed": int((relook & ~ok0)[cls == c].sum()),
                         "broken": int((~relook & ok0)[cls == c].sum())}
                for c in "CSFZR" if (cls == c).any()}
    return ({"plain": (ok0, np.zeros(len(items))),
             "binary": (ok0, -binary),
             "missing": (ok0, -missing),
             "controller": (relook, -missing),
             "relook_binary": (relook, -binary),
             "oracle": (o_final, o_conf)},
            {"relook_per_fold": chosen, "relook_by_class": by_class, "oracle_plan": plan})


def _bootstrap(items, stat, n_boot=1000, seed=42):
    figs = np.array([it["figure_id"] for it in items])
    uniq = np.unique(figs)
    index = {g: np.flatnonzero(figs == g) for g in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([index[g] for g in rng.choice(uniq, len(uniq))])
        draws.append(stat(idx))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(items: list[dict], n_boot: int = 1000) -> dict:
    pol, details = policies(items)
    curves = {p: curve(*pol[p]) for p in POLICIES}
    out = {"n": len(items),
           "classes": {c: int(sum(it["cls"] == c for it in items)) for c in "CSFZR"},
           "policies": {}, "ablations": {}, **details}
    for p in POLICIES:
        ok, conf = pol[p]
        out["policies"][p] = {
            "aurc": aurc(curves[p]),
            "aurc_ci95": _bootstrap(items, lambda i: aurc(curve(ok[i], conf[i])), n_boot),
            "accuracy_full_coverage": float(ok.mean()),
            "risk_at": {str(c): risk_at(curves[p], c) for c in COVERAGES},
            "curve": curves[p].tolist()}
    for name, (a, b) in ABLATIONS.items():
        (oa, ca), (ob, cb) = pol[a], pol[b]
        diff = aurc(curves[a]) - aurc(curves[b])
        out["ablations"][name] = {
            "policies": [a, b], "aurc_difference": diff,
            "ci95": _bootstrap(items, lambda i: aurc(curve(oa[i], ca[i]))
                               - aurc(curve(ob[i], cb[i])), n_boot)}
    return out


def report(results: dict[str, dict]) -> str:
    lines = ["# E5: type-aware controller vs selective prediction", "",
             "Written by `python -m src.controller.e5`; method in its docstring. "
             "AURC is the mean risk over the risk-coverage curve (lower is better); "
             "95% intervals are cluster bootstraps over figures.", ""]
    for model, r in results.items():
        lines += [f"## {model} ({r['n']} value questions: "
                  + ", ".join(f"{c} {n}" for c, n in r["classes"].items() if n) + ")", "",
                  "| Policy | AURC [95% CI] | " + " | ".join(f"risk @ {c:.0%}" for c in COVERAGES)
                  + " |", "| --- | --- |" + " --- |" * len(COVERAGES)]
        for p in POLICIES:
            e = r["policies"][p]
            lo, hi = e["aurc_ci95"]
            lines.append(f"| {p} | {e['aurc']:.3f} [{lo:.3f}, {hi:.3f}] | "
                         + " | ".join(f"{e['risk_at'][str(c)]:.3f}" for c in COVERAGES) + " |")
        lines += ["", "| Comparison | AURC difference [95% CI] |", "| --- | --- |"]
        for name, a in r["ablations"].items():
            lo, hi = a["ci95"]
            lines.append(f"| {name} ({a['policies'][0]} minus {a['policies'][1]}) | "
                         f"{a['aurc_difference']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        folds = r["relook_per_fold"]
        lines += ["", "Re-look rule per fold (chosen on the other four): "
                  + "; ".join(f"fold {k}: " + (f"{v['repair']} above {v['threshold']:.3f}, "
                                               f"{v['relooked']} re-looked"
                                               if v["repair"] else "never re-look")
                              for k, v in folds.items()) + ".",
                  "", "What the re-look did, by class (C correct, S misread, F made-up number, "
                  "Z \"0\", R correct refusal):", "",
                  "| Class | Questions | Re-looked | Fixed | Broken |", "| --- | --- | --- | --- | --- |",
                  *[f"| {c} | {v['n']} | {v['relooked']} | {v['fixed']} | {v['broken']} |"
                    for c, v in r["relook_by_class"].items()],
                  "", "Oracle plan: " + "; ".join(
                      f"{c} → {v['action']} ({v['accuracy']:.3f})"
                      for c, v in r["oracle_plan"].items()) + ".", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("results"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    results = {}
    for m in args.models:
        results[m] = evaluate(load(m, args.root), args.n_boot)
        (args.root / "e5" / f"{m}_e5.json").write_text(
            json.dumps(results[m], indent=2) + "\n")
    out = args.root / "e5.md"
    notes = ""                     # hand-written notes below the marker survive a rerun
    if out.is_file() and NOTES_MARKER in out.read_text(encoding="utf-8"):
        notes = out.read_text(encoding="utf-8").split(NOTES_MARKER, 1)[1]
    out.write_text(report(results) + "\n" + NOTES_MARKER + notes, encoding="utf-8")
    print(report(results))


if __name__ == "__main__":
    main()
