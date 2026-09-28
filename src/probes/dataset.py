"""Assemble cached activations and labels into probe training sets (P5.1 to P5.3).

`src/probes/sweep.py` already holds the probe primitives. It wants three things
per task:

    X           : {layer index: (n_items, hidden)}   float array per layer
    y           : (n_items,)                          binary
    figure_ids  : (n_items,)                          for the cluster bootstrap

This module is what turns the artefacts on disk into those three, and it is
where the two mistakes that would quietly invalidate every probe number live.

**Mistake one: joining by position.** The activation cache, the predictions
manifest and the label files are written by three different programs at three
different times. Nothing guarantees they share a row order, and filtering to
"incorrect items only" destroys any order they did share. Everything here joins
on `item_id` and asserts the join covered what it should.

**Mistake two: splitting by question.** ChartQA has several questions per
figure, so a question-level split puts the same figure in train and test and
inflates every probe. Splits are assigned per figure, once, globally, and
`assert_no_figure_leak` is called on every dataset this module hands out.

Unknowns that are still open are collected in `ProbeConfig` rather than
scattered through the code. None of them change the shapes, so the pipeline can
be written and tested before they are decided.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

# The four pooled vectors written per layer by the caching job. Which of these
# carries the signal is an empirical question: HALP reports Qwen2.5-VL is best
# served by visual-only features where other architectures rely on late
# query-token states, so this is swept rather than assumed.
POSITIONS = ("vision_mean", "vision_max", "query_last", "query_mean")

Task = Literal["binary", "structural", "fabrication"]
TYPE_TASKS = ("structural", "fabrication")

# Labels that exist in the rubric. "ambiguous" is a real answer, not a failure
# to decide, and it is dropped from the type probes rather than coerced.
STRUCTURAL, FABRICATION, AMBIGUOUS = "structural", "fabrication", "ambiguous"
# The label set agreed on 27 Sep. Only STRUCTURAL and FABRICATION are type-probe
# classes; the other three are excluded from the type probes and counted.
COMPUTATION, NOT_AN_ERROR = "computation", "not_an_error"
ALL_LABELS = (STRUCTURAL, FABRICATION, COMPUTATION, NOT_AN_ERROR, AMBIGUOUS)


# ---------------------------------------------------------------------------
# Open hyperparameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeConfig:
    """Everything not yet decided, in one place.

    Defaults are placeholders chosen to be safe, not chosen to be right. Each
    is annotated with who decides it and what it would take to settle it.
    """

    # OPEN. Which layers were cached is Yash's call and lands in
    # configs/activations.yaml. None means "every layer present in the file",
    # which is also what `layers: all` resolves to.
    layers: tuple[int, ...] | None = None

    # OPEN. Which pooled position to probe. Swept, not assumed: run the whole
    # pipeline once per position and report all four. `query_last` is the
    # pre-generation state the hypothesis is literally about, so it is the
    # default, but SPEC 4.1 expects vision-side features to win for Qwen.
    position: str = "query_last"

    # OPEN. L2 strength. Selected on validation only, never on test.
    l2: float = 1.0

    # OPEN, and it changes what E3 can say. SPEC calls the type probes
    # "structural one-vs-rest" and "fabrication one-vs-rest" without saying
    # what "rest" is.
    #
    #   "errors_only"     rest = the other failure type. This is the hypothesis
    #                     as written: the state before a structural misreading
    #                     is distinguishable from the state before a
    #                     fabrication. But with exactly two classes the two
    #                     probes are the same probe with the label flipped, so
    #                     AUROC(fabrication) == 1 - AUROC(structural) exactly,
    #                     and E3's off-diagonal is algebraically forced rather
    #                     than measured. Report the diagonal and say so; do not
    #                     present the off-diagonal as evidence.
    #
    #   "include_correct" rest = the other failure type AND the correct items.
    #                     The two probes are then genuinely different models and
    #                     cross-transfer carries information. Costs a class
    #                     imbalance, since correct items outnumber errors ~4:1.
    #
    # Run both. They answer different questions and neither is wrong.
    type_probe_rest: str = "errors_only"

    # z-score every feature per layer, with mean and std estimated on TRAIN
    # rows only and then applied to val and test.
    #
    # Why it is on by default: on the real Qwen cache the activation norm grows
    # roughly 50x with depth (query_last is ~15 at L0 and ~812 at L27), and
    # fit_logistic applies one L2 penalty to raw features. Unstandardised, a
    # given l2 is ~50x weaker at the last layer than the first, so the L2 sweep
    # and the layer sweep become entangled and a layer can "win" because its
    # scale happens to suit the grid. That distorts the per-layer curve, which
    # is the E1 deliverable. Train-only statistics, for the same reason
    # residualise() takes fit_on: pooled statistics leak test information.
    standardize: bool = True

    # Split sizes. Figure-level. Seed is the project-wide 42 (TASKS 4.5).
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    seed: int = 42

    def __post_init__(self):
        if self.position not in POSITIONS:
            raise ValueError(f"position must be one of {POSITIONS}, got {self.position!r}")
        if self.type_probe_rest not in ("errors_only", "include_correct"):
            raise ValueError(
                "type_probe_rest must be 'errors_only' or 'include_correct', "
                f"got {self.type_probe_rest!r}")
        if not 0 < self.val_fraction + self.test_fraction < 1:
            raise ValueError("val + test fractions must leave a non-empty train split")


@dataclass(frozen=True)
class ProbeDataset:
    """One task, split three ways, ready for `sweep_layers`/`evaluate_at`."""
    task: str
    X: dict[str, dict[int, np.ndarray]]      # split -> layer -> (n, hidden)
    y: dict[str, np.ndarray]                 # split -> (n,)
    figure_ids: dict[str, np.ndarray]        # split -> (n,)
    item_ids: dict[str, np.ndarray]          # split -> (n,)
    # The underlying class per item: "structural", "fabrication" or "correct".
    # Kept separately from the binary `y` because E3's cross-transfer needs to
    # tell the three groups apart, and collapsing them into y loses "correct".
    classes: dict[str, np.ndarray] = field(default_factory=dict)
    dropped_ambiguous: int = 0
    layers: tuple[int, ...] = field(default_factory=tuple)

    def n(self, split: str) -> int:
        return len(self.y[split])

    def positive_rate(self, split: str) -> float:
        return float(np.mean(self.y[split])) if self.n(split) else float("nan")


# ---------------------------------------------------------------------------
# Cache validation
# ---------------------------------------------------------------------------

def validate_cache(
    path: str | pathlib.Path,
    expect_layers: int | None = 28,
    expect_hidden: int | None = 3584,
    expect_positions: Sequence[str] = POSITIONS,
) -> dict:
    """Check the activation cache is shaped the way the extractor promised.

    Cheap, and it runs before any probe is fitted. The failures it catches are
    the ones that otherwise produce a number rather than an error: a missing
    pooling position quietly narrows the sweep, and a NaN row makes
    `fit_logistic` return weights that score like noise. Neither raises on its
    own.

    Returns a report dict; raises only on structural problems.
    """
    report: dict = {"path": str(path), "problems": [], "warnings": []}

    with np.load(path, allow_pickle=True) as npz:
        keys = list(npz.files)
        if "item_ids" not in keys:
            raise ValueError(f"{path}: no item_ids array; the cache cannot be joined")

        item_ids = [str(x) for x in npz["item_ids"]]
        report["n_items"] = len(item_ids)
        report["n_unique_item_ids"] = len(set(item_ids))
        if len(set(item_ids)) != len(item_ids):
            report["problems"].append("duplicate item_ids")
        report["has_figure_ids"] = "figure_ids" in keys
        if "figure_ids" not in keys:
            report["problems"].append(
                "no figure_ids array; the cluster bootstrap would have to "
                "re-join to the predictions file to group by figure")

        found: dict[str, set[int]] = {p: set() for p in expect_positions}
        other = []
        for k in keys:
            if k in ("item_ids", "figure_ids"):
                continue
            for pos in expect_positions:
                if k.startswith("L") and k.endswith(f"_{pos}"):
                    found[pos].add(int(k[1:-len(pos) - 1]))
                    break
            else:
                other.append(k)
        report["layers_by_position"] = {p: sorted(v) for p, v in found.items()}
        report["unrecognised_keys"] = other

        for pos in expect_positions:
            if not found[pos]:
                report["problems"].append(f"position {pos!r} absent from the cache")
            elif expect_layers is not None and len(found[pos]) != expect_layers:
                report["warnings"].append(
                    f"position {pos!r} has {len(found[pos])} layers, "
                    f"expected {expect_layers}")

        # dtype, shape and finiteness on the arrays that are present
        n_nonfinite = 0
        for k in keys:
            if k in ("item_ids", "figure_ids") or k in other:
                continue
            arr = npz[k]
            if arr.shape[0] != len(item_ids):
                report["problems"].append(
                    f"{k} has {arr.shape[0]} rows, expected {len(item_ids)}")
            if expect_hidden is not None and arr.shape[-1] != expect_hidden:
                report["problems"].append(
                    f"{k} has width {arr.shape[-1]}, expected {expect_hidden}")
            if arr.dtype != np.float16:
                report["warnings"].append(f"{k} is {arr.dtype}, expected float16")
            bad = int((~np.isfinite(arr.astype(np.float32))).sum())
            n_nonfinite += bad
        report["n_nonfinite_values"] = n_nonfinite
        if n_nonfinite:
            report["problems"].append(
                f"{n_nonfinite} non-finite values; a NaN row does not raise, it "
                "makes the probe score like noise")

    if report["problems"]:
        raise ValueError(
            f"activation cache at {path} failed validation:\n  - "
            + "\n  - ".join(report["problems"]))
    return report


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# One entry per (path, position, layers). The sweep calls build_dataset once
# per position per task, which is 12 reads of a ~1.9 GiB compressed file if
# nothing is held. Decompression dominates the runtime otherwise. Each cached
# position is about 0.5 GiB at full scale, so the whole cache is ~1.9 GiB of
# RAM, which is nothing on a compute node. Call `clear_activation_cache()` if
# that is ever the wrong trade.
_ACT_CACHE: dict[tuple, tuple] = {}


def clear_activation_cache() -> None:
    _ACT_CACHE.clear()


def cached_layers(path: str | pathlib.Path, position: str = "query_last") -> list[int]:
    """Every layer index cached for `position`, read from the key names alone.

    The probe scripts default to this rather than to a fixed count, because the
    models differ: Qwen2.5-VL has 28 decoder layers, LLaVA-NeXT 32, and a
    hard-coded 28 would silently drop LLaVA's deepest four.
    """
    suffix = f"_{position}"
    with np.load(path, allow_pickle=True) as npz:
        layers = sorted(int(k[1:-len(suffix)]) for k in npz.files
                        if k.startswith("L") and k.endswith(suffix))
    if not layers:
        raise ValueError(f"{path} has no arrays for position {position!r}")
    return layers


def load_activations(
    path: str | pathlib.Path,
    position: str,
    layers: Sequence[int] | None = None,
) -> tuple[dict[int, np.ndarray], list[str], list[str]]:
    """Read one pooled position out of the cache.

    Returns `(by_layer, item_ids, figure_ids)`. `item_ids[i]` names the item in
    row `i` of every array, which is what makes the join checkable rather than
    assumed.

    Memoised: see `_ACT_CACHE`.
    """
    key = (str(path), position, tuple(layers) if layers is not None else None)
    if key in _ACT_CACHE:
        return _ACT_CACHE[key]
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}, got {position!r}")

    with np.load(path, allow_pickle=True) as npz:
        if "item_ids" not in npz:
            raise ValueError(
                f"{path} has no `item_ids` array. Row order is then unknowable "
                "and the cache cannot be joined to labels; re-run the caching "
                "job (see inf.md Section 3.3).")
        item_ids = [str(x) for x in npz["item_ids"]]
        figure_ids = ([str(x) for x in npz["figure_ids"]]
                      if "figure_ids" in npz else [])

        suffix = f"_{position}"
        # Map layer -> key WITHOUT reading anything. An .npz decompresses a
        # member only when it is indexed, so selecting keys first means asking
        # for one layer costs one decompression, not 28.
        available = {}
        for key in npz.files:
            if key.startswith("L") and key.endswith(suffix):
                available[int(key[1:-len(suffix)])] = key

        if not available:
            raise ValueError(
                f"{path} contains no arrays for position {position!r}. "
                f"Keys present: {sorted(npz.files)[:6]}")

        if layers is not None:
            missing = set(layers) - set(available)
            if missing:
                raise ValueError(f"requested layers absent from cache: {sorted(missing)}")
            wanted = list(layers)
        else:
            wanted = sorted(available)
        found = {L: npz[available[L]] for L in wanted}

    n = len(item_ids)
    for L, arr in found.items():
        if arr.shape[0] != n:
            raise ValueError(
                f"layer {L} has {arr.shape[0]} rows but there are {n} item_ids; "
                "the cache is internally inconsistent")

    _ACT_CACHE[key] = (found, item_ids, figure_ids)
    return found, item_ids, figure_ids


def load_predictions(path: str | pathlib.Path) -> dict[str, dict]:
    """Read the inference manifest into `{item_id: row}`.

    Expects the schema in inf.md Section 3.2. `item_id` is required: without it
    there is nothing to join on.
    """
    rows: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            iid = obj.get("item_id")
            if not iid:
                raise ValueError(
                    f"{path}:{lineno} has no item_id. The predictions file must "
                    "carry it (inf.md Section 3.2); recomputing it here would "
                    "duplicate the hash definition and invite drift.")
            if iid in rows:
                raise ValueError(f"{path}:{lineno} duplicate item_id {iid!r}")
            rows[iid] = obj
    return rows


def resolve_labels(
    annotations_dir: str | pathlib.Path,
    min_raters: int = 1,
) -> tuple[dict[str, str], dict[str, int]]:
    """Collapse annotation files into one label per item.

    *annotations_dir* may be a directory, in which case every top-level
    `*.jsonl` is one rater (the convention in `src/label/multi_rater.py`), or a
    single `.jsonl` file, read as one rater. The single-file form is how the
    LLM labels are stored.

    Resolution rule, deliberately conservative:
      - one rater          -> that label
      - clear majority     -> the majority label
      - tie, or any rater said `ambiguous` and no majority -> `ambiguous`

    A tie becomes `ambiguous` rather than being broken arbitrarily, because an
    item two people read differently is exactly what the rubric calls
    unresolvable. Those items are dropped from the type probes and counted.

    Returns `(labels, stats)`.
    """
    from src.label.multi_rater import load_raters

    path = pathlib.Path(annotations_dir)
    if path.is_file():
        # A single label file, which is how the LLM labeller's output is stored
        # (docs/annotation.md: results/labels/<model>_<split>.claude.jsonl). It
        # lives outside annotations/ on purpose, so it never counts as a rater
        # in the human kappa. Read it as one rater; later lines win, matching
        # multi_rater's append-only convention.
        ratings, bad_lines = {path.stem: {}}, 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                if not rec.get("item_id") or rec.get("label") not in ALL_LABELS:
                    bad_lines += 1
                    continue
                ratings[path.stem][rec["item_id"]] = rec
    else:
        ratings, bad_lines = load_raters(path)

    per_item: dict[str, list[str]] = {}
    for _rater, by_item in ratings.items():
        for iid, rec in by_item.items():
            per_item.setdefault(iid, []).append(rec["label"])

    labels: dict[str, str] = {}
    n_tied = 0
    for iid, votes in per_item.items():
        if len(votes) < min_raters:
            continue
        counts: dict[str, int] = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        top = max(counts.values())
        winners = [k for k, c in counts.items() if c == top]
        if len(winners) == 1:
            labels[iid] = winners[0]
        else:
            labels[iid] = AMBIGUOUS
            n_tied += 1

    stats = {
        "n_raters": len(ratings),
        "n_items_labelled": len(labels),
        "n_ties_to_ambiguous": n_tied,
        "n_bad_lines": bad_lines,
    }
    return labels, stats


# ---------------------------------------------------------------------------
# Figure-level splits (TASKS 5.4)
# ---------------------------------------------------------------------------

def assign_figure_splits(
    figure_ids: Iterable[str],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, str]:
    """Map every figure to exactly one of train/val/test.

    Assigned per figure and once, globally, so that the binary probe and the two
    type probes share a split even though they run over different populations.
    If each task split independently, an item could be train for one probe and
    test for another, and the comparison between probes would be meaningless.

    Assignment is by hashing the figure id rather than by shuffling, so it is
    stable when figures are added: re-running after more inference does not
    reshuffle the figures already assigned.
    """
    if not 0 < val_fraction + test_fraction < 1:
        raise ValueError("val + test fractions must leave a non-empty train split")

    out: dict[str, str] = {}
    for fid in dict.fromkeys(figure_ids):          # dedupe, keep order
        h = hashlib.sha1(f"{seed}\x1f{fid}".encode("utf-8")).hexdigest()
        u = int(h[:8], 16) / 0xFFFFFFFF            # deterministic uniform in [0,1]
        if u < test_fraction:
            out[fid] = "test"
        elif u < test_fraction + val_fraction:
            out[fid] = "val"
        else:
            out[fid] = "train"
    return out


def assert_no_figure_leak(figure_ids_by_split: Mapping[str, Sequence[str]]) -> None:
    """Fail if any figure appears in more than one split.

    Called on every dataset this module produces. TASKS 5.4 asks for an
    assertion in code rather than trust in the split script, because a leak
    inflates every probe number and looks like a good result.
    """
    seen: dict[str, str] = {}
    for split, fids in figure_ids_by_split.items():
        for fid in set(fids):
            if fid in seen and seen[fid] != split:
                raise AssertionError(
                    f"figure {fid!r} appears in both {seen[fid]!r} and {split!r}. "
                    "A figure contributes several questions, so this leaks the "
                    "figure across the split boundary and inflates the probe.")
            seen[fid] = split


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

def standardize_by_train(
    X: Mapping[str, Mapping[int, np.ndarray]],
    eps: float = 1e-6,
) -> dict[str, dict[int, np.ndarray]]:
    """z-score each layer's features using statistics from the train split only.

    Returns float32 arrays. Constant features (std below `eps` on train) are
    left centred rather than divided by ~0, which would turn a dead unit into a
    huge one.
    """
    out: dict[str, dict[int, np.ndarray]] = {s: {} for s in X}
    for layer in X["train"]:
        tr = X["train"][layer].astype(np.float32)
        mu = tr.mean(axis=0)
        sd = tr.std(axis=0)
        sd = np.where(sd < eps, 1.0, sd)
        for split in X:
            out[split][layer] = ((X[split][layer].astype(np.float32) - mu) / sd
                                 ).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_dataset(
    task: Task,
    activations_path: str | pathlib.Path,
    predictions_path: str | pathlib.Path,
    annotations_dir: str | pathlib.Path | None = None,
    config: ProbeConfig | None = None,
) -> ProbeDataset:
    """Assemble one probe task.

    Populations differ by task, and deliberately so:

      binary       every item that was run, correct and incorrect alike. The
                   probe asks "is this answer wrong", so it needs both classes.
      structural   incorrect items only, y = (label == "structural")
      fabrication  incorrect items only, y = (label == "fabrication")

    The type probes need `annotations_dir`; the binary probe does not, and runs
    before any labelling exists. `ambiguous` items are dropped from the type
    probes and counted in `dropped_ambiguous`.
    """
    cfg = config or ProbeConfig()
    if task not in ("binary",) + TYPE_TASKS:
        raise ValueError(f"unknown task {task!r}")
    if task in TYPE_TASKS and annotations_dir is None:
        raise ValueError(f"task {task!r} needs annotations_dir")

    by_layer, act_ids, act_figs = load_activations(
        activations_path, cfg.position, cfg.layers)
    preds = load_predictions(predictions_path)

    row_of = {iid: i for i, iid in enumerate(act_ids)}

    # Figure ids: prefer the cache's own copy, fall back to the manifest.
    if act_figs:
        fig_of = dict(zip(act_ids, act_figs))
    else:
        fig_of = {iid: preds[iid]["figure_id"] for iid in act_ids if iid in preds}

    labels: dict[str, str] = {}
    dropped = 0
    if task in TYPE_TASKS:
        labels, _stats = resolve_labels(annotations_dir)

    # --- choose the population -------------------------------------------
    keep: list[str] = []
    for iid in act_ids:
        row = preds.get(iid)
        if row is None:
            continue                       # cached but never scored; skip
        if task == "binary":
            keep.append(iid)
            continue
        if row["correct"]:
            # Correct items are the negative class only when "rest" is defined
            # to include them; see ProbeConfig.type_probe_rest.
            if cfg.type_probe_rest == "include_correct":
                keep.append(iid)
            continue
        lab = labels.get(iid)
        if lab is None:
            continue                       # not yet labelled
        if lab not in (STRUCTURAL, FABRICATION):
            dropped += 1                   # ambiguous, computation, not_an_error
            continue
        keep.append(iid)

    if not keep:
        raise ValueError(
            f"task {task!r} has no usable items. Check that the predictions and "
            "the activation cache share item_ids (inf.md Section 3.1).")

    # --- split by figure --------------------------------------------------
    split_of_figure = assign_figure_splits(
        (fig_of[i] for i in act_ids if i in fig_of),
        cfg.val_fraction, cfg.test_fraction, cfg.seed)

    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for iid in keep:
        buckets[split_of_figure[fig_of[iid]]].append(iid)

    # --- materialise ------------------------------------------------------
    X: dict[str, dict[int, np.ndarray]] = {}
    y: dict[str, np.ndarray] = {}
    figs: dict[str, np.ndarray] = {}
    iids: dict[str, np.ndarray] = {}
    klass: dict[str, np.ndarray] = {}

    for split, members in buckets.items():
        idx = np.array([row_of[i] for i in members], dtype=int)
        X[split] = {L: arr[idx] for L, arr in by_layer.items()}
        if task == "binary":
            vals = [not preds[i]["correct"] for i in members]
        else:
            # A correct item is a negative for both type probes: it is neither
            # a structural misreading nor a fabrication.
            vals = [(not preds[i]["correct"]) and labels[i] == task
                    for i in members]
        y[split] = np.asarray(vals, dtype=float)
        figs[split] = np.array([fig_of[i] for i in members], dtype=object)
        iids[split] = np.array(members, dtype=object)
        klass[split] = np.array(
            ["correct" if preds[i]["correct"] else labels.get(i, AMBIGUOUS)
             for i in members], dtype=object)

    if cfg.standardize:
        X = standardize_by_train(X)

    assert_no_figure_leak({s: figs[s] for s in figs})

    return ProbeDataset(
        task=task, X=X, y=y, figure_ids=figs, item_ids=iids, classes=klass,
        dropped_ambiguous=dropped, layers=tuple(sorted(by_layer)),
    )
