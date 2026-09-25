"""Prediction rows, activation shards and the final cache (inf.md 3 and 7).

Numpy only, so all of it is testable without a GPU.

    predictions   results/predictions/<model>_test.jsonl, one row per question,
                  appended and fsynced as each item finishes.
    shards        shard_<first item_id>.npz, one per ~250 items, written
                  atomically. Each carries its own item_ids, so assembly never
                  depends on file names or order.
    final cache   one compressed .npz: item_ids, figure_ids and an (N, hidden)
                  float16 array per `L{layer}_{position}` key.

An item counts as done only once its activations are in a shard. Its
prediction row is written first, so a job killed between the two leaves a row
with no activations; the item is then regenerated and its new answer must equal
the stored one exactly (see `cache_chartqa.run`).
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
from typing import Mapping, Sequence

import numpy as np

from src.label.pool import item_id
from src.probes.dataset import load_predictions

PREDICTION_FIELDS = ("item_id", "model", "figure_id", "question", "gold",
                     "prediction", "correct", "split", "source")


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def append_prediction(row: dict, path: pathlib.Path) -> None:
    """Append one row and fsync it, so a crash costs at most the item in flight."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_predictions(path: pathlib.Path) -> dict[str, dict]:
    """`{item_id: row}`; empty when the file does not exist. Duplicate ids raise."""
    return load_predictions(path) if path.is_file() else {}


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------

def _atomic_savez(path: pathlib.Path, arrays: Mapping[str, np.ndarray], compress: bool) -> None:
    """Write to a temporary name in the same directory, then rename into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        (np.savez_compressed if compress else np.savez)(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def copy_atomic(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy so that `dst` is either absent or complete, never half-written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def shard_name(item_ids: Sequence[str]) -> str:
    """Unique because every item lands in exactly one shard."""
    return f"shard_{item_ids[0]}.npz"


def write_shard(path: pathlib.Path, item_ids: Sequence[str], figure_ids: Sequence[str],
                acts: np.ndarray, layers: Sequence[int], positions: Sequence[str]) -> None:
    """`acts` is (n_items, n_layers, n_positions, hidden) float16."""
    if acts.shape[:3] != (len(item_ids), len(layers), len(positions)):
        raise ValueError(f"shard acts shape {acts.shape} does not match "
                         f"{len(item_ids)} items x {len(layers)} layers x {len(positions)} positions")
    if acts.dtype != np.float16:
        raise ValueError(f"shard acts are {acts.dtype}, expected float16")
    _atomic_savez(path, {
        "item_ids": np.asarray(item_ids, dtype=str),
        "figure_ids": np.asarray(figure_ids, dtype=str),
        "layers": np.asarray(layers, dtype=np.int64),
        "positions": np.asarray(positions, dtype=str),
        "acts": acts,
    }, compress=False)


def read_shard(path: pathlib.Path) -> dict:
    with np.load(path) as z:
        return {"item_ids": [str(x) for x in z["item_ids"]],
                "figure_ids": [str(x) for x in z["figure_ids"]],
                "layers": [int(x) for x in z["layers"]],
                "positions": [str(x) for x in z["positions"]],
                "acts": z["acts"]}


def shard_paths(shard_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(shard_dir.glob("shard_*.npz")) if shard_dir.is_dir() else []


def cached_item_ids(shard_dir: pathlib.Path) -> set[str]:
    """Every item_id with persisted activations. Reads only the id arrays."""
    ids: set[str] = set()
    for path in shard_paths(shard_dir):
        with np.load(path) as z:
            ids.update(str(x) for x in z["item_ids"])
    return ids


# ---------------------------------------------------------------------------
# Final cache
# ---------------------------------------------------------------------------

def cache_key(layer: int, position: str) -> str:
    return f"L{layer}_{position}"


def assemble(shard_dir: pathlib.Path, order: Sequence[str], layers: Sequence[int],
             positions: Sequence[str], out: pathlib.Path) -> None:
    """Merge every shard into one compressed .npz, rows in `order`.

    `order` is the dataset order of the items, so the row order is fixed by the
    data rather than by which shard happened to finish first.
    """
    row_of: dict[str, tuple[int, int]] = {}
    shards = []
    for path in shard_paths(shard_dir):
        s = read_shard(path)
        if s["layers"] != list(layers) or s["positions"] != list(positions):
            raise ValueError(f"{path.name} holds layers {s['layers']} and positions "
                             f"{s['positions']}, the run expects {list(layers)} and {list(positions)}")
        for r, iid in enumerate(s["item_ids"]):
            if iid in row_of:
                raise ValueError(f"item {iid} appears in more than one shard")
            row_of[iid] = (len(shards), r)
        shards.append(s)

    if set(row_of) != set(order) or len(order) != len(set(order)):
        missing = len(set(order) - set(row_of))
        extra = len(set(row_of) - set(order))
        raise ValueError(f"shards do not cover the items exactly: {missing} missing, "
                         f"{extra} not in the dataset")

    rows = [row_of[iid] for iid in order]
    acts = np.stack([shards[s]["acts"][r] for s, r in rows])     # (N, L, P, hidden)
    figure_ids = [shards[s]["figure_ids"][r] for s, r in rows]

    arrays = {"item_ids": np.asarray(order, dtype="<U16"),
              "figure_ids": np.asarray(figure_ids, dtype=object)}
    for j, layer in enumerate(layers):
        for k, pos in enumerate(positions):
            arrays[cache_key(layer, pos)] = np.ascontiguousarray(acts[:, j, k])
    _atomic_savez(out, arrays, compress=True)


def check_predictions(rows: Mapping[str, dict], model_key: str, expected: int) -> list[str]:
    """Acceptance A. Returns problems; empty means it passed."""
    problems = []
    if len(rows) != expected:
        problems.append(f"{len(rows)} prediction rows, expected {expected}")
    for iid, row in rows.items():
        absent = [f for f in PREDICTION_FIELDS if f not in row]
        if absent:
            problems.append(f"row {iid} lacks {absent}")
            continue
        if row["model"] != model_key:
            problems.append(f"row {iid} has model {row['model']!r}, expected {model_key!r}")
        if item_id(model_key, row["figure_id"], row["question"], row.get("source")) != iid:
            problems.append(f"row {iid} does not hash to its own item_id")
    return problems


def check_cache(npz_path: pathlib.Path, rows: Mapping[str, dict], layers: Sequence[int],
                positions: Sequence[str], hidden: int) -> list[str]:
    """Acceptance B. Returns problems; empty means it passed."""
    problems = []
    with np.load(npz_path, allow_pickle=True) as z:
        ids = [str(x) for x in z["item_ids"]]
        figs = [str(x) for x in z["figure_ids"]]
        n = len(ids)
        if len(set(ids)) != n:
            problems.append("duplicate item_ids in the cache")
        if set(ids) != set(rows):
            problems.append(f"cache and predictions hold different items: "
                            f"{len(set(ids) - set(rows))} only in the cache, "
                            f"{len(set(rows) - set(ids))} only in predictions")
        if len(figs) != n:
            problems.append(f"{len(figs)} figure_ids for {n} item_ids")
        elif any(rows[i]["figure_id"] != f for i, f in zip(ids, figs) if i in rows):
            problems.append("figure_ids are not aligned with item_ids")

        expected_keys = {cache_key(L, p) for L in layers for p in positions}
        present = {k for k in z.files if k not in ("item_ids", "figure_ids")}
        if present != expected_keys:
            problems.append(f"activation keys differ from the run: {len(expected_keys - present)} "
                            f"missing, {len(present - expected_keys)} unexpected")
        for key in sorted(expected_keys & present):
            arr = z[key]
            if arr.shape != (n, hidden) or arr.dtype != np.float16:
                problems.append(f"{key} is {arr.dtype}{arr.shape}, expected float16{(n, hidden)}")
            elif not np.isfinite(arr).all():
                problems.append(f"{key} holds non-finite values")
    return problems
