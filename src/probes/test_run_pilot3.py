"""run_pilot3 on generated pilots with a planted missing-bar direction."""

import json
import sys

import numpy as np

from src.probes import run_pilot3 as rp
from src.synth.absent_pairs import generate as gen_pilot1
from src.synth.items import load_manifest_items
from src.synth.out_of_range import generate as gen_pilot3

HIDDEN, LAYERS = 8, (0, 1)
SPLITS = ("train", "validation", "test")


def _pilot(tmp, name, generate, n_charts, seed):
    d = tmp / name
    generate(d, num_charts=n_charts, seed=seed)
    manifest = d / "manifest.jsonl"
    rng = np.random.default_rng(seed)
    absent_answers = ["37", "0", "None"]
    preds, ids, figs = [], [], []
    for n, it in enumerate(load_manifest_items(manifest, SPLITS)):
        if it["absent"]:
            pred = absent_answers[n % 3]
        else:
            pred = it["gold"] if rng.random() < 0.8 else "999"
        iid = f"{name}_{n}"
        preds.append({"item_id": iid, "model": "m", "figure_id": it["figure_id"],
                      "question": it["question"], "gold": it["gold"], "prediction": pred,
                      "correct": False, "split": it["split"], "source": "synthetic",
                      **{k: it[k] for k in ("template", "absent", "phrasing", "asks_about")}})
        ids.append(iid)
        figs.append(it["figure_id"])
    acts = {}
    for L in LAYERS:
        A = rng.normal(size=(len(preds), HIDDEN))
        if L == 1:
            A[:, 0] += 3.0 * np.array([p["absent"] for p in preds])
        acts[f"L{L}_query_last"] = A.astype(np.float16)
    np.savez_compressed(d / "a.npz", item_ids=np.array(ids),
                        figure_ids=np.array(figs, dtype=object), **acts)
    (d / "p.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    return [str(d / "a.npz"), str(d / "p.jsonl"), str(manifest)]


def test_missing_bar_probe_transfers_to_out_of_range_years(tmp_path, monkeypatch):
    p1 = _pilot(tmp_path, "p1", gen_pilot1, 60, 1)
    p3 = _pilot(tmp_path, "p3", gen_pilot3, 60, 3)
    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["run_pilot3", "--pilot", "pilot1", *p1,
                                      "--target", *p3, "--l2", "1", "100", "--jobs", "2",
                                      "--out", str(out)])
    rp.main()
    r = json.loads(out.read_text())
    ms = r["probes"]["missing_vs_shown"]
    assert ms["selected_on_pilots_1_2"]["layer"] == 1
    assert ms["pilot3"]["absent_vs_shown"]["auroc"] > 0.9
    assert abs(ms["pilot3"]["S_vs_C"]["auroc"] - 0.5) < 0.25      # misreads are shown bars
    assert r["probes"]["fabrication"]["pilot3"]["F_vs_C"]["auroc"] > 0.9
    c = r["outcomes"]["counts"]
    assert c["R"] and c["Z"] and c["F"] and c["C"]
    assert sum(c.values()) == 120
