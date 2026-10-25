import json

import numpy as np
import pytest

from src.controller import e5


def test_curve_averages_ties_and_ranks_by_confidence():
    ok = np.array([True, True, False, False])
    flat = e5.curve(ok, np.zeros(4))
    assert np.allclose(flat, 0.5)                      # one tie: expected risk everywhere
    perfect = e5.curve(ok, np.array([2.0, 2.0, 1.0, 1.0]))
    assert np.allclose(perfect, [0, 0, 1 / 3, 0.5])
    assert e5.aurc(perfect) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    assert e5.risk_at(perfect, 0.5) == 0 and e5.risk_at(perfect, 1.0) == 0.5


def _items(n=200, seed=0, crop_breaks_right=False):
    """Half shown bars (one in seven misread, spread over folds), half missing bars.

    The misread probe is planted on S."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        absent = i % 2 == 1
        cls = ("F" if i % 4 == 1 else "Z") if absent else ("S" if i % 14 == 0 else "C")
        right = cls == "C"
        ok = {"I_0": right, "I_reask": right, "I_upsample": right, "I_verify": right,
              "I_crop": (cls == "S") or (right and not crop_breaks_right)}
        items.append({"item_id": f"i{i}", "figure_id": f"f{i // 2}", "fold": (i // 2) % 5,
                      "cls": cls, "ok": ok,
                      "misread": float(cls == "S") + rng.normal(0, 0.05),
                      "missing": float(absent) + rng.normal(0, 0.05),
                      "binary": float(not right) + rng.normal(0, 0.05)})
    return items


def test_relook_with_crop_fixes_misreads_when_crop_is_safe():
    items = _items()
    final, chosen, relooked = e5.crossfit_relook(items)
    assert all(v["repair"] == "I_crop" for v in chosen.values())
    s = np.array([it["cls"] == "S" for it in items])
    assert final[s].all() and relooked[s].all()


def test_relook_declines_a_repair_that_breaks_more_than_it_fixes():
    # misread probe fires on nothing useful: scores are noise, crop breaks right answers
    items = _items(crop_breaks_right=True)
    for it in items:
        it["misread"] = float(np.random.default_rng(len(it["item_id"])).normal())
    final, chosen, _ = e5.crossfit_relook(items)
    ok0 = np.array([it["ok"]["I_0"] for it in items])
    assert final.mean() >= ok0.mean() - 0.02


def test_oracle_and_policies_order():
    items = _items()
    final, conf, plan = e5.oracle(items)
    assert plan["S"]["action"] == "I_crop" and plan["C"]["action"] == "I_0"
    assert conf[[it["cls"] == "F" for it in items]].max() < conf[[it["cls"] == "C" for it in items]].min()
    r = e5.evaluate(items, n_boot=50)
    a = {p: r["policies"][p]["aurc"] for p in e5.POLICIES}
    assert a["oracle"] <= a["controller"] <= a["plain"]
    assert a["controller"] < a["missing"]               # the re-look fixes the misreads
    assert "controller" in e5.report({"m": r})
    assert r["relook_by_class"]["S"]["fixed"] == r["relook_by_class"]["S"]["n"]


def test_load_refuses_questions_without_outcomes(tmp_path):
    (tmp_path / "e5").mkdir()
    (tmp_path / "e4").mkdir()
    score = {"item_id": "a", "figure_id": "f", "fold": 0, "cls": "C",
             "binary": 0.1, "missing": 0.1, "misread": 0.1}
    (tmp_path / "e5" / "m_scores.jsonl").write_text(json.dumps(score) + "\n")
    rows = [{"item_id": "a", "intervention": iv, "recovered": True, "applicable": True}
            for iv in ("I_0", "I_reask", "I_upsample", "I_crop")]          # no I_verify
    (tmp_path / "e4" / "m_synth_pilot.correct.outcomes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="lack"):
        e5.load("m", tmp_path)
    with open(tmp_path / "e4" / "m_synth_pilot.correct.outcomes.jsonl", "a") as f:
        f.write(json.dumps({"item_id": "a", "intervention": "I_verify", "recovered": True}) + "\n")
    assert e5.load("m", tmp_path)[0]["ok"]["I_verify"] is True
