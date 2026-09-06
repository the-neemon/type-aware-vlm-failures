"""Sample size against detectable effect, for results/e4-preregistration.md.

Answers how many items per cell E4 needs to detect a given difference in
differences. Slow, a few minutes. Run from the repo root:

    PYTHONPATH=. python3 scripts/e4_power_sweep.py
"""
import numpy as np
from src.intervene.matrix import Outcome, did_with_ci

def trial(n_per_cell, true_delta, per_fig=3, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for ftype, base in (("structural", 0.25), ("fabrication", 0.10)):
        n_fig = max(1, n_per_cell // per_fig)
        lift = true_delta if ftype == "structural" else 0.0
        for f in range(n_fig):
            fid = f"{ftype}_{f}"
            eff = rng.normal(scale=0.08)
            for _ in range(per_fig):
                p0 = np.clip(base + eff, 0.01, 0.99)
                p1 = np.clip(base + eff + 0.15 + lift, 0.01, 0.99)
                out.append(Outcome(fid, ftype, "I_0", rng.random() < p0))
                out.append(Outcome(fid, ftype, "I_crop", rng.random() < p1))
    return did_with_ci(out, "I_crop", n_boot=250, seed=seed)

print("power to detect a true difference in differences\n")
print("  n/cell   delta=0.05   delta=0.10   delta=0.15")
for n in (400, 800, 1600, 3200):
    row = []
    for d in (0.05, 0.10, 0.15):
        res = [trial(n, d, seed=s) for s in range(150)]
        row.append(np.mean([r["excludes_zero"] and r["delta"] > 0 for r in res]))
    print(f"  {n:>5}      {row[0]:.2f}         {row[1]:.2f}         {row[2]:.2f}")
