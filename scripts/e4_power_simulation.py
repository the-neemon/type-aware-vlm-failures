"""Power simulation behind results/e4-preregistration.md (P6.7).

How often does the pre-registered difference-in-differences detect a true
effect, at 400 items per cell? Run from the repo root:

    PYTHONPATH=. python3 scripts/e4_power_simulation.py
"""
import numpy as np
from src.intervene.matrix import Outcome, did_with_ci

def trial(n_per_cell, true_delta, base_struct=0.25, base_fab=0.10,
          per_fig=3, seed=0):
    """Simulate one E4 run and ask whether the CI excludes zero."""
    rng = np.random.default_rng(seed)
    out = []
    for ftype, base in ((("structural"), base_struct), (("fabrication"), base_fab)):
        n_fig = max(1, n_per_cell // per_fig)
        # crop helps structural by (base + true_delta), fabrication by base
        lift = true_delta if ftype == "structural" else 0.0
        for f in range(n_fig):
            fid = f"{ftype}_{f}"
            # figure-level difficulty, so outcomes cluster within figure
            eff = rng.normal(scale=0.08)
            for _ in range(per_fig):
                p0 = np.clip(base + eff, 0.01, 0.99)
                p1 = np.clip(base + eff + 0.15 + lift, 0.01, 0.99)
                out.append(Outcome(fid, ftype, "I_0", rng.random() < p0))
                out.append(Outcome(fid, ftype, "I_crop", rng.random() < p1))
    return did_with_ci(out, "I_crop", n_boot=300, seed=seed)

print("n per cell = 400, 3 errors per figure, 200 simulated runs\n")
print(" true delta   power (CI excludes 0)   mean estimate")
for d in (0.0, 0.05, 0.10, 0.15, 0.20):
    res = [trial(400, d, seed=s) for s in range(200)]
    power = np.mean([r["excludes_zero"] and r["delta"] > 0 for r in res])
    print(f"   {d:.2f}          {power:.2f}                  {np.mean([r['delta'] for r in res]):+.3f}")
