"""Weight sensitivity over the simplex  w_F + w_N + w_S = 1  (fix #16).

For every cohort and every grid point the EXACT optimum of P1 is computed (same
cohorts, budget, eta, lambda_F at every grid point), and compared with the
declared baseline (equal weights) by Jaccard similarity of the approved sets and
Kendall's tau of the utility rankings.

    python scripts/02_weight_ablation.py [--cohorts 30]
"""
import argparse
import itertools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from qfar.config import load_config, write_manifest  # noqa: E402
from qfar.data import sample_cohorts  # noqa: E402
from qfar.experiment import build_instance, cohort_risk, evaluate  # noqa: E402
from qfar.metrics import jaccard, mean_sd  # noqa: E402
from qfar.solvers import solve_exact  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None); ap.add_argument("--cohorts", type=int, default=None)
a = ap.parse_args()
cfg = load_config(a.config)
K = a.cohorts or cfg["experiments"]["weight_cohorts"]
step = cfg["experiments"]["weight_grid_step"]; m = int(round(1 / step))
grid = [(i / m, j / m, round(1 - i / m - j / m, 10)) for i, j in itertools.product(range(m + 1), repeat=2) if i + j <= m]
bw = cfg["scores"]["weights"]; base = (bw["F"], bw["N"], bw["S"])
OUT = ROOT / "results" / "02_weight_sensitivity"; OUT.mkdir(parents=True, exist_ok=True)
test = pd.read_csv(ROOT / "data" / "processed" / "test_scored.csv")
al = cfg["allocation"]
cohorts = sample_cohorts(test, al["cohort_size"], K, cfg["seed"] + 2, al["min_per_group"])

rows = []
for c, coh in enumerate(cohorts):
    risk = cohort_risk(coh, cfg, scenario_seed=cfg["seed"] + 1000 * c)
    i0 = build_instance(coh, cfg, risk, base); x0 = solve_exact(i0.q)["x"]
    for w in grid:
        inst = build_instance(coh, cfg, risk, w)
        r = solve_exact(inst.q)
        row = evaluate(inst, r)
        row.update(cohort=c, wF=w[0], wN=w[1], wS=w[2], jaccard_vs_baseline=jaccard(r["x"], x0),
                   kendall_tau_vs_baseline=kendalltau(inst.scores.U, i0.scores.U)[0])
        rows.append(row)
    print(f"cohort {c} done", flush=True)

d = pd.DataFrame(rows); d.to_csv(OUT / "weight_raw.csv", index=False)
cols = ["utility", "contrib_F", "contrib_N", "contrib_S", "approved", "budget_utilisation", "abs_dpd",
        "dpd", "dir_A_over_B", "dir_B_over_A", "expected_return_inr", "expected_loss_inr", "cvar_inr",
        "realised_loss_inr", "jaccard_vs_baseline", "kendall_tau_vs_baseline"]
s = mean_sd(d, ["wF", "wN", "wS"], cols).reset_index(); s.to_csv(OUT / "weight_summary.csv", index=False)
corner = s[(s[["wF", "wN", "wS"]].max(axis=1) == 1.0) | (np.isclose(s.wF, .3) & np.isclose(s.wN, .3))]
pd.set_option("display.width", 250)
show = ["wF", "wN", "wS", "approved", "contrib_F", "contrib_N", "contrib_S", "abs_dpd", "expected_loss_inr",
        "cvar_inr", "jaccard_vs_baseline", "kendall_tau_vs_baseline"]
print(f"\n{K} cohorts x {len(grid)} weight vectors. Corners and (0.3,0.3,0.4):")
print(corner[show].round(3).to_string(index=False))
print("\nMean Jaccard vs baseline over the whole simplex:", round(d.jaccard_vs_baseline.mean(), 3))
write_manifest(OUT, cfg, script="02_weight_ablation.py", n_cohorts=K, grid_points=len(grid),
               baseline_weights=base, cohort_seed=cfg["seed"] + 2)
