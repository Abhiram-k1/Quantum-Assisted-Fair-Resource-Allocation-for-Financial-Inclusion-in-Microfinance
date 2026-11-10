"""Component ablation (fix #17).

(a) Score components E1..E7: which of F, N, S enter U (included components get
    equal weight). Risk and fairness penalties at their baseline values.
(b) Penalty components at equal score weights: utility only, +variance,
    +fairness, +both (P1), +both +financial CVaR for several lambda_R (P2).

Cohorts, budgets and default scenarios are identical across all settings, and
every setting is solved EXACTLY (enumeration), so differences are caused only by
the objective components.

    python scripts/04_component_ablation.py [--cohorts 30]
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from qfar.config import load_config, write_manifest  # noqa: E402
from qfar.data import sample_cohorts  # noqa: E402
from qfar.experiment import build_instance, cohort_risk, evaluate  # noqa: E402
from qfar.metrics import jaccard, mean_sd  # noqa: E402
from qfar.solvers import solve_exact  # noqa: E402

COMPONENTS = {"E1_F": (1, 0, 0), "E2_N": (0, 1, 0), "E3_S": (0, 0, 1), "E4_F+N": (1, 1, 0),
              "E5_F+S": (1, 0, 1), "E6_N+S": (0, 1, 1), "E7_F+N+S": (1, 1, 1)}

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None); ap.add_argument("--cohorts", type=int, default=None)
a = ap.parse_args()
cfg = load_config(a.config); al = cfg["allocation"]
K = a.cohorts or cfg["experiments"]["ablation_cohorts"]
eta0, lf0, lr0 = al["eta"], al["lambda_F"], al["lambda_R"]
PENALTIES = {"P0_utility_only": (0, 0, None), "P1a_+variance": (eta0, 0, None),
             "P1b_+fairness": (0, lf0, None), "P1_variance+fairness": (eta0, lf0, None),
             "P1_fairness_x5": (eta0, 5 * lf0, None),
             **{f"P2_+CVaR_lamR={lr}": (eta0, lf0, lr) for lr in sorted({lr0, 1.0, 2.0, 5.0})}}
OUT = ROOT / "results" / "04_component_ablation"; OUT.mkdir(parents=True, exist_ok=True)
test = pd.read_csv(ROOT / "data" / "processed" / "test_scored.csv")
cohorts = sample_cohorts(test, al["cohort_size"], K, cfg["seed"] + 4, al["min_per_group"])

rows = []
for c, coh in enumerate(cohorts):
    risk = cohort_risk(coh, cfg, scenario_seed=cfg["seed"] + 1000 * c)
    base = solve_exact(build_instance(coh, cfg, risk, (1/3, 1/3, 1/3)).q)["x"]
    for name, inc in COMPONENTS.items():
        w = tuple(v / sum(inc) for v in inc)
        inst = build_instance(coh, cfg, risk, w); r = solve_exact(inst.q)
        row = evaluate(inst, r); row.update(study="score_components", setting=name, cohort=c,
                                            jaccard_vs_E7=jaccard(r["x"], base)); rows.append(row)
    for name, (eta, lf, lr) in PENALTIES.items():
        inst = build_instance(coh, cfg, risk, (1/3, 1/3, 1/3), eta=eta, lam_F=lf, lam_R=lr or 0.0)
        r = solve_exact(inst.q, inst.tail if lr else None)
        row = evaluate(inst, r); row.update(study="penalties", setting=name, cohort=c,
                                            jaccard_vs_E7=jaccard(r["x"], base)); rows.append(row)
    print(f"cohort {c} done", flush=True)

d = pd.DataFrame(rows); d.to_csv(OUT / "ablation_raw.csv", index=False)
cols = ["utility", "contrib_F", "contrib_N", "contrib_S", "approved", "budget_utilisation", "AR_A", "AR_B",
        "abs_dpd", "dir_A_over_B", "expected_return_inr", "expected_loss_inr", "loss_std_inr", "cvar_inr",
        "realised_loss_inr", "jaccard_vs_E7"]
pd.set_option("display.width", 250)
show = ["utility", "approved", "AR_A", "AR_B", "abs_dpd", "expected_loss_inr", "loss_std_inr", "cvar_inr",
        "realised_loss_inr", "jaccard_vs_E7"]
for study in ["score_components", "penalties"]:
    dd = d[d.study == study]
    s = mean_sd(dd, "setting", cols).loc[list(dict.fromkeys(dd.setting))]
    s.to_csv(OUT / f"{study}_summary.csv")
    print(f"\n== {study} (mean over {K} cohorts) ==\n", s[show].round(3).to_string())
write_manifest(OUT, cfg, script="04_component_ablation.py", n_cohorts=K, cohort_seed=cfg["seed"] + 4,
               penalties={k: dict(eta=v[0], lambda_F=v[1], lambda_R=v[2]) for k, v in PENALTIES.items()})
