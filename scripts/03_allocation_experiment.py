"""Solver comparison on identical cohorts (fixes #11, #12, #14, #15, #18).

For every cohort ALL solvers receive the same Instance (applicants, scores, budget,
covariance, default scenarios). Reported per solver: mean and sample SD across cohorts.

    python scripts/03_allocation_experiment.py [--cohorts 20] [--n 8] [--jobs 4]
"""
import argparse
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qfar.config import load_config, write_manifest  # noqa: E402
from qfar.data import sample_cohorts  # noqa: E402
from qfar.experiment import build_instance, cohort_risk, evaluate, timed  # noqa: E402
from qfar.metrics import mean_sd  # noqa: E402
from qfar.quantum import StateVec, interp_init, solve_qaoa, solve_vqe  # noqa: E402
from qfar.solvers import (solve_exact, solve_exact_qubo, solve_greedy, solve_milp,  # noqa: E402
                          solve_sa)


def run_cohort(args):
    c, coh, cfg = args
    seed = cfg["seed"] + 1000 * c
    inst = build_instance(coh, cfg, cohort_risk(coh, cfg, scenario_seed=seed))
    q, tail, sv_cfg = inst.q, inst.tail, cfg["solvers"]
    ref1 = timed(solve_exact, q); ref2 = timed(solve_exact, q, tail)
    ground = solve_exact_qubo(q)
    assert ground["raw_feasible"] and abs(ground["J1"] - ref1["J1"]) < 1e-9, "QUBO encoding check failed"

    runs = [ref1, ref2,
            timed(solve_milp, q),                                         # P1, same objective as exact
            timed(solve_milp, q, tail),                                   # P2 (CVaR-aware)
            timed(solve_milp, q, fair_eps=cfg["allocation"]["hard_fairness_eps"], utility_only=True),
            timed(solve_greedy, q),
            timed(solve_sa, q, sv_cfg["sa"]["steps"], sv_cfg["sa"]["restarts"], seed)]
    ref2["solver"] = "exact_enumeration_P2"; ref1["solver"] = "exact_enumeration_P1"
    runs[3]["solver"] = "milp_linearised_cvar"

    sv = StateVec(q); qa = sv_cfg["qaoa"]; init = None
    for p in qa["depths"]:
        r = timed(solve_qaoa, q, p=p, restarts=qa["restarts"], max_iter=qa["max_iter"],
                  shots=qa["shots"], seed=seed + p, init=init, sv=sv)
        init = interp_init(r["params"], p); runs.append(r)
    cq = sv_cfg["cvar_qaoa"]
    for p in cq["depths"]:
        runs.append(timed(solve_qaoa, q, p=p, alpha_q=cq["alpha_q"], restarts=cq["restarts"],
                          max_iter=cq["max_iter"], shots=cq["shots"], seed=seed + 10 + p, sv=sv))
    vq = sv_cfg["vqe"]
    runs.append(timed(solve_vqe, q, reps=vq["reps"], restarts=vq["restarts"], max_iter=vq["max_iter"],
                      shots=vq["shots"], seed=seed + 20, sv=sv))
    rows = []
    for r in runs:
        row = evaluate(inst, r, ref1, ref2); row.update(cohort=c, n_A=q.n_A, n_B=q.n_B); rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--cohorts", type=int, default=None)
    ap.add_argument("--n", type=int, default=None, help="applicants per cohort")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.n: cfg["allocation"]["cohort_size"] = a.n
    K = a.cohorts or cfg["experiments"]["allocation_cohorts"]
    OUT = ROOT / "results" / "03_allocation"; OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(ROOT / "data" / "processed" / "test_scored.csv")
    al = cfg["allocation"]
    cohorts = sample_cohorts(test, al["cohort_size"], K, cfg["seed"] + 3, al["min_per_group"])
    pd.concat([c[["applicant_id"]].assign(cohort=i) for i, c in enumerate(cohorts)]).to_csv(OUT / "cohorts.csv", index=False)

    t0 = time.time(); rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, rr in enumerate(ex.map(run_cohort, [(c, coh, cfg) for c, coh in enumerate(cohorts)])):
            rows += rr; print(f"cohort {i} done ({time.time()-t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT / "allocation_raw.csv", index=False)

    order = list(dict.fromkeys(d.solver))
    main_cols = ["J1", "J2", "gap_J1", "hit_opt_P1", "hit_opt_P2", "raw_feasible", "repaired", "approved",
                 "utility", "budget_utilisation", "expected_return_inr", "expected_loss_inr", "loss_std_inr",
                 "cvar_inr", "realised_loss_inr", "dpd", "abs_dpd", "dir_A_over_B", "dir_B_over_A", "runtime_s"]
    s = mean_sd(d.astype({"hit_opt_P1": float, "hit_opt_P2": float, "raw_feasible": float, "repaired": float}),
                "solver", main_cols).loc[order]
    s.to_csv(OUT / "allocation_summary.csv")
    qcols = ["qubits", "depth", "two_qubit_gates", "shots", "nfev_total", "P_opt_exact", "P_opt_shots",
             "P_opt_uniform", "P_feasible_exact", "P_feasible_shots", "P_feasible_uniform", "hit_opt_P1",
             "gap_J1", "most_frequent_feasible", "most_frequent_J1", "J1"]
    dq = d[d.P_opt_exact.notna()].astype({"hit_opt_P1": float, "most_frequent_feasible": float})
    qs = mean_sd(dq, "solver", qcols); qs.to_csv(OUT / "quantum_summary.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    show = ["J1", "J2", "hit_opt_P1", "hit_opt_P2", "approved", "budget_utilisation", "expected_loss_inr",
            "cvar_inr", "abs_dpd", "dir_A_over_B", "runtime_s"]
    print("\nMEAN across cohorts:\n", s[show].round(3).to_string())
    print("\nQUANTUM (statevector simulation):\n", qs[[c for c in qcols]].round(4).to_string())
    write_manifest(OUT, cfg, script="03_allocation_experiment.py", n_cohorts=K,
                   cohort_seed=cfg["seed"] + 3, scenario_seed_rule="seed + 1000*cohort",
                   qubo_example=build_instance(cohorts[0], cfg, cohort_risk(cohorts[0], cfg, cfg["seed"])).q.summary())
