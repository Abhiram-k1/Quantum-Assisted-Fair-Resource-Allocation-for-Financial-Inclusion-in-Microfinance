"""Scaling benchmark (N in {4, 6, 8, 10}: Standard QAOA vs Warm-Start QAOA vs SA vs Exact/MILP)
and Qiskit noisy + Zero-Noise Extrapolation study (N in {4, 6, 8}, scales c in {1, 3, 5}).

Usage:
    python scripts/06_scaling_and_noise.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qfar.config import load_config, write_manifest
from qfar.data import sample_cohorts
from qfar.experiment import build_instance, cohort_risk, evaluate, timed
from qfar.metrics import mean_sd
from qfar.pipeline import ensure_pd_predictions
from qfar.qiskit_backend import run_noisy_and_zne
from qfar.quantum import StateVec, solve_qaoa, solve_warm_start_qaoa
from qfar.solvers import solve_exact, solve_milp, solve_sa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--cohorts", type=int, default=None, help="Override experiments.scaling_cohorts")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiments"]
    k = int(args.cohorts or exp.get("scaling_cohorts", 10))
    sizes = [int(n) for n in exp.get("scaling_sizes", [4, 6, 8, 10])]
    noise_sizes = set(int(n) for n in exp.get("noise_sizes", [4, 6, 8]))

    out = ROOT / "results/05_scaling_and_noise"
    out.mkdir(parents=True, exist_ok=True)

    test_scored = ROOT / "data/processed/test_scored.csv"
    if test_scored.exists():
        test = pd.read_csv(test_scored)
    else:
        from qfar.data import load_pool, split_pool
        _, _, test_raw = split_pool(load_pool(cfg), cfg)
        test, _ = ensure_pd_predictions(test_raw, cfg)

    sol = cfg["solvers"]
    ws_cfg = sol.get("warm_start_qaoa", sol["qaoa"])
    ncfg = cfg.get("noise", {})
    scales = tuple(int(c) for c in ncfg.get("zne_scales", [1, 3, 5]))
    p1_dep = float(ncfg.get("p1_depol", 0.002))
    p2_dep = float(ncfg.get("p2_depol", 0.01))
    p_ro = float(ncfg.get("p_readout", 0.02))
    n_traj = int(ncfg.get("n_trajectories", 256))

    scaling_rows: list[dict] = []
    noise_rows: list[dict] = []

    for N in sizes:
        min_g = 1 if N <= 4 else int(cfg["allocation"]["min_per_group"])
        cohorts = sample_cohorts(test, n=N, k=k, seed=int(cfg["seed"]) + 500 + N, min_per_group=min_g)
        for idx, coh in enumerate(cohorts):
            seed = int(cfg["seed"]) + N * 100 + idx
            risk = cohort_risk(coh, cfg, scenario_seed=seed + 2000)
            inst = build_instance(coh, cfg, risk)
            q = inst.q

            r_p1 = timed(solve_exact, q)
            r_milp = timed(solve_milp, q, inst.tail)
            r_sa = timed(solve_sa, q, **sol["sa"], seed=seed)
            results = [r_p1, r_milp, r_sa]

            q_std_p1 = None
            q_ws_p1 = None
            if q.n <= 16:
                sv = StateVec(q)
                restarts = int(sol["qaoa"]["restarts"]) if N <= 8 else min(3, int(sol["qaoa"]["restarts"]))
                max_it = int(sol["qaoa"]["max_iter"]) if N <= 8 else min(180, int(sol["qaoa"]["max_iter"]))
                q_std_p1 = timed(solve_qaoa, q, p=1, alpha_q=1.0, restarts=restarts,
                                 max_iter=max_it, shots=int(sol["qaoa"]["shots"]), seed=seed, sv=sv)
                q_std_p2 = timed(solve_qaoa, q, p=2, alpha_q=1.0, restarts=restarts,
                                 max_iter=max_it, shots=int(sol["qaoa"]["shots"]), seed=seed, sv=sv)
                q_ws_p1 = timed(solve_warm_start_qaoa, q, p=1, alpha_q=1.0, eps=float(ws_cfg.get("eps", 0.05)),
                                restarts=restarts, max_iter=max_it, shots=int(ws_cfg.get("shots", 1024)),
                                seed=seed, sv=sv)
                q_ws_p2 = timed(solve_warm_start_qaoa, q, p=2, alpha_q=1.0, eps=float(ws_cfg.get("eps", 0.05)),
                                restarts=restarts, max_iter=max_it, shots=int(ws_cfg.get("shots", 1024)),
                                seed=seed, sv=sv)
                results.extend([q_std_p1, q_std_p2, q_ws_p1, q_ws_p2])

            for r in results:
                row = evaluate(inst, r, ref_p1=r_p1, ref_p2=r_milp)
                row["N"] = N
                row["cohort"] = idx
                scaling_rows.append(row)

            if N in noise_sizes and q_std_p1 is not None and q_ws_p1 is not None:
                for label, qres, use_ws in (("qaoa_p1", q_std_p1, False), ("warm_start_qaoa_p1", q_ws_p1, True)):
                    thetas = 2.0 * np.arcsin(np.sqrt(qres["c_star"])) if use_ws else None
                    nz = run_noisy_and_zne(
                        q, qres["params"], p=1, scales=scales, p1_depol=p1_dep, p2_depol=p2_dep,
                        p_readout=p_ro, shots=int(ncfg.get("shots", 1024)), seed=seed,
                        warm_start_thetas=thetas, n_trajectories=n_traj,
                    )
                    ideal_e = float(qres["expectation"])
                    e_c1 = float(nz["per_scale"][0]["expectation"])
                    e_c3 = float(nz["per_scale"][1]["expectation"]) if len(nz["per_scale"]) > 1 else np.nan
                    e_c5 = float(nz["per_scale"][2]["expectation"]) if len(nz["per_scale"]) > 2 else np.nan
                    e_zne = float(nz["zne"]["zne_energy"])
                    err_c1 = abs(e_c1 - ideal_e)
                    err_zne = abs(e_zne - ideal_e)
                    noise_rows.append(dict(
                        N=N, qubits=q.n, cohort=idx, solver=label,
                        ideal_expectation=ideal_e,
                        noisy_expectation_c1=e_c1,
                        noisy_expectation_c3=e_c3,
                        noisy_expectation_c5=e_c5,
                        zne_expectation=e_zne,
                        err_c1=err_c1,
                        err_zne=err_zne,
                        zne_rel_error_reduction=((err_c1 - err_zne) / err_c1) if err_c1 > 1e-12 else 0.0,
                        p_feasible_ideal=float(qres["p_feasible_exact"]),
                        p_feasible_noisy_c1=float(nz["per_scale"][0]["p_feasible_exact"]),
                        noisy_c1_J1=float(nz["per_scale"][0]["J1"]),
                        noisy_c1_gap_J1=(float(nz["per_scale"][0]["J1"]) - r_p1["J1"]) / max(abs(r_p1["J1"]), 1e-9),
                        noisy_c1_feasible=bool(nz["per_scale"][0]["feasible"]),
                        method=nz["per_scale"][0]["simulation_method"],
                    ))
        print(f"Completed N={N} ({len(cohorts)} cohorts)", flush=True)

    df_sc = pd.DataFrame(scaling_rows)
    df_sc.to_csv(out / "scaling_per_cohort.csv", index=False)

    sc_cols = [c for c in ("qubits", "J1", "gap_J1", "hit_opt_P1", "P_opt_exact",
                           "P_feasible_exact", "abs_dpd", "cvar_inr", "runtime_s") if c in df_sc.columns]
    df_sc_sum = mean_sd(df_sc, ["N", "solver"], sc_cols)
    df_sc_sum.to_csv(out / "scaling_summary.csv")

    df_nz = pd.DataFrame(noise_rows)
    if len(df_nz):
        df_nz.to_csv(out / "noisy_zne_per_cohort.csv", index=False)
        nz_cols = ["qubits", "ideal_expectation", "noisy_expectation_c1", "noisy_expectation_c3",
                   "noisy_expectation_c5", "zne_expectation", "err_c1", "err_zne",
                   "zne_rel_error_reduction", "p_feasible_ideal", "p_feasible_noisy_c1", "noisy_c1_gap_J1"]
        df_nz_sum = mean_sd(df_nz, ["N", "solver"], nz_cols)
        df_nz_sum.to_csv(out / "noisy_zne_summary.csv")
    else:
        df_nz_sum = pd.DataFrame()

    write_manifest(out, cfg, scaling_cohorts=k, scaling_sizes=sizes, noise_sizes=sorted(noise_sizes),
                   n_scaling_rows=len(df_sc), n_noise_rows=len(df_nz))

    print("\n=== Scaling Summary (mean across cohorts) ===")
    print(df_sc.groupby(["N", "solver"])[["qubits", "gap_J1", "hit_opt_P1", "P_opt_exact", "P_feasible_exact", "runtime_s"]].mean().round(4))
    if len(df_nz):
        print("\n=== Noisy + ZNE Summary (mean across cohorts) ===")
        print(df_nz.groupby(["N", "solver"])[["qubits", "ideal_expectation", "noisy_expectation_c1", "zne_expectation", "err_c1", "err_zne", "p_feasible_ideal", "p_feasible_noisy_c1"]].mean().round(4))


if __name__ == "__main__":
    main()

