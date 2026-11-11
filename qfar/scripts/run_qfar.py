"""Run the unified Q-FAR fairness- and risk-aware quantum microfinance allocation CLI.

Examples:
    python scripts/run_qfar.py --cohort 0 --qiskit --zne
    python scripts/run_qfar.py --csv ../data/sample_applicants.csv --solver warm_start_qaoa_p1 --qiskit --zne
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qfar.config import write_manifest
from qfar.pipeline import DEFAULT_SOLVERS, run_allocation


def main():
    ap = argparse.ArgumentParser(description="Q-FAR end-to-end allocation pipeline CLI")
    ap.add_argument("--config", default=None, help="Optional YAML config path")
    ap.add_argument("--csv", default=None, help="Optional applicant CSV (v2 Q-FAR or v1 legacy schema)")
    ap.add_argument("--cohort", type=int, default=0, help="Cohort index from test pool when --csv is omitted")
    ap.add_argument("--solver", default="milp_p2", help="Primary solver for applicant selection table")
    ap.add_argument("--solvers", nargs="+", default=list(DEFAULT_SOLVERS), help="Solvers to compare")
    ap.add_argument("--budget-frac", type=float, default=None, help="Override allocation.budget_fraction")
    ap.add_argument("--eta", type=float, default=None, help="Override allocation.eta")
    ap.add_argument("--lambda-f", type=float, default=None, help="Override allocation.lambda_F")
    ap.add_argument("--lambda-r", type=float, default=None, help="Override allocation.lambda_R")
    ap.add_argument("--qiskit", action="store_true", help="Export Qiskit circuit ASCII/QASM2 diagnostics")
    ap.add_argument("--zne", action="store_true", help="Run Qiskit noisy + ZNE evaluation")
    ap.add_argument("--out-dir", default=None, help="Optional output directory for CSVs and artifacts")
    args = ap.parse_args()

    alloc_ov = {}
    if args.budget_frac is not None:
        alloc_ov["budget_fraction"] = args.budget_frac
    if args.eta is not None:
        alloc_ov["eta"] = args.eta
    if args.lambda_f is not None:
        alloc_ov["lambda_F"] = args.lambda_f
    if args.lambda_r is not None:
        alloc_ov["lambda_R"] = args.lambda_r
    overrides = {"allocation": alloc_ov} if alloc_ov else None

    res = run_allocation(
        cfg_path=args.config,
        overrides=overrides,
        csv_path=args.csv,
        cohort_idx=args.cohort,
        primary_solver=args.solver,
        solvers=args.solvers,
        include_qiskit_circuit=args.qiskit,
        include_noisy_zne=args.zne,
    )

    m = res.metadata
    print("=" * 78)
    print(f"Q-FAR Working Module | primary_solver={res.primary_solver} | PD={m['pd_source']}")
    print(
        f"N={m['n_applicants']} applicants | K={m['slack_qubits']} slack | "
        f"n={m['total_qubits']} qubits | B={m['budget_inr']:,.0f} INR ({m['budget_units']} units) | "
        f"social_mode={m['social_kind']}"
    )
    if m["skipped_solvers"]:
        print("Skipped solvers:", ", ".join(m["skipped_solvers"]))
    print("-" * 78)

    show_cols = [
        c for c in (
            "solver", "problem", "feasible", "J1", "J2", "utility", "abs_dpd",
            "cvar_inr", "budget_pct", "approved", "hit_opt_P1", "hit_opt_P2",
            "P_opt_exact", "P_feasible_exact", "runtime_s",
        )
        if c in res.comparison.columns
    ]
    print("\n[Solver Comparison]")
    print(res.comparison[show_cols].round(4).to_string(index=False))

    coh_cols = [
        c for c in (
            "applicant_id", "group", "requested_amount", "w_units", "p_repay",
            "F", "N", "S", "U", "approved",
        )
        if c in res.cohort.columns
    ]
    print(f"\n[Applicant Decisions: {res.primary_solver}] Approved={len(res.selected_ids)} -> {res.selected_ids}")
    print(res.cohort[coh_cols].round(4).to_string(index=False))

    if res.circuit_artifacts is not None:
        ca = res.circuit_artifacts
        print(
            f"\n[Qiskit Circuit ({ca['source_solver']})]: qubits={ca['qubits']}, "
            f"logical_depth={ca['logical_depth']}, transpiled_depth={ca['transpiled_depth']}, "
            f"cx={ca['transpiled_cx']}"
        )

    if res.noisy_diagnostics is not None:
        nd = res.noisy_diagnostics
        zne = nd["zne"]
        print(f"\n[Qiskit Noisy + ZNE ({nd['source_solver']})]: ideal <H>={nd['ideal_expectation']:.4f}")
        for r in nd["per_scale"]:
            print(
                f"  scale c={r['scale_factor']}: noisy <H>={r['expectation']:.4f}, "
                f"P(feasible)={r['p_feasible_exact']:.3f}, feasible={r['feasible']}, J1={r['J1']:.4f}"
            )
        print(f"  Richardson ZNE <H>(0) = {zne['zne_energy']:.4f}")

    if args.out_dir is not None:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        res.cohort.to_csv(out / "cohort_decisions.csv", index=False)
        res.comparison.to_csv(out / "solver_comparison.csv", index=False)
        if res.circuit_artifacts is not None:
            (out / "qaoa_circuit.txt").write_text(res.circuit_artifacts["circuit_text"], encoding="utf-8")
            (out / "qaoa_circuit.qasm").write_text(res.circuit_artifacts["qasm2"], encoding="utf-8")
        if res.noisy_diagnostics is not None:
            zne_payload = {
                "source_solver": res.noisy_diagnostics["source_solver"],
                "ideal_expectation": res.noisy_diagnostics["ideal_expectation"],
                "zne": res.noisy_diagnostics["zne"],
                "per_scale_summary": [
                    {
                        "scale_factor": r["scale_factor"],
                        "expectation": r["expectation"],
                        "p_feasible_exact": r["p_feasible_exact"],
                        "feasible": r["feasible"],
                        "J1": r["J1"],
                    }
                    for r in res.noisy_diagnostics["per_scale"]
                ],
            }
            (out / "noisy_zne.json").write_text(json.dumps(zne_payload, indent=2), encoding="utf-8")
        write_manifest(out, res.instance.cfg, primary_solver=res.primary_solver, metadata=res.metadata)
        print(f"\nSaved artifacts to {out}")


if __name__ == "__main__":
    main()

