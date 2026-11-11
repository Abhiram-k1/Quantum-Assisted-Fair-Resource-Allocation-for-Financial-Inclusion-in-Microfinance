"""Unified end-to-end Q-FAR allocation module.

Connects data loading / schema normalisation (both v2 Q-FAR and legacy v1 CSVs),
frozen PD inference, cohort credit-risk simulation, QUBO + MILP optimisation,
quantum solvers (QAOA, CVaR-QAOA, Warm-Start QAOA, VQE), and optional Qiskit
circuit export + depolarising/readout noise & ZNE diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .data import PD_FEATURES, load_pool, sample_cohorts, split_pool
from .experiment import Instance, build_instance, cohort_risk, evaluate, timed
from .pd_model import build_model
from .quantum import StateVec, solve_qaoa, solve_vqe, solve_warm_start_qaoa
from .solvers import (
    solve_exact,
    solve_exact_qubo,
    solve_greedy,
    solve_milp,
    solve_sa,
)

DEFAULT_SOLVERS = (
    "exact_p1",
    "exact_p2",
    "exact_qubo",
    "milp_p2",
    "milp_hard_fair",
    "greedy",
    "sa",
    "qaoa_p1",
    "qaoa_p2",
    "cvar_qaoa_p1",
    "warm_start_qaoa_p1",
    "vqe_L2",
)


@dataclass
class AllocationResult:
    cohort: pd.DataFrame
    comparison: pd.DataFrame
    primary_solver: str
    selected_ids: list
    instance: Instance
    solver_outputs: dict[str, dict]
    circuit_artifacts: dict | None
    noisy_diagnostics: dict | None
    metadata: dict


def _normalise_group(val) -> int:
    s = str(val).strip()
    su = s.upper()
    if (
        su in {"A", "GROUP_A", "GROUP A", "0", "0.0"}
        or su.startswith(("GROUP_A", "GROUP A", "A_"))
        or su.endswith(("_A", " A"))
        or "MARGINAL" in su
    ):
        return 0
    if (
        su in {"B", "GROUP_B", "GROUP B", "1", "1.0"}
        or su.startswith(("GROUP_B", "GROUP B", "B_"))
        or su.endswith(("_B", " B"))
        or "GENERAL" in su
    ):
        return 1
    raise ValueError(f"Unrecognised demographic group label: {val!r}")


def normalize_applicant_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise v2 Q-FAR or legacy v1 applicant tables into the canonical Q-FAR schema."""
    out = df.copy()
    if "group" not in out.columns and "demographic_group" in out.columns:
        out["group"] = out["demographic_group"].map(_normalise_group).astype(int)
    elif "group" in out.columns:
        out["group"] = out["group"].map(_normalise_group).astype(int)
    else:
        raise ValueError("Input CSV must contain 'group' or 'demographic_group'")
    out["group_label"] = out["group"].map({0: "A", 1: "B"})

    if "applicant_id" not in out.columns:
        out["applicant_id"] = [f"APP_{i:04d}" for i in range(len(out))]
    if "village" not in out.columns:
        out["village"] = "V0"
    if "repaid" not in out.columns:
        out["repaid"] = np.nan

    for col in PD_FEATURES:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")
        out[col] = pd.to_numeric(out[col], errors="raise")
    if "is_women_led" not in out.columns:
        raise ValueError("Missing required column: is_women_led")
    if out["is_women_led"].dtype == bool:
        out["is_women_led"] = out["is_women_led"].astype(int)
    else:
        out["is_women_led"] = (
            out["is_women_led"].astype(str).str.strip().str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
            .astype(int)
        )
    return out


def ensure_pd_predictions(
    df: pd.DataFrame,
    cfg: dict,
    model_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Attach frozen out-of-sample ``p_repay`` predictions if not already present."""
    out = df.copy()
    if "p_repay" in out.columns and out["p_repay"].notna().all():
        out["p_repay"] = np.clip(out["p_repay"].astype(float), 1e-6, 1.0 - 1e-6)
        return out, "precomputed"

    mpath = Path(model_path) if model_path is not None else (ROOT / "results/01_pd_model/pd_model.joblib")
    if mpath.exists():
        bundle = joblib.load(mpath)
        model, cols = bundle["model"], bundle["cols"]
        source = f"joblib:{mpath.name}"
    else:
        pool = load_pool(cfg)
        train, _, _ = split_pool(pool, cfg)
        cols = list(PD_FEATURES)
        model = build_model("logreg", "none", seed=int(cfg["seed"]))
        model.fit(train[cols].values, train["repaid"].astype(int).values)
        source = "trained_logreg_fallback"

    X_in = out[cols] if hasattr(model, "feature_names_in_") else out[cols].values
    out["p_repay"] = np.clip(model.predict_proba(X_in)[:, 1], 1e-6, 1.0 - 1e-6)
    return out, source


def load_cohort_input(
    cfg: dict,
    csv_path: str | Path | None = None,
    cohort_idx: int = 0,
    use_precomputed_test: bool = True,
    pd_model_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load either an external applicant CSV or cohort ``cohort_idx`` from the test split."""
    if csv_path is not None:
        raw = pd.read_csv(csv_path)
        norm = normalize_applicant_df(raw)
        scored, pd_src = ensure_pd_predictions(norm, cfg, model_path=pd_model_path)
        return scored.reset_index(drop=True), pd_src

    test_scored = ROOT / "data/processed/test_scored.csv"
    if use_precomputed_test and test_scored.exists():
        pool = pd.read_csv(test_scored)
        pd_src = "test_scored.csv"
    else:
        full_pool = load_pool(cfg)
        _, _, test_df = split_pool(full_pool, cfg)
        pool, pd_src = ensure_pd_predictions(test_df, cfg, model_path=pd_model_path)

    al = cfg["allocation"]
    n = int(al["cohort_size"])
    k = max(int(cohort_idx) + 1, int(cfg["experiments"].get("allocation_cohorts", 30)))
    cohorts = sample_cohorts(pool, n=n, k=k, seed=int(cfg["seed"]) + 100, min_per_group=int(al["min_per_group"]))
    if cohort_idx < 0 or cohort_idx >= len(cohorts):
        raise IndexError(f"cohort_idx={cohort_idx} out of range [0, {len(cohorts) - 1}]")
    return cohorts[cohort_idx].reset_index(drop=True), pd_src


def _run_solver(name: str, inst: Instance, sv: StateVec | None, cfg: dict, seed: int) -> dict:
    q = inst.q
    sol = cfg["solvers"]
    al = cfg["allocation"]
    if name == "exact_p1":
        res = timed(solve_exact, q)
    elif name == "exact_p2":
        res = timed(solve_exact, q, inst.tail)
    elif name == "exact_qubo":
        res = timed(solve_exact_qubo, q)
    elif name == "milp_p1":
        res = timed(solve_milp, q)
    elif name == "milp_p2":
        res = timed(solve_milp, q, inst.tail)
    elif name == "milp_hard_fair":
        res = timed(solve_milp, q, None, al["hard_fairness_eps"], True)
    elif name == "greedy":
        res = timed(solve_greedy, q)
    elif name == "sa":
        res = timed(solve_sa, q, **sol["sa"], seed=seed)
    elif name == "qaoa_p1":
        res = timed(solve_qaoa, q, p=1, alpha_q=1.0, restarts=sol["qaoa"]["restarts"],
                    max_iter=sol["qaoa"]["max_iter"], shots=sol["qaoa"]["shots"], seed=seed, sv=sv)
    elif name == "qaoa_p2":
        res = timed(solve_qaoa, q, p=2, alpha_q=1.0, restarts=sol["qaoa"]["restarts"],
                    max_iter=sol["qaoa"]["max_iter"], shots=sol["qaoa"]["shots"], seed=seed, sv=sv)
    elif name == "qaoa_p3":
        res = timed(solve_qaoa, q, p=3, alpha_q=1.0, restarts=sol["qaoa"]["restarts"],
                    max_iter=sol["qaoa"]["max_iter"], shots=sol["qaoa"]["shots"], seed=seed, sv=sv)
    elif name == "cvar_qaoa_p1":
        cq = sol["cvar_qaoa"]
        res = timed(solve_qaoa, q, p=1, alpha_q=cq["alpha_q"], restarts=cq["restarts"],
                    max_iter=cq["max_iter"], shots=cq["shots"], seed=seed, sv=sv)
    elif name == "warm_start_qaoa_p1":
        ws = sol.get("warm_start_qaoa", sol["qaoa"])
        res = timed(solve_warm_start_qaoa, q, p=1, alpha_q=1.0, eps=float(ws.get("eps", 0.05)),
                    restarts=int(ws.get("restarts", 4)), max_iter=int(ws.get("max_iter", 250)),
                    shots=int(ws.get("shots", 1024)), seed=seed, sv=sv)
    elif name == "vqe_L2":
        res = timed(solve_vqe, q, **sol["vqe"], seed=seed, sv=sv)
    else:
        raise ValueError(f"Unknown solver name: {name}")
    res["solver"] = name
    return res


def run_allocation(
    cfg_path: str | Path | None = None,
    overrides: dict | None = None,
    csv_path: str | Path | None = None,
    cohort_idx: int = 0,
    primary_solver: str = "milp_p2",
    solvers: Iterable[str] = DEFAULT_SOLVERS,
    include_qiskit_circuit: bool = False,
    include_noisy_zne: bool = False,
    pd_model_path: str | Path | None = None,
) -> AllocationResult:
    """Run the full Q-FAR pipeline on one cohort and return decisions + comparisons."""
    cfg = load_config(cfg_path, overrides=overrides)
    seed = int(cfg["seed"]) + int(cohort_idx)
    coh, pd_src = load_cohort_input(cfg, csv_path=csv_path, cohort_idx=cohort_idx, pd_model_path=pd_model_path)

    risk = cohort_risk(coh, cfg, scenario_seed=seed + 1000)
    inst = build_instance(coh, cfg, risk)
    q = inst.q

    solver_list = list(dict.fromkeys([*solvers, primary_solver]))
    # Ensure P1 and P2 references are computed when tractable so gaps/hit rates are populated
    can_exact = q.N <= 16
    can_quantum = q.n <= 15
    if can_exact and "exact_p1" not in solver_list:
        solver_list.insert(0, "exact_p1")
    if can_exact and "exact_p2" not in solver_list:
        solver_list.insert(1 if can_exact else 0, "exact_p2")

    sv = StateVec(q) if (can_quantum and any(s.startswith(("qaoa", "cvar_qaoa", "warm_start", "vqe")) for s in solver_list)) else None

    outputs: dict[str, dict] = {}
    skipped: list[str] = []
    for sname in solver_list:
        if sname.startswith("exact") and not can_exact:
            skipped.append(f"{sname} (N={q.N}>16)")
            continue
        if sname.startswith(("qaoa", "cvar_qaoa", "warm_start", "vqe")) and not can_quantum:
            skipped.append(f"{sname} (qubits={q.n}>15)")
            continue
        outputs[sname] = _run_solver(sname, inst, sv, cfg, seed=seed)

    ref_p1 = outputs.get("exact_p1") or outputs.get("milp_p1")
    if ref_p1 is None:
        ref_p1 = timed(solve_milp, q)
    ref_p2 = outputs.get("exact_p2") or outputs.get("milp_p2") or ref_p1

    rows = [evaluate(inst, res, ref_p1=ref_p1, ref_p2=ref_p2) for res in outputs.values()]
    comparison = pd.DataFrame(rows)
    if "solver" in comparison.columns:
        comparison["budget_pct"] = 100.0 * comparison["budget_utilisation"]

    if primary_solver not in outputs:
        primary_solver = "milp_p2" if "milp_p2" in outputs else next(iter(outputs))

    chosen_x = outputs[primary_solver]["x"].astype(bool)
    cohort_view = coh.copy()
    for col in ("F", "N", "S", "U"):
        cohort_view[col] = inst.scores[col].values
    cohort_view["w_units"] = q.w
    cohort_view["approved"] = chosen_x
    selected_ids = cohort_view.loc[chosen_x, "applicant_id"].tolist()

    circuit_artifacts = None
    noisy_diagnostics = None
    if include_qiskit_circuit or include_noisy_zne:
        from .qiskit_backend import export_circuit_artifacts, run_noisy_and_zne

        qkey = next((k for k in ("warm_start_qaoa_p1", "qaoa_p1", "cvar_qaoa_p1") if k in outputs), None)
        if qkey is not None:
            qres = outputs[qkey]
            thetas = (2.0 * np.arcsin(np.sqrt(qres["c_star"]))) if "c_star" in qres else None
            p_val = int(qres.get("p", 1))
            if include_qiskit_circuit:
                circuit_artifacts = export_circuit_artifacts(q, qres["params"], p=p_val, warm_start_thetas=thetas)
                circuit_artifacts["source_solver"] = qkey
            if include_noisy_zne:
                ncfg = cfg.get("noise", {})
                noisy_diagnostics = run_noisy_and_zne(
                    q, qres["params"], p=p_val,
                    scales=tuple(ncfg.get("zne_scales", [1, 3, 5])),
                    p1_depol=float(ncfg.get("p1_depol", 0.002)),
                    p2_depol=float(ncfg.get("p2_depol", 0.01)),
                    p_readout=float(ncfg.get("p_readout", 0.02)),
                    shots=int(ncfg.get("shots", 1024)),
                    seed=seed,
                    warm_start_thetas=thetas,
                    n_trajectories=int(ncfg.get("n_trajectories", 256)),
                )
                noisy_diagnostics["ideal_expectation"] = float(qres["expectation"])
                noisy_diagnostics["source_solver"] = qkey

    return AllocationResult(
        cohort=cohort_view,
        comparison=comparison,
        primary_solver=primary_solver,
        selected_ids=selected_ids,
        instance=inst,
        solver_outputs=outputs,
        circuit_artifacts=circuit_artifacts,
        noisy_diagnostics=noisy_diagnostics,
        metadata={
            "n_applicants": q.N,
            "slack_qubits": q.K,
            "total_qubits": q.n,
            "budget_inr": float(inst.budget_inr),
            "budget_units": int(q.B),
            "social_kind": inst.scores.attrs.get("social_kind", "unknown"),
            "pd_source": pd_src,
            "skipped_solvers": skipped,
        },
    )
