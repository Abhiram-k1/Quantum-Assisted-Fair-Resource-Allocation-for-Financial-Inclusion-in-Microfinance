"""Glue code: cohort -> scores -> risk model -> QUBO -> solver results -> metric rows.

Every solver for a given cohort receives the SAME Instance (same applicants,
scores, budget, covariance and default scenarios), fix #20.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import fairness_metrics, portfolio_metrics
from .qubo import AllocationQUBO, budget_units
from .risk import default_covariance, loss_exposure, simulate_defaults, variance_matrix
from .scoring import compute_scores
from .solvers import objective_p2


@dataclass
class CohortRisk:
    Sigma: np.ndarray          # Cov(D_i, D_j)
    exposure: np.ndarray       # a_i = LGD L_i / L_ref
    Gamma: np.ndarray          # variance matrix in L_ref^2 units
    D: np.ndarray              # S x N default scenarios


@dataclass
class Instance:
    cohort: pd.DataFrame
    scores: pd.DataFrame
    risk: CohortRisk
    q: AllocationQUBO
    budget_inr: float
    tail: dict
    cfg: dict = field(repr=False)


def cohort_risk(coh: pd.DataFrame, cfg: dict, scenario_seed: int) -> CohortRisk:
    rk, fin, nm = cfg["risk"], cfg["finance"], cfg["normalisation"]
    pd_ = 1 - coh["p_repay"].values
    Sigma = default_covariance(pd_, coh["village"].values, rk["rho_global"], rk["rho_village"])
    a = loss_exposure(coh["requested_amount"].values, fin["lgd"], nm["loss_reference_inr"])
    D = simulate_defaults(pd_, coh["village"].values, rk["rho_global"], rk["rho_village"],
                          rk["n_scenarios"], scenario_seed)
    return CohortRisk(Sigma, a, variance_matrix(Sigma, a), D)


def build_instance(coh: pd.DataFrame, cfg: dict, risk: CohortRisk, weights=None,
                   eta=None, lam_F=None, lam_R=None) -> Instance:
    al = cfg["allocation"]
    eta = al["eta"] if eta is None else eta
    lam_F = al["lambda_F"] if lam_F is None else lam_F
    lam_R = al["lambda_R"] if lam_R is None else lam_R
    sc = compute_scores(coh, coh["p_repay"].values, cfg, weights)
    w, _ = budget_units(coh["requested_amount"].values, 0, al["money_unit_inr"])
    B = int(np.floor(al["budget_fraction"] * w.sum()))
    q = AllocationQUBO(sc["U"].values, w, B, coh["group"].values, risk.Gamma, eta, lam_F,
                       margin=al["lambda_B_margin"])
    tail = dict(D=risk.D, a=risk.exposure, alpha=cfg["risk"]["cvar_alpha"], lam_R=lam_R)
    return Instance(coh, sc, risk, q, B * al["money_unit_inr"], tail, cfg)


def timed(fn, *args, **kw):
    t = time.perf_counter(); r = fn(*args, **kw); r["runtime_s"] = time.perf_counter() - t
    return r


def evaluate(inst: Instance, res: dict, ref_p1: dict | None = None, ref_p2: dict | None = None) -> dict:
    q, sc, coh, cfg = inst.q, inst.scores, inst.cohort, inst.cfg
    x = res["x"]; wF, wN, wS = sc.attrs["weights"]
    row = dict(solver=res["solver"], problem=res["problem"], raw_feasible=res["raw_feasible"],
               repaired=res["was_repaired"], feasible=res["feasible"], runtime_s=res.get("runtime_s", np.nan),
               J1=res["J1"], J2=float(objective_p2(q, inst.tail, x)[0]),
               utility=float(sc.U.values @ x), contrib_F=float(wF * sc.F.values @ x),
               contrib_N=float(wN * sc.N.values @ x), contrib_S=float(wS * sc.S.values @ x),
               variance_term=float(x @ q.Gamma @ x), budget_units_used=int(q.w @ x), B_units=q.B,
               qubits=q.n, lam_B=q.lam_B, lam_B_min=q.lam_B_min)
    if ref_p1 is not None:
        row["gap_J1"] = (row["J1"] - ref_p1["J1"]) / max(abs(ref_p1["J1"]), 1e-9)
        row["hit_opt_P1"] = bool(np.array_equal(x, ref_p1["x"]))
    if ref_p2 is not None:
        row["hit_opt_P2"] = bool(np.array_equal(x, ref_p2["x"]))
    row.update(fairness_metrics(x, q.groups))
    fin, rk, nm = cfg["finance"], cfg["risk"], cfg["normalisation"]
    row.update(portfolio_metrics(x, coh["requested_amount"].values, sc["p_repay"].values,
                                 coh["repaid"].values, fin["interest_rate"], fin["lgd"],
                                 inst.risk.Sigma, inst.risk.D, inst.risk.exposure,
                                 nm["loss_reference_inr"], rk["cvar_alpha"], inst.budget_inr))
    if "probs" in res:                                   # quantum (statevector) solvers
        mx = q.marginal_x(res["probs"])
        k_opt = q.x_index(ref_p1["x"])
        feas_x = (((np.arange(2 ** q.N)[:, None] >> np.arange(q.N)) & 1) @ q.w) <= q.B
        row.update(
            P_opt_exact=float(mx[k_opt]), P_opt_shots=float(np.mean(res["shot_x_indices"] == k_opt)),
            P_opt_uniform=1 / 2 ** q.N, P_feasible_exact=res["p_feasible_exact"],
            P_feasible_shots=res["p_feasible_shots"], P_feasible_uniform=float(feas_x.mean()),
            depth=res["depth"], two_qubit_gates=res["two_qubit_gates"], shots=res["shots"],
            nfev_total=res["nfev_total"], nfev_best=res["nfev_best"],
            most_frequent_feasible=res["most_frequent_feasible"],
            most_frequent_J1=res["most_frequent_J1"], selection_rule=res["selection_rule"])
    return row
