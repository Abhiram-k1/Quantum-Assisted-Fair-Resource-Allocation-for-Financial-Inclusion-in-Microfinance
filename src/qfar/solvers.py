"""Classical solvers and baselines (fixes #11, #12).

Two optimisation problems are used throughout (see docs/FORMULATION.md):

P1 (QUBO-compatible, variance-aware, soft fairness)
    min_x  J1(x) = -U.x + eta x'Gamma x + lam_F d(x)^2      s.t.  w.x <= B
P2 (tail-risk-aware)
    min_x  J2(x) = J1(x) + lam_R CVaR_alpha(L(x))           s.t.  w.x <= B

Solver                         problem it actually solves
-----------------------------  ------------------------------------------------------
exact_enumeration              P1 or P2, exactly (brute force, small N only)
exact_qubo_ground              ground state of the penalised QUBO (checks the encoding)
milp_linearised                P1 or P2, exactly (McCormick-linearised products,
                               Rockafellar-Uryasev CVaR); same problem as enumeration
milp_hard_fair_utility         DIFFERENT problem: max U.x s.t. budget, |d(x)| <= eps
                               (no risk term) -- a separate policy baseline
greedy_utility_per_cost        heuristic: rank by U_i / w_i, add while budget allows
simulated_annealing            heuristic on the penalised QUBO of P1
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .qubo import AllocationQUBO
from .risk import cvar, scenario_losses


# ------------------------------------------------------------------ helpers
def repair(q: AllocationQUBO, x):
    """Drop the lowest U_i/w_i applicant until the budget holds (reported, never hidden)."""
    x = np.array(x, int); changed = False
    while q.w @ x > q.B:
        idx = np.where(x == 1)[0]
        x[idx[np.argmin(q.U[idx] / q.w[idx])]] = 0; changed = True
    return x, changed


def finalize(name, problem, q, x_raw, **extra) -> dict:
    x_raw = np.asarray(x_raw, int)[: q.N]
    x, rep = repair(q, x_raw)
    return dict(solver=name, problem=problem, x=x, x_raw=x_raw, raw_feasible=q.is_feasible(x_raw),
                was_repaired=rep, feasible=q.is_feasible(x), J1=float(q.objective_x(x)[0]), **extra)


def all_x(N: int) -> np.ndarray:
    return ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1).astype(np.int8)


def objective_p2(q, tail, X) -> np.ndarray:
    X = np.atleast_2d(X)
    return q.objective_x(X) + tail["lam_R"] * np.atleast_1d(
        cvar(scenario_losses(tail["D"], tail["a"], X), tail["alpha"]))


# ------------------------------------------------------------------ exact
def solve_exact(q: AllocationQUBO, tail: dict | None = None) -> dict:
    """Brute-force optimum of P1 (tail=None) or P2 over budget-feasible x."""
    X = all_x(q.N); feas = X @ q.w <= q.B
    J = np.full(len(X), np.inf)
    J[feas] = objective_p2(q, tail, X[feas]) if tail else q.objective_x(X[feas])
    k = int(np.argmin(J))
    return finalize("exact_enumeration", "P2" if tail else "P1", q, X[k], J=float(J[k]))


def solve_exact_qubo(q: AllocationQUBO) -> dict:
    E = q.all_energies(); k = int(np.argmin(E))
    bits = q.all_bits[k]
    return finalize("exact_qubo_ground", "P1-QUBO", q, bits[: q.N], energy=float(E[k]),
                    slack_residual=int(q.w @ bits[: q.N] + q.slack_coef @ bits[q.N:] - q.B))


# ------------------------------------------------------------------ MILP
def solve_milp(q: AllocationQUBO, tail: dict | None = None, fair_eps: float | None = None,
               utility_only: bool = False, time_limit: float = 120.0) -> dict:
    """Exact MILP.  Binary products x_i x_j are replaced by y_ij with the McCormick
    constraints y <= x_i, y <= x_j, y >= x_i + x_j - 1, y >= 0 (exact for binary x).
    CVaR (if ``tail``) uses t + sum_s u_s / ((1-alpha) S), u_s >= L_s(x) - t, u_s >= 0.

    utility_only=True, fair_eps=eps gives the separate hard-fairness baseline.
    """
    N = q.N
    Q = np.zeros((N, N)) if utility_only else q.eta * q.Gamma + q.lam_F * np.outer(q.sigma, q.sigma)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if abs(Q[i, j]) > 1e-15]
    S = 0 if tail is None else tail["D"].shape[0]
    nv = N + len(pairs) + (1 + S if tail else 0)
    c = np.zeros(nv)
    c[:N] = -q.U + np.diag(Q)
    for k, (i, j) in enumerate(pairs):
        c[N + k] = 2 * Q[i, j]
    o = N + len(pairs)
    if tail:
        c[o] = tail["lam_R"]
        c[o + 1:] = tail["lam_R"] / ((1 - tail["alpha"]) * S)

    rows = 1 + 3 * len(pairs) + S + (1 if fair_eps is not None else 0)
    A = lil_matrix((rows, nv)); lo = np.full(rows, -np.inf); hi = np.full(rows, np.inf)
    A[0, :N] = q.w; hi[0] = q.B; r = 1
    for k, (i, j) in enumerate(pairs):
        A[r, N + k] = 1; A[r, i] = -1; hi[r] = 0; r += 1
        A[r, N + k] = 1; A[r, j] = -1; hi[r] = 0; r += 1
        A[r, N + k] = 1; A[r, i] = -1; A[r, j] = -1; lo[r] = -1; r += 1
    if tail:
        La = tail["D"] * tail["a"][None, :]            # S x N loss contributions
        for s in range(S):                             # u_s + t - L_s(x) >= 0
            A[r, :N] = -La[s]; A[r, o] = 1; A[r, o + 1 + s] = 1; lo[r] = 0; r += 1
    if fair_eps is not None:
        A[r, :N] = q.sigma; lo[r] = -fair_eps; hi[r] = fair_eps; r += 1

    integrality = np.zeros(nv); integrality[:N] = 1
    lb = np.zeros(nv); ub = np.ones(nv)
    if tail:
        lb[o] = -np.inf; ub[o] = np.inf; ub[o + 1:] = np.inf
    res = milp(c, integrality=integrality, bounds=Bounds(lb, ub),
               constraints=LinearConstraint(A.tocsr(), lo, hi), options={"time_limit": time_limit})
    ok = res.status == 0
    x = np.round(res.x[:N]).astype(int) if res.x is not None else np.zeros(N, int)
    if utility_only:
        name, problem = "milp_hard_fair_utility", f"utility s.t. |d|<={fair_eps}"
    else:
        name = "milp_linearised" + ("_hardfair" if fair_eps is not None else "")
        problem = "P2" if tail else "P1"
    return finalize(name, problem, q, x, milp_status=int(res.status), milp_optimal=ok,
                    milp_objective=float(res.fun) if ok else np.nan)


# ------------------------------------------------------------------ heuristics
def solve_greedy(q: AllocationQUBO) -> dict:
    """Greedy Utility-per-Cost Baseline: ignores risk and fairness terms."""
    x = np.zeros(q.N, int); used = 0
    for i in np.argsort(-q.U / q.w, kind="stable"):
        if used + q.w[i] <= q.B:
            x[i] = 1; used += q.w[i]
    return finalize("greedy_utility_per_cost", "heuristic", q, x)


def solve_sa(q: AllocationQUBO, steps: int = 6000, restarts: int = 8, seed: int = 0) -> dict:
    """Single-bit-flip simulated annealing on the full penalised QUBO (x and slack bits)."""
    rng = np.random.default_rng(seed); M = q.M; n = q.n
    best_y, best_E = None, np.inf
    for _ in range(restarts):
        y = rng.integers(0, 2, n).astype(float)
        E = float(y @ M @ y)
        d = 1 - 2 * y
        T0 = max(1e-6, float(np.median(np.abs(2 * d * (M @ y) + M.diagonal()))))
        T, cool = T0, (1e-4) ** (1 / steps)
        for _s in range(steps):
            k = rng.integers(n); dk = 1 - 2 * y[k]
            dE = 2 * dk * (M[k] @ y) + M[k, k]
            if dE < 0 or rng.random() < np.exp(-dE / T):
                y[k] += dk; E += dE
                if E < best_E - 1e-12:
                    best_E, best_y = E, y.copy()
            T *= cool
        if best_y is None:
            best_y, best_E = y.copy(), E
    return finalize("simulated_annealing", "P1-QUBO", q, best_y[: q.N].astype(int),
                    energy=best_E + q.offset)
