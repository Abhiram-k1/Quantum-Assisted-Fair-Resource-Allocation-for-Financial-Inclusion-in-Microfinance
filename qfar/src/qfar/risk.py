"""Default-dependence model, portfolio variance and FINANCIAL CVaR (fix #9).

One model of correlated defaults feeds BOTH risk measures so they are consistent:

  latent  A_i = sqrt(rho_g) Z + sqrt(rho_v) Z_v(i) + sqrt(1 - rho_g - rho_v) eps_i
  default D_i = 1[A_i < Phi^{-1}(q_i)],   q_i = 1 - p_i  (model-predicted PD)

(Gaussian one-factor copula with a global and a village factor.)

Loss of applicant i in units of a fixed reference amount L_ref:
  a_i = LGD_i * L_i / L_ref

* Variance proxy:   Var[L(x)] = x^T Gamma x,  Gamma = diag(a) Sigma diag(a),
  Sigma_ij = Cov(D_i, D_j) = Phi_2(tau_i, tau_j; R_ij) - q_i q_j   (exact under the copula).
* Financial CVaR:   scenario losses  L_s(x) = sum_i a_i D_si x_i,
  CVaR_alpha = min_t [ t + 1/((1-alpha) S) sum_s max(L_s - t, 0) ]  (Rockafellar-Uryasev).

Financial CVaR is an UPPER-tail statistic of portfolio LOSSES. It is a different
quantity from the lower-tail statistic of QUBO energies used by CVaR-QAOA
(see quantum.py).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal, norm


def latent_correlation(village, rho_g: float, rho_v: float) -> np.ndarray:
    v = np.asarray(village)
    if rho_g < 0 or rho_v < 0 or rho_g + rho_v >= 1:
        raise ValueError("need rho_g, rho_v >= 0 and rho_g + rho_v < 1")
    R = rho_g + rho_v * (v[:, None] == v[None, :])
    np.fill_diagonal(R, 1.0)
    return R


def default_covariance(pd_, village, rho_g: float, rho_v: float) -> np.ndarray:
    """Exact covariance of the default indicators D_i under the copula."""
    q = np.clip(np.asarray(pd_, float), 1e-9, 1 - 1e-9)
    tau = norm.ppf(q)
    R = latent_correlation(village, rho_g, rho_v)
    n = len(q)
    S = np.diag(q * (1 - q))
    cache: dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = (round(tau[i], 12), round(tau[j], 12), round(R[i, j], 12))
            if key not in cache:
                if R[i, j] == 0:
                    joint = q[i] * q[j]
                else:
                    joint = multivariate_normal.cdf([tau[i], tau[j]], mean=[0, 0],
                                                    cov=[[1, R[i, j]], [R[i, j], 1]])
                cache[key] = joint - q[i] * q[j]
            S[i, j] = S[j, i] = cache[key]
    return S


def loss_exposure(amounts_inr, lgd, loss_ref_inr: float) -> np.ndarray:
    return np.asarray(lgd, float) * np.asarray(amounts_inr, float) / loss_ref_inr


def variance_matrix(Sigma, exposure) -> np.ndarray:
    a = np.asarray(exposure, float)
    return a[:, None] * Sigma * a[None, :]


def simulate_defaults(pd_, village, rho_g: float, rho_v: float, n_scenarios: int, seed: int) -> np.ndarray:
    """D (S x N) with D_si = 1 if applicant i defaults in scenario s."""
    rng = np.random.default_rng(seed)
    q = np.clip(np.asarray(pd_, float), 1e-9, 1 - 1e-9)
    v = np.asarray(village)
    _, vidx = np.unique(v, return_inverse=True)
    Z = rng.standard_normal((n_scenarios, 1))
    Zv = rng.standard_normal((n_scenarios, vidx.max() + 1))[:, vidx]
    eps = rng.standard_normal((n_scenarios, len(q)))
    A = np.sqrt(rho_g) * Z + np.sqrt(rho_v) * Zv + np.sqrt(1 - rho_g - rho_v) * eps
    return (A < norm.ppf(q)[None, :]).astype(np.int8)


def scenario_losses(D, exposure, X) -> np.ndarray:
    """Losses for one allocation x (-> shape S) or many allocations X (-> shape K x S)."""
    X = np.asarray(X, float)
    return (X * exposure) @ D.T


def cvar(losses, alpha: float) -> np.ndarray | float:
    """Empirical CVaR_alpha of losses (upper tail) via the Rockafellar-Uryasev formula.

    The RU objective is convex piecewise-linear in t and is minimised at the
    empirical alpha-quantile t* = L_(ceil(alpha S)), so plugging t* in is exact.
    Accepts a 1-D array (S,) or a 2-D array (K, S) of K portfolios.
    """
    L = np.atleast_2d(np.asarray(losses, float))
    S = L.shape[1]
    k = int(np.ceil(alpha * S)) - 1
    t = np.partition(L, k, axis=1)[:, k]
    c = t + np.maximum(L - t[:, None], 0).sum(1) / ((1 - alpha) * S)
    return float(c[0]) if np.ndim(losses) == 1 else c


def var(losses, alpha: float):
    L = np.atleast_2d(np.asarray(losses, float))
    k = int(np.ceil(alpha * L.shape[1])) - 1
    v = np.partition(L, k, axis=1)[:, k]
    return float(v[0]) if np.ndim(losses) == 1 else v


def cvar_ru_bruteforce(losses, alpha: float) -> float:
    """Reference implementation: minimise the RU function over every candidate t."""
    L = np.asarray(losses, float)
    f = [t + np.maximum(L - t, 0).sum() / ((1 - alpha) * len(L)) for t in np.unique(L)]
    return float(min(f))
