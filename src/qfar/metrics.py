"""Evaluation metrics: fairness (directional), portfolio risk/return, feasibility (fixes #8, #9, #18).

All money values are in INR unless the name ends in ``_ref`` (units of L_ref).
"Expected return" is a model-based expectation under the assumed r and LGD,
not realised profit. "Realised" metrics use the held-out repayment labels of the
(synthetic) test applicants.
"""
from __future__ import annotations

import numpy as np

from .risk import cvar, scenario_losses, var


def fairness_metrics(x, groups) -> dict:
    x, g = np.asarray(x, float), np.asarray(groups)
    arA = float(x[g == 0].mean()); arB = float(x[g == 1].mean())
    return dict(AR_A=arA, AR_B=arB,
                dpd=arA - arB,                      # signed d(x): >0 means Group A approved more often
                abs_dpd=abs(arA - arB),
                dir_A_over_B=(arA / arB) if arB > 0 else np.nan,
                dir_B_over_A=(arB / arA) if arA > 0 else np.nan,
                lower_rate_group=("A" if arA < arB else "B" if arB < arA else "equal"))


def portfolio_metrics(x, amounts_inr, p_repay, repaid, r, lgd, Sigma, D, exposure_ref,
                      loss_ref_inr, alpha, budget_inr) -> dict:
    x = np.asarray(x, float); L = np.asarray(amounts_inr, float) * x
    p = np.asarray(p_repay, float); y = np.asarray(repaid, float)
    a_inr = lgd * L
    losses = scenario_losses(D, exposure_ref, x) * loss_ref_inr
    return dict(
        approved=int(x.sum()), lent_inr=float(L.sum()),
        budget_utilisation=float(L.sum() / budget_inr),
        expected_return_inr=float(np.sum(L * (p * r - (1 - p) * lgd))),
        expected_loss_inr=float(np.sum(a_inr * (1 - p))),
        loss_std_inr=float(np.sqrt(max(a_inr @ Sigma @ a_inr, 0.0))),
        scenario_mean_loss_inr=float(losses.mean()),
        var_inr=var(losses, alpha), cvar_inr=cvar(losses, alpha),
        realised_loss_inr=float(np.sum(a_inr * (1 - y))),
        realised_return_inr=float(np.sum(L * (y * r - (1 - y) * lgd))),
        realised_default_rate=float((x * (1 - y)).sum() / x.sum()) if x.sum() else np.nan,
    )


def jaccard(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def mean_sd(df, by, cols):
    """Mean and sample standard deviation (ddof = 1) across cohorts."""
    g = df.groupby(by)[cols]
    m, s = g.mean(), g.std(ddof=1)
    out = m.copy()
    for c in cols:
        out[c] = m[c]; out[c + "_sd"] = s[c]
    out["n_cohorts"] = g.size() if hasattr(g, "size") else len(df)
    return out[[c for col in cols for c in (col, col + "_sd")] + ["n_cohorts"]]
