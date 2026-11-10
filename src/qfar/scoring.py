"""Stage 2: applicant scores F (financial viability), N (need), S (inclusion / priority).

All normalisations use FIXED, documented reference values from the config
(fixes #2, #3, #20): an applicant's score never depends on which other
applicants happen to be in the same cohort.

F_i  expected return per unit lent  mu_i = p_i r_i - (1 - p_i) LGD_i,
     normalised by its theoretical bounds [-LGD_max, r_max]:
         F_i = (mu_i + LGD_max) / (r_max + LGD_max)  in [0, 1].
     With a single r and LGD this reduces exactly to F_i = p_i (an affine map
     preserves order); F differs from p only if r_i or LGD_i vary by applicant.
     F is a normalised expected-return score under assumed r and LGD -- it is
     NOT realised profit.

N_i  = a_P P_i + a_D D_i + a_I (1 - I_i)
     D_i = clip(dependents_i / D_max, 0, 1)
     I_i = clip((PCI_i - PCI_min) / (PCI_max - PCI_min), 0, 1),
     PCI_i = monthly_income_i / (dependents_i + 1).

S_i  Option A (Inclusion Score, used when the data contain the indicators):
         S_i = a1 E1 + a2 E2 + a3 E3 + a4 E4, sum a_j = 1
         E1 first-time formal borrower, E2 no formal account,
         E3 underserved locality, E4 clip(branch_distance / dist_max, 0, 1).
     Option B (Social Priority Score, fallback): 0.6 women_led + 0.4 (1 - L_i / L_max).
     Option B is a policy priority rule, not a measure of structural exclusion.

U_i  = w_F F_i + w_N N_i + w_S S_i,  w >= 0, sum w = 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import INCLUSION_COLUMNS


def _clip01(v):
    return np.clip(np.asarray(v, float), 0.0, 1.0)


def expected_return(p, r, lgd):
    p = np.asarray(p, float)
    return p * r - (1 - p) * lgd


def financial_score(p, r, lgd, r_max=None, lgd_max=None):
    r_arr, l_arr = np.broadcast_to(r, np.shape(p)), np.broadcast_to(lgd, np.shape(p))
    r_max = float(np.max(r_arr)) if r_max is None else r_max
    lgd_max = float(np.max(l_arr)) if lgd_max is None else lgd_max
    mu = expected_return(p, r_arr, l_arr)
    return mu, (mu + lgd_max) / (r_max + lgd_max)


def need_score(df: pd.DataFrame, norm: dict, w: dict):
    assert abs(sum(w.values()) - 1) < 1e-9
    pci = df["monthly_income"].values / (df["dependents"].values + 1)
    I = _clip01((pci - norm["pci_min"]) / (norm["pci_max"] - norm["pci_min"]))
    D = _clip01(df["dependents"].values / norm["dependents_max"])
    P = _clip01(df["poverty_index"].values)
    return w["poverty"] * P + w["dependents"] * D + w["low_income"] * (1 - I), pci


def social_score(df: pd.DataFrame, cfg: dict):
    """Returns (S, kind) where kind is 'inclusion' (Option A) or 'priority' (Option B)."""
    sc, norm = cfg["scores"], cfg["normalisation"]
    if all(c in df.columns for c in INCLUSION_COLUMNS):
        a = sc["inclusion_weights"]
        assert abs(sum(a.values()) - 1) < 1e-9
        E4 = _clip01(df["branch_distance_km"].values / norm["branch_distance_max_km"])
        S = (a["first_time_formal_borrower"] * df["first_time_formal_borrower"].values
             + a["no_formal_account"] * df["no_formal_account"].values
             + a["underserved_locality"] * df["underserved_locality"].values
             + a["branch_distance"] * E4)
        return S.astype(float), "inclusion"
    a = sc["priority_weights"]
    L = _clip01(df["requested_amount"].values / sc["loan_amount_max_inr"])
    return a["women_led"] * df["is_women_led"].values + a["small_loan"] * (1 - L), "priority"


def compute_scores(df: pd.DataFrame, p_repay, cfg: dict, weights=None) -> pd.DataFrame:
    """Scores for every applicant in ``df``. ``weights`` = (w_F, w_N, w_S)."""
    if weights is None:
        w = cfg["scores"]["weights"]; weights = (w["F"], w["N"], w["S"])
    wF, wN, wS = weights
    if min(weights) < 0 or abs(wF + wN + wS - 1) > 1e-9:
        raise ValueError("weights must be non-negative and sum to 1")
    fin = cfg["finance"]
    mu, F = financial_score(p_repay, fin["interest_rate"], fin["lgd"])
    N, pci = need_score(df, cfg["normalisation"], cfg["scores"]["need_weights"])
    S, kind = social_score(df, cfg)
    out = pd.DataFrame(dict(p_repay=np.asarray(p_repay, float), exp_return=mu, F=F, N=N, S=S,
                            U=wF * F + wN * N + wS * S, pci=pci), index=df.index)
    out.attrs["social_kind"] = kind
    out.attrs["weights"] = tuple(weights)
    return out
