"""Applicant data: SYNTHETIC generator (default) and a loader for real labelled data.

IMPORTANT: with ``source: synthetic`` every number in this project comes from a
simulation. Repayment outcomes are drawn from a known latent model, so results
are a *synthetic-data simulation study*, not evidence about real-world repayment
prediction. See docs/DATA_CARD.md.

Column roles (fix #19) -- each column has exactly one documented role:

=============================  =================================================
role                           columns
=============================  =================================================
target                         repaid (1 = repaid, 0 = defaulted)
fairness group (evaluation)    group  (0 = Group A, 1 = Group B)
protected, never a model input is_women_led (used only by the Option-B fallback)
repayment-model features       requested_amount, monthly_income, dti_ratio,
                               repayment_history, poverty_index, dependents
Need score                     poverty_index, dependents, monthly_income
Inclusion score (Option A)     first_time_formal_borrower, no_formal_account,
                               underserved_locality, branch_distance_km
risk dependence                village
=============================  =================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

GROUP_LABELS = {0: "Group A", 1: "Group B"}

PD_FEATURES = ["requested_amount", "monthly_income", "dti_ratio",
               "repayment_history", "poverty_index", "dependents"]
INCLUSION_COLUMNS = ["first_time_formal_borrower", "no_formal_account",
                     "underserved_locality", "branch_distance_km"]
REQUIRED_COLUMNS = (["applicant_id", "group", "village", "is_women_led", "repaid"]
                    + PD_FEATURES)
PROTECTED_COLUMNS = ["group", "is_women_led"]

N_VILLAGES = 12


def generate_pool(n: int, seed: int) -> pd.DataFrame:
    """Synthetic applicant pool with a known ground-truth repayment mechanism.

    Group A and Group B are *synthetic* groups with different simulated
    socio-economic distributions (Group A: lower income, more dependents, higher
    poverty index, smaller requested loans, more often first-time formal
    borrowers in underserved localities). They are not real demographic
    categories.
    """
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    A = g == 0
    income = np.where(A, rng.uniform(7000, 20000, n), rng.uniform(18000, 45000, n))
    dti = np.where(A, rng.uniform(.20, .65, n), rng.uniform(.15, .50, n))
    hist = np.where(A, rng.beta(6, 2, n), rng.beta(7, 2, n))       # prior (incl. informal/SHG) repayment record
    poverty = np.where(A, rng.uniform(.55, .95, n), rng.uniform(.15, .50, n))
    deps = np.where(A, rng.choice([2, 3, 4, 5, 6], n, p=[.15, .25, .30, .20, .10]),
                    rng.choice([0, 1, 2, 3, 4], n, p=[.10, .30, .35, .15, .10]))
    women = np.where(A, rng.random(n) < .75, rng.random(n) < .45).astype(int)
    amount = np.where(A, rng.choice([15000, 20000, 25000, 30000], n, p=[.3, .4, .2, .1]),
                      rng.choice([20000, 30000, 40000, 50000], n, p=[.2, .4, .25, .15])).astype(float)
    village = np.where(A, rng.integers(0, N_VILLAGES // 2, n),
                       N_VILLAGES // 2 + rng.integers(0, N_VILLAGES // 2, n))

    # Financial-exclusion indicators (simulated). Villages 0-3 and 6-7 are
    # "underserved" (no bank branch within 5 km); distance is drawn accordingly.
    underserved_v = np.isin(np.arange(N_VILLAGES), [0, 1, 2, 3, 6, 7])
    underserved = underserved_v[village].astype(int)
    dist = np.where(underserved == 1, rng.uniform(5, 25, n), rng.uniform(0.5, 6, n)).round(1)
    first_time = (rng.random(n) < np.where(A, .65, .30)).astype(int)
    no_account = (rng.random(n) < np.where(A, .40, .12) + .15 * underserved).astype(int)

    # Ground truth of the simulation: latent repayment propensity with an
    # UNOBSERVED village shock (creates correlated defaults) + noise.
    village_shock = rng.normal(0, 0.45, N_VILLAGES)[village]
    z = (2.9 + 4.0 * (hist - .75) - 3.0 * (dti - .35) + 0.8 * np.log(income / 20000)
         - 1.0 * (poverty - .5) - 0.3 * (amount / income) + village_shock + rng.normal(0, .5, n))
    repaid = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    return pd.DataFrame(dict(
        applicant_id=[f"APP_{i:05d}" for i in range(n)], group=g, village=village,
        monthly_income=income.round(2), dti_ratio=dti.round(3), repayment_history=hist.round(3),
        poverty_index=poverty.round(3), dependents=deps.astype(int), is_women_led=women,
        requested_amount=amount, first_time_formal_borrower=first_time,
        no_formal_account=no_account, underserved_locality=underserved,
        branch_distance_km=dist, repaid=repaid))


def load_labelled_csv(path: str, column_map: dict | None = None,
                      repaid_values=(1, "1", "repaid", "Fully Paid", "fully_paid")) -> pd.DataFrame:
    """Load REAL historical loans with an observed repayment outcome.

    ``column_map`` renames source columns to the schema in this module. The
    target is mapped to 1 (repaid) / 0 (defaulted). Inclusion columns are
    optional: if absent, scoring falls back to the Social Priority Score
    (Option B) instead of inventing them.
    """
    df = pd.read_csv(path).rename(columns=column_map or {})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"labelled data is missing required columns: {missing}")
    df["repaid"] = df["repaid"].isin(repaid_values).astype(int)
    return validate(df)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    if not set(df["repaid"].unique()) <= {0, 1}:
        raise ValueError("repaid must be binary")
    if not set(df["group"].unique()) <= {0, 1}:
        raise ValueError("group must be 0 (Group A) or 1 (Group B)")
    if (df["requested_amount"] <= 0).any():
        raise ValueError("requested_amount must be positive")
    return df


def load_pool(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    if d["source"] == "synthetic":
        return generate_pool(d["pool_size"], cfg["seed"])
    if d["source"] == "csv":
        return load_labelled_csv(d["csv_path"])
    raise ValueError(d["source"])


def split_pool(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Stratified train / validation / test split (fix #1)."""
    s = cfg["data"]["split"]
    assert abs(s["train"] + s["validation"] + s["test"] - 1) < 1e-9
    train, rest = train_test_split(df, train_size=s["train"], stratify=df["repaid"],
                                   random_state=cfg["seed"])
    val, test = train_test_split(rest, test_size=s["test"] / (s["validation"] + s["test"]),
                                 stratify=rest["repaid"], random_state=cfg["seed"] + 1)
    return {"train": train, "validation": val, "test": test}


def sample_cohorts(df: pd.DataFrame, n: int, k: int, seed: int, min_per_group: int = 2) -> list[pd.DataFrame]:
    """k reproducible cohorts of n applicants, each with >= min_per_group per group.

    The same list is passed to every solver/setting so differences in results
    are due to the algorithms, not to different applicant populations.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < k:
        c = df.iloc[rng.choice(len(df), n, replace=False)]
        if (c.group == 0).sum() >= min_per_group and (c.group == 1).sum() >= min_per_group:
            out.append(c.reset_index(drop=True))
    return out
