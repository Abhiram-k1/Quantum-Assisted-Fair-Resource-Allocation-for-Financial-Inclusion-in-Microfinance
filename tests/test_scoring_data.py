"""Scores use fixed normalisation; data roles; reproducibility."""
import numpy as np
import pandas as pd
import pytest

from qfar.data import (INCLUSION_COLUMNS, PROTECTED_COLUMNS, generate_pool, sample_cohorts,
                       split_pool, load_labelled_csv)
from qfar.experiment import build_instance, cohort_risk
from qfar.pd_model import FEATURE_SETS
from qfar.scoring import compute_scores


@pytest.fixture(scope="module")
def pool(cfg):
    df = generate_pool(1500, cfg["seed"])
    df["p_repay"] = np.clip(0.5 + 0.5 * df.repayment_history - 0.3 * df.dti_ratio, 0.01, 0.99)
    return df


def test_scores_are_independent_of_cohort_composition(pool, cfg):
    """Fix #2/#3/#20: an applicant's F, N, S, U must not change when other applicants change."""
    base = compute_scores(pool, pool.p_repay, cfg)
    for seed in range(5):
        idx = np.random.default_rng(seed).choice(len(pool), 8, replace=False)
        sub = pool.iloc[idx]
        s = compute_scores(sub, sub.p_repay, cfg)
        for col in ["F", "N", "S", "U"]:
            assert np.allclose(s[col].values, base[col].values[idx])


def test_score_ranges_and_F_definition(pool, cfg):
    s = compute_scores(pool, pool.p_repay, cfg)
    for col in ["F", "N", "S", "U"]:
        assert s[col].between(0, 1).all()
    r, lgd = cfg["finance"]["interest_rate"], cfg["finance"]["lgd"]
    assert np.allclose(s.exp_return, pool.p_repay * r - (1 - pool.p_repay) * lgd)
    assert np.allclose(s.F, (s.exp_return + lgd) / (r + lgd))
    assert np.allclose(s.F, pool.p_repay)          # documented consequence of constant r, LGD
    assert s.attrs["social_kind"] == "inclusion"


def test_social_score_falls_back_to_priority_without_indicators(pool, cfg):
    s = compute_scores(pool.drop(columns=INCLUSION_COLUMNS), pool.p_repay, cfg)
    assert s.attrs["social_kind"] == "priority"
    L = np.clip(pool.requested_amount / cfg["scores"]["loan_amount_max_inr"], 0, 1)
    assert np.allclose(s.S, 0.6 * pool.is_women_led + 0.4 * (1 - L))


def test_weights_must_be_on_simplex(pool, cfg):
    with pytest.raises(ValueError):
        compute_scores(pool, pool.p_repay, cfg, (0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        compute_scores(pool, pool.p_repay, cfg, (1.2, -0.1, -0.1))


def test_protected_attributes_are_not_model_features():
    for cols in FEATURE_SETS.values():
        assert not set(cols) & set(PROTECTED_COLUMNS)
        assert not set(cols) & set(INCLUSION_COLUMNS)


def test_split_is_disjoint_and_stratified(pool, cfg):
    sp = split_pool(pool, cfg)
    ids = [set(d.applicant_id) for d in sp.values()]
    assert sum(len(i) for i in ids) == len(pool) and not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    rates = [d.repaid.mean() for d in sp.values()]
    assert max(rates) - min(rates) < 0.02


# 10. reproducibility ----------------------------------------------------------------------------
def test_reproducibility_with_fixed_seeds(cfg):
    a, b = generate_pool(800, 11), generate_pool(800, 11)
    pd.testing.assert_frame_equal(a, b)
    assert not generate_pool(800, 12).equals(a)
    a["p_repay"] = b["p_repay"] = 0.8
    ca, cb = sample_cohorts(a, 8, 4, 3), sample_cohorts(b, 8, 4, 3)
    for x, y in zip(ca, cb):
        pd.testing.assert_frame_equal(x, y)
    ra, rb = cohort_risk(ca[0], cfg, 99), cohort_risk(cb[0], cfg, 99)
    assert np.array_equal(ra.D, rb.D) and np.allclose(ra.Sigma, rb.Sigma)
    ia, ib = build_instance(ca[0], cfg, ra), build_instance(cb[0], cfg, rb)
    assert np.array_equal(ia.q.M, ib.q.M)


def test_cohorts_have_both_groups(pool):
    for c in sample_cohorts(pool, 8, 20, 0, min_per_group=2):
        assert (c.group == 0).sum() >= 2 and (c.group == 1).sum() >= 2


def test_labelled_csv_loader(tmp_path, pool):
    p = tmp_path / "loans.csv"
    out = pool.drop(columns=["p_repay"]).rename(columns={"repaid": "loan_status"})
    out["loan_status"] = np.where(out.loan_status == 1, "Fully Paid", "Charged Off")
    out.to_csv(p, index=False)
    df = load_labelled_csv(p, {"loan_status": "repaid"})
    assert np.array_equal(df.repaid.values, pool.repaid.values)
    with pytest.raises(ValueError):
        load_labelled_csv(p)                        # target column missing without mapping
