"""Covariance validity and financial CVaR."""
import numpy as np
import pytest

from qfar.risk import (cvar, cvar_ru_bruteforce, default_covariance, latent_correlation,
                       scenario_losses, simulate_defaults, var, variance_matrix)


def _cohort(seed, N=10):
    r = np.random.default_rng(seed)
    return r.uniform(.02, .4, N), r.integers(0, 4, N)


# 6. covariance matrix validity --------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_default_covariance_is_valid(seed):
    pd_, v = _cohort(seed)
    S = default_covariance(pd_, v, .05, .25)
    assert np.allclose(S, S.T)
    assert np.allclose(np.diag(S), pd_ * (1 - pd_))
    assert np.linalg.eigvalsh(S).min() > -1e-10
    # positive latent correlation => non-negative default covariances
    assert np.all(S[np.triu_indices(len(pd_), 1)] >= -1e-12)
    G = variance_matrix(S, np.random.default_rng(seed).uniform(.5, 2, len(pd_)))
    assert np.linalg.eigvalsh(G).min() > -1e-10


def test_latent_correlation_bounds():
    with pytest.raises(ValueError):
        latent_correlation([0, 1], .6, .5)


def test_simulated_defaults_match_analytic_moments():
    pd_, v = _cohort(7, N=6)
    S = default_covariance(pd_, v, .05, .25)
    D = simulate_defaults(pd_, v, .05, .25, 200_000, seed=1).astype(float)
    assert np.allclose(D.mean(0), pd_, atol=4e-3)
    assert np.allclose(np.cov(D.T), S, atol=4e-3)


def test_variance_term_equals_scenario_loss_variance():
    pd_, v = _cohort(3, N=6); a = np.linspace(.5, 1.5, 6); x = np.array([1, 0, 1, 1, 0, 1])
    G = variance_matrix(default_covariance(pd_, v, .05, .25), a)
    L = scenario_losses(simulate_defaults(pd_, v, .05, .25, 300_000, 2), a, x)
    assert abs(L.var() - x @ G @ x) < 0.02 * (x @ G @ x) + 1e-3


# 7. financial CVaR ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [.8, .9, .95, .99])
def test_cvar_matches_rockafellar_uryasev_minimisation(alpha):
    L = np.random.default_rng(0).gamma(2, 1, 1000).round(2)
    assert abs(cvar(L, alpha) - cvar_ru_bruteforce(L, alpha)) < 1e-9


def test_cvar_simple_cases():
    L = np.arange(1, 101, dtype=float)              # 1..100
    assert abs(cvar(L, .95) - np.mean(np.arange(96, 101))) < 1e-12
    assert var(L, .95) == 95
    assert abs(cvar(np.full(50, 3.0), .9) - 3.0) < 1e-12
    assert cvar(L, .95) >= var(L, .95) >= L.mean() - 1e-12


def test_cvar_batch_equals_rowwise():
    L = np.random.default_rng(1).normal(size=(7, 400))
    assert np.allclose(cvar(L, .9), [cvar(r, .9) for r in L])


def test_cvar_is_monotone_and_subadditive_on_scenarios():
    pd_, v = _cohort(4, N=8); a = np.ones(8)
    D = simulate_defaults(pd_, v, .05, .25, 5000, 3)
    x1 = np.array([1, 1, 0, 0, 1, 0, 0, 0]); x2 = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    c1, c2 = cvar(scenario_losses(D, a, x1), .95), cvar(scenario_losses(D, a, x2), .95)
    c12 = cvar(scenario_losses(D, a, x1 + x2), .95)
    assert c12 <= c1 + c2 + 1e-9                      # coherent risk measure: sub-additive
    assert c12 >= c1 - 1e-9                           # adding loans with non-negative losses
