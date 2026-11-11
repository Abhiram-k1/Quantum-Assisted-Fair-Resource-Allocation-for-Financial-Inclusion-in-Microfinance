"""QUBO formulation: slack encoding, energy, Ising mapping, penalty bound, budget feasibility."""
import itertools

import numpy as np
import pytest
from conftest import rand_qubo

from qfar.qubo import AllocationQUBO, budget_units, slack_coefficients
from qfar.solvers import all_x, solve_exact, solve_exact_qubo


# 2. slack-variable encoding ------------------------------------------------------
@pytest.mark.parametrize("B", list(range(1, 300)))
def test_slack_represents_exactly_0_to_B(B):
    c = slack_coefficients(B)
    reach = {0}
    for ck in c:
        reach |= {r + ck for r in reach}
    assert reach == set(range(B + 1))
    assert len(c) == int(np.ceil(np.log2(B + 1)))


def test_complete_slack_gives_zero_residual():
    q = rand_qubo(0, N=8)
    for x in all_x(q.N):
        if q.is_feasible(x):
            s = q.complete_slack(x)
            assert q.w @ x + q.slack_coef @ s == q.B


def test_budget_units_rounding_is_conservative():
    w, B = budget_units([12_300, 5_000, 7_499], 24_999, unit=5000)
    assert w.tolist() == [3, 1, 2] and B == 4
    # any x feasible in units is feasible in rupees
    amt = np.array([12_300, 5_000, 7_499])
    for x in itertools.product([0, 1], repeat=3):
        if w @ x <= B:
            assert amt @ x <= 24_999


# 3. QUBO energy correctness -----------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_qubo_energy_matches_explicit_formula(seed):
    q = rand_qubo(seed, N=7, eta=.7, lam_F=1.3)
    rng = np.random.default_rng(seed)
    for _ in range(200):
        y = rng.integers(0, 2, q.n); x, s = y[:q.N], y[q.N:]
        d = x[q.groups == 0].sum() / q.n_A - x[q.groups == 1].sum() / q.n_B
        e = (-q.U @ x + q.eta * x @ q.Gamma @ x + q.lam_F * d ** 2
             + q.lam_B * (q.w @ x + q.slack_coef @ s - q.B) ** 2)
        assert abs(e - q.energy(y)) < 1e-8


# 4. Ising equivalence -----------------------------------------------------------
@pytest.mark.parametrize("N", [3, 5, 8, 12, 20])
def test_ising_energy_equals_qubo_energy(N):
    q = rand_qubo(N, N=N, eta=.5, lam_F=1.0)
    rng = np.random.default_rng(N)
    for _ in range(300):
        y = rng.integers(0, 2, q.n)
        assert abs(q.energy(y) - q.ising_energy(y)) < 1e-7 * max(1, abs(q.energy(y)))


def test_ising_coefficients_formula():
    q = rand_qubo(1, N=6, eta=.4, lam_F=.8)
    h, J, _ = q.ising()
    assert np.allclose(h, -0.5 * q.M @ np.ones(q.n))
    iu = np.triu_indices(q.n, 1)
    assert np.allclose(J[iu], 0.5 * q.M[iu])


# 1 & 8. budget feasibility / ground state == constrained optimum -------------------------------
@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("eta,lam_F", [(0, 0), (.8, 0), (.8, 2.0), (0, 5.0)])
def test_qubo_ground_state_is_feasible_constrained_optimum(seed, eta, lam_F):
    q = rand_qubo(seed, N=7, eta=eta, lam_F=lam_F)
    a, b = solve_exact(q), solve_exact_qubo(q)
    assert b["raw_feasible"] and not b["was_repaired"] and b["slack_residual"] == 0
    assert abs(a["J1"] - b["J1"]) < 1e-9


@pytest.mark.parametrize("frac", [0.1, 0.25, 0.5, 0.75, 0.95])
@pytest.mark.parametrize("seed", range(4))
def test_feasibility_across_budgets(seed, frac):
    q = rand_qubo(100 + seed, N=8, eta=1.0, lam_F=2.0, budget_frac=frac)
    b = solve_exact_qubo(q)
    assert b["raw_feasible"] and b["slack_residual"] == 0


def test_penalty_bound_separates_infeasible_states():
    """Every state with non-zero budget residual has energy > 0 >= ground state of feasible ones."""
    for seed in range(6):
        q = rand_qubo(seed, N=8, eta=1.0, lam_F=3.0)
        Y = q.all_bits.astype(int)
        resid = Y[:, :q.N] @ q.w + Y[:, q.N:] @ q.slack_coef - q.B
        E = q.all_energies()
        assert E[resid != 0].min() > 0 >= E[resid == 0].min()
        assert E[resid != 0].min() >= q.lam_B - q.lam_B_min - 1e-9


@pytest.mark.parametrize("N", [20, 40, 80])
def test_penalty_bound_large_instances_by_sampling(N):
    """Analytic guarantee checked on large applicant sets (no enumeration possible)."""
    q = rand_qubo(N, N=N, eta=1.0, lam_F=2.0, w_range=(1, 30))
    rng = np.random.default_rng(N)
    for _ in range(5000):
        y = rng.integers(0, 2, q.n)
        r = q.w @ y[:q.N] + q.slack_coef @ y[q.N:] - q.B
        if r != 0:
            assert q.energy(y) > 0.0          # empty allocation (x=0, s=B) has energy 0
    x0 = np.zeros(q.N, int)
    assert abs(q.energy(np.concatenate([x0, q.complete_slack(x0)]))) < 1e-12 * q.offset  # float round-off only


def test_too_small_penalty_can_break_feasibility():
    """Documents WHY the bound matters: with lam_B far below sum U+ the ground state can be infeasible."""
    broke = False
    for seed in range(30):
        q = rand_qubo(seed, N=7, lam_B=0.05)
        broke |= not solve_exact_qubo(q)["raw_feasible"]
    assert broke


def test_rejects_non_psd_gamma():
    with pytest.raises(ValueError):
        AllocationQUBO([1, 1], [1, 1], 1, [0, 1], np.array([[0, 1], [1, 0.]]), eta=1.0)


# 5. fairness penalty --------------------------------------------------------------------
def test_fairness_term_is_parity_gap_squared():
    q = rand_qubo(5, N=6, lam_F=1.0)
    for x in all_x(q.N):
        gap = x[q.groups == 0].mean() - x[q.groups == 1].mean()
        assert abs(q.sigma @ x - gap) < 1e-12
        assert abs(q.objective_x(x)[0] - (-q.U @ x + q.eta * x @ q.Gamma @ x + q.lam_F * gap ** 2)) < 1e-12
