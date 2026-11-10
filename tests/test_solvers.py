"""Solver correctness: exact vs QUBO vs MILP, MILP vs its stated objective, quantum simulator sanity."""
import numpy as np
import pytest
from conftest import rand_qubo

from qfar.quantum import (StateVec, energy_cvar_lower, interp_init, qaoa_resources, solve_qaoa,
                          solve_vqe)
from qfar.risk import cvar, scenario_losses, simulate_defaults
from qfar.solvers import (all_x, objective_p2, solve_exact, solve_exact_qubo, solve_greedy,
                          solve_milp, solve_sa)


def _tail(q, seed, S=400, lam_R=0.8):
    r = np.random.default_rng(seed)
    D = simulate_defaults(r.uniform(.05, .4, q.N), r.integers(0, 3, q.N), .05, .25, S, seed)
    return dict(D=D, a=q.w / q.w.mean(), alpha=.9, lam_R=lam_R)


# 8/9. exact vs MILP on the SAME objective -------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("eta,lam_F", [(0, 0), (1.0, 2.0), (2.0, 0), (0, 6.0)])
def test_linearised_milp_equals_enumeration_P1(seed, eta, lam_F):
    q = rand_qubo(seed, N=9, eta=eta, lam_F=lam_F)
    a, m = solve_exact(q), solve_milp(q)
    assert m["milp_optimal"] and m["raw_feasible"]
    assert abs(a["J1"] - m["J1"]) < 1e-7
    assert abs(m["milp_objective"] - m["J1"]) < 1e-6        # MILP objective == stated objective at x


@pytest.mark.parametrize("seed", range(6))
def test_cvar_milp_equals_enumeration_P2(seed):
    q = rand_qubo(seed, N=8, eta=1.0, lam_F=2.0); tail = _tail(q, seed)
    a, m = solve_exact(q, tail), solve_milp(q, tail)
    assert m["milp_optimal"]
    J2m = objective_p2(q, tail, m["x"])[0]
    assert abs(a["J"] - J2m) < 1e-6
    assert abs(m["milp_objective"] - J2m) < 1e-6


@pytest.mark.parametrize("seed", range(8))
def test_hard_fair_milp_solves_its_stated_problem(seed):
    eps = 0.25
    q = rand_qubo(seed, N=9, eta=1.0, lam_F=2.0)
    m = solve_milp(q, fair_eps=eps, utility_only=True)
    X = all_x(q.N); ok = (X @ q.w <= q.B) & (np.abs(X @ q.sigma) <= eps + 1e-12)
    best = (X[ok] @ q.U).max()
    assert m["milp_optimal"] and abs(q.U @ m["x"] - best) < 1e-7
    assert abs(q.sigma @ m["x"]) <= eps + 1e-9 and q.is_feasible(m["x"])


def test_hard_fair_constraint_is_actually_enforced_in_p1_milp():
    for seed in range(6):
        q = rand_qubo(seed, N=9, eta=1.0, lam_F=0.0)
        m = solve_milp(q, fair_eps=0.1)
        if m["milp_optimal"]:
            assert abs(q.sigma @ m["x"]) <= 0.1 + 1e-9


# heuristics -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_greedy_and_sa_are_feasible(seed):
    q = rand_qubo(seed, N=10, eta=1.0, lam_F=2.0)
    g = solve_greedy(q)
    assert g["raw_feasible"] and g["solver"] == "greedy_utility_per_cost"
    order = np.argsort(-q.U / q.w, kind="stable")
    assert g["x"][order[0]] == 1 or q.w[order[0]] > q.B
    s = solve_sa(q, steps=3000, restarts=4, seed=seed)
    assert s["feasible"]
    assert s["J1"] >= solve_exact(q)["J1"] - 1e-9


def test_sa_finds_optimum_on_small_instances():
    hits = sum(np.array_equal(solve_sa(q := rand_qubo(s, N=7, eta=1, lam_F=2), seed=s)["x"],
                              solve_exact(q)["x"]) for s in range(10))
    assert hits >= 7


# quantum simulator ---------------------------------------------------------------------------------
def test_statevector_unitarity_and_energy_cvar():
    q = rand_qubo(4, N=5); sv = StateVec(q)
    st = sv.mixer(sv.plus() * np.exp(-1j * .3 * sv.Etil), .4)
    assert abs(np.linalg.norm(st) - 1) < 1e-10
    p = np.abs(st) ** 2
    assert abs(energy_cvar_lower(p, sv.E, 1.0) - p @ sv.E) < 1e-9
    assert energy_cvar_lower(p, sv.E, .1) <= p @ sv.E + 1e-12
    assert abs(energy_cvar_lower(p, sv.E, 1e-9) - sv.E.min()) < 1e-6 * (abs(sv.E.min()) + 1)


def test_mixer_matches_dense_matrix():
    q = rand_qubo(2, N=2); sv = StateVec(q)
    X = np.array([[0, 1], [1, 0]]); I = np.eye(2)
    n = sv.n; beta = .37
    H = sum(np.kron(np.kron(np.eye(2 ** (n - 1 - i)), X), np.eye(2 ** i)) for i in range(n))
    w, V = np.linalg.eigh(H)
    U = V @ np.diag(np.exp(-1j * beta * w)) @ V.conj().T
    psi = np.random.default_rng(0).normal(size=sv.dim) + 0j; psi /= np.linalg.norm(psi)
    assert np.allclose(U @ psi, sv.mixer(psi, beta))
    assert I.shape == (2, 2)


def test_qaoa_and_vqe_outputs_are_consistent():
    q = rand_qubo(3, N=4, eta=.5, lam_F=1.0)
    for r in (solve_qaoa(q, p=1, restarts=2, max_iter=60, shots=256, seed=0),
              solve_qaoa(q, p=1, alpha_q=.25, restarts=2, max_iter=60, shots=256, seed=0),
              solve_vqe(q, reps=1, restarts=2, max_iter=80, shots=256, seed=0)):
        assert abs(r["probs"].sum() - 1) < 1e-9 and r["feasible"]
        assert 0 <= r["p_feasible_exact"] <= 1 and r["selection_rule"] == "lowest-energy sample"
        assert r["depth"] > 0 and r["qubits"] == q.n


def test_interp_init_shapes_and_endpoints():
    p1 = np.array([.4, .2])
    p2 = interp_init(p1, 1)
    assert p2.shape == (4,) and np.allclose(p2, [.4, .4, .2, .2])
    assert qaoa_resources(rand_qubo(0, N=4), 2)["parameters"] == 4


def test_objective_p2_is_p1_plus_cvar():
    q = rand_qubo(1, N=6, eta=1, lam_F=1); tail = _tail(q, 1)
    x = np.array([1, 0, 1, 0, 1, 0])
    want = q.objective_x(x)[0] + tail["lam_R"] * cvar(scenario_losses(tail["D"], tail["a"], x), tail["alpha"])
    assert abs(objective_p2(q, tail, x)[0] - want) < 1e-12
    assert solve_exact_qubo(q)["problem"] == "P1-QUBO"
