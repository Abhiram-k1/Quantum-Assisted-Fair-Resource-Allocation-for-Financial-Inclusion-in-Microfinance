"""Ideal STATEVECTOR simulation of QAOA, CVaR-QAOA and VQE on the P1 QUBO (fixes #14, #15).

Nothing here runs on quantum hardware. States are exact (noise-free) 2^n-dimensional
vectors; ``shots`` samples are then drawn from the exact output distribution to
mimic measurement.

* QAOA:      |psi(g,b)> = prod_l exp(-i b_l sum X) exp(-i g_l H~) |+>^n,
             H~ = H / ||H||_1 where ||H||_1 = sum|h_i| + sum|J_ij| (computable from
             the Ising coefficients, as it would be on hardware). Minimises <H>.
* CVaR-QAOA: same circuit, minimises the mean of the LOWEST alpha_q fraction of the
             QUBO-energy distribution (Barkoutsos et al., 2020). This "quantum CVaR"
             is a lower-tail statistic of ENERGIES; it is unrelated to financial CVaR
             of portfolio LOSSES (risk.py).
* VQE:       hardware-efficient RY + CZ-ring ansatz, minimises <H>.

Selection rule: the returned allocation is the LOWEST-ENERGY sampled bitstring
among the shots (not the most frequent one); the most frequent one is reported too.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .qubo import AllocationQUBO
from .solvers import finalize


def energy_cvar_lower(probs, E, alpha, order=None) -> float:
    """Mean of the lowest-alpha probability mass of the energy distribution."""
    if alpha >= 1.0:
        return float(probs @ E)
    o = np.argsort(E) if order is None else order
    p, e = probs[o], E[o]; cum = np.cumsum(p)
    take = np.clip(alpha - (cum - p), 0, p)
    return float((take * e).sum() / alpha)


class StateVec:
    def __init__(self, q: AllocationQUBO):
        self.q, self.n, self.dim = q, q.n, 2 ** q.n
        self.E = q.all_energies(); self.order = np.argsort(self.E)
        h, J, _ = q.ising()
        self.norm = float(np.abs(h).sum() + np.abs(J).sum())
        self.Etil = (self.E - self.E.mean()) / self.norm        # constant shift = global phase only
        bits = q.all_bits.astype(int)
        ring = [(i, (i + 1) % self.n) for i in range(self.n)] if self.n > 2 else [(0, 1)]
        self.ring = ring
        self.cz_sign = 1 - 2 * (sum(bits[:, i] * bits[:, j] for i, j in ring) % 2)
        xbits = bits[:, : q.N]
        self.feasible_mask = (xbits @ q.w) <= q.B

    def plus(self):
        return np.full(self.dim, 1 / np.sqrt(self.dim), complex)

    def _apply(self, st, gate, qubit):                  # qubit i = axis n-1-i (C order)
        ax = self.n - 1 - qubit
        t = np.moveaxis(np.tensordot(gate, st.reshape([2] * self.n), axes=(1, ax)), 0, ax)
        return t.reshape(self.dim)

    def mixer(self, st, beta):                           # exp(-i beta sum_i X_i)
        c, s = np.cos(beta), -1j * np.sin(beta)
        g = np.array([[c, s], [s, c]])
        for i in range(self.n):
            st = self._apply(st, g, i)
        return st

    def ry_all(self, st, angles):
        for i, th in enumerate(angles):
            c, s = np.cos(th / 2), np.sin(th / 2)
            st = self._apply(st, np.array([[c, -s], [s, c]], complex), i)
        return st


# ------------------------------------------------------------------ circuit resources
def _asap_depth(n, gates):
    """gates: list of (qubits tuple, duration). Returns depth under ASAP scheduling
    on all-to-all connectivity (no routing / SWAPs)."""
    t = np.zeros(n, int)
    for qs, d in gates:
        start = max(t[list(qs)]); t[list(qs)] = start + d
    return int(t.max())


def qaoa_resources(q: AllocationQUBO, p: int) -> dict:
    h, J, _ = q.ising(); n = q.n
    zz = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(J[i, j]) > 1e-12]
    z = [i for i in range(n) if abs(h[i]) > 1e-12]
    layer = [((i,), 1) for i in z]
    for i, j in zz:
        layer.append(((i, j), 3))                       # RZZ = CNOT - RZ - CNOT
    layer += [((i,), 1) for i in range(n)]              # RX mixer
    gates = [((i,), 1) for i in range(n)] + layer * p   # H layer + p layers
    return dict(qubits=n, depth=_asap_depth(n, gates), two_qubit_gates=2 * len(zz) * p,
                rzz_terms=len(zz), parameters=2 * p, connectivity="all-to-all (no routing)")


def vqe_resources(q: AllocationQUBO, reps: int, ring) -> dict:
    n = q.n
    gates = []
    for _ in range(reps):
        gates += [((i,), 1) for i in range(n)] + [((i, j), 1) for i, j in ring]
    gates += [((i,), 1) for i in range(n)]
    return dict(qubits=n, depth=_asap_depth(n, gates), two_qubit_gates=len(ring) * reps,
                parameters=n * (reps + 1), connectivity="ring")


# ------------------------------------------------------------------ outcome
def _outcome(name, q, sv, probs, params, nfev_total, nfev_best, shots, rng, resources, extra):
    probs = probs / probs.sum()
    samp = rng.choice(len(probs), size=shots, p=probs)
    best_k = samp[np.argmin(sv.E[samp])]
    vals, cnt = np.unique(samp, return_counts=True)
    freq_k = vals[np.argmax(cnt)]
    x_freq = q.all_bits[freq_k][: q.N].astype(int)
    samp_x = q.all_bits[samp][:, : q.N].astype(int)
    return finalize(
        name, "P1-QUBO", q, q.all_bits[best_k][: q.N],
        selection_rule="lowest-energy sample", probs=probs, params=np.asarray(params),
        expectation=float(probs @ sv.E), shots=shots, nfev_total=nfev_total, nfev_best=nfev_best,
        p_feasible_exact=float(probs[sv.feasible_mask].sum()),
        p_feasible_shots=float(np.mean(samp_x @ q.w <= q.B)),
        shot_x_indices=(samp_x << np.arange(q.N)).sum(1),
        most_frequent_x=x_freq, most_frequent_feasible=q.is_feasible(x_freq),
        most_frequent_J1=float(q.objective_x(x_freq)[0]) if q.is_feasible(x_freq) else np.nan,
        **resources, **extra)


def solve_qaoa(q: AllocationQUBO, p: int = 1, alpha_q: float = 1.0, restarts: int = 4,
               max_iter: int = 250, shots: int = 1024, seed: int = 0, init=None, sv=None) -> dict:
    rng = np.random.default_rng(seed); sv = sv or StateVec(q); count = [0]

    def evolve(par):
        st = sv.plus()
        for l in range(p):
            st = sv.mixer(st * np.exp(-1j * par[l] * sv.Etil), par[p + l])
        return np.abs(st) ** 2

    def obj(par):
        count[0] += 1
        return energy_cvar_lower(evolve(par), sv.E, alpha_q, sv.order)

    starts = [] if init is None else [np.asarray(init, float)]
    while len(starts) < restarts:
        starts.append(np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)]))
    best, best_nfev = None, 0
    for x0 in starts:
        c0 = count[0]
        r = minimize(obj, x0, method="COBYLA", options={"maxiter": max_iter, "rhobeg": 0.3})
        if best is None or r.fun < best.fun:
            best, best_nfev = r, count[0] - c0
    name = f"qaoa_p{p}" if alpha_q >= 1 else f"cvar_qaoa_p{p}_a{alpha_q}"
    return _outcome(name, q, sv, evolve(best.x), best.x, count[0], best_nfev, shots, rng,
                    qaoa_resources(q, p), dict(p=p, alpha_q=alpha_q, restarts=restarts))


def interp_init(params_p: np.ndarray, p: int) -> np.ndarray:
    """INTERP initialisation (Zhou et al., 2020): depth-p optimum -> depth-(p+1) start."""
    g, b = params_p[:p], params_p[p:]
    def up(v):
        v = np.concatenate([[0.0], v, [0.0]])
        return np.array([(i / p) * v[i] + ((p - i) / p) * v[i + 1] for i in range(p + 1)])
    return np.concatenate([up(g), up(b)])


def solve_vqe(q: AllocationQUBO, reps: int = 2, restarts: int = 4, max_iter: int = 400,
              shots: int = 1024, seed: int = 0, sv=None) -> dict:
    rng = np.random.default_rng(seed); sv = sv or StateVec(q); n = q.n; count = [0]

    def evolve(par):
        st = np.zeros(sv.dim, complex); st[0] = 1
        for l in range(reps):
            st = sv.ry_all(st, par[l * n:(l + 1) * n]) * sv.cz_sign
        return np.abs(sv.ry_all(st, par[reps * n:])) ** 2

    def obj(par):
        count[0] += 1
        return float(evolve(par) @ sv.E)

    best, best_nfev = None, 0
    for _ in range(restarts):
        c0 = count[0]
        r = minimize(obj, rng.normal(0, np.pi / 2, n * (reps + 1)), method="COBYLA",
                     options={"maxiter": max_iter, "rhobeg": 0.5})
        if best is None or r.fun < best.fun:
            best, best_nfev = r, count[0] - c0
    return _outcome(f"vqe_L{reps}", q, sv, evolve(best.x), best.x, count[0], best_nfev, shots, rng,
                    vqe_resources(q, reps, sv.ring), dict(reps=reps, restarts=restarts))
