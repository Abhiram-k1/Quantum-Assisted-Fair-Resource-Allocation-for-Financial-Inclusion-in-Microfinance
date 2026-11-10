r"""QUBO / Ising model of the allocation problem P1 (fixes #6, #7, #8, #10, #13).

Variables  y = [x_1..x_N, s_0..s_{K-1}]  (n = N + K binary variables / qubits)

  E(x, s) = - sum_i U_i x_i                              utility
            + eta * x^T Gamma x                          portfolio-loss VARIANCE (not CVaR)
            + lam_F * d(x)^2                             SOFT fairness (Approach A)
            + lam_B * (sum_i w_i x_i + sum_k c_k s_k - B)^2   budget via slack bits

d(x) = sum_{i in A} x_i / n_A - sum_{i in B} x_i / n_B = sigma^T x,
sigma_i = +1/n_A (Group A), -1/n_B (Group B).

Budget. w_i and B are INTEGER multiples of a money unit (loan amounts are rounded
UP and the budget DOWN, so integer feasibility implies feasibility in rupees).
Slack coefficients c = [1, 2, ..., 2^(K-2), B - (2^(K-1) - 1)] represent every
integer 0..B exactly and nothing above B.

Penalty bound (sufficient, proved in docs/FORMULATION.md):
    lam_B > sum_i max(U_i, 0).
Because the residual r = w.x + c.s - B is an integer, any assignment with r != 0
has energy >= -sum_i max(U_i,0) + lam_B > 0, while the empty allocation with
s = B has energy exactly 0 (risk, fairness and budget terms all vanish).
Requires the risk and fairness terms to be >= 0, i.e. Gamma PSD (checked) and
lam_F, eta >= 0. Hence every ground state is budget-feasible and its x
minimises the penalty-free objective J(x) over feasible x.

Qubit convention: qubit i is bit i of the basis-state index k:  (k >> i) & 1.
"""
from __future__ import annotations

import numpy as np


def budget_units(amounts_inr, budget_inr: float, unit: int):
    """Integer loan sizes (rounded up) and budget (rounded down) in money units."""
    w = np.ceil(np.asarray(amounts_inr, float) / unit - 1e-9).astype(int)
    return w, int(np.floor(budget_inr / unit + 1e-9))


def slack_coefficients(B: int) -> np.ndarray:
    K = int(np.ceil(np.log2(B + 1)))
    return np.array([2 ** k for k in range(K - 1)] + [B - (2 ** (K - 1) - 1)], int)


def penalty_lower_bound(U) -> float:
    return float(np.maximum(np.asarray(U, float), 0).sum())


class AllocationQUBO:
    def __init__(self, U, w, B, groups, Gamma=None, eta=0.0, lam_F=0.0, lam_B=None, margin=1.1):
        self.U = np.asarray(U, float); self.w = np.asarray(w, int); self.B = int(B)
        self.groups = np.asarray(groups, int); self.N = len(self.U)
        if self.B < 1 or np.any(self.w < 1):
            raise ValueError("budget and loan sizes must be positive integers (money units)")
        if eta < 0 or lam_F < 0:
            raise ValueError("eta and lam_F must be non-negative")
        self.eta, self.lam_F = float(eta), float(lam_F)
        self.Gamma = np.zeros((self.N, self.N)) if Gamma is None else np.asarray(Gamma, float)
        if not np.allclose(self.Gamma, self.Gamma.T) or np.linalg.eigvalsh(self.Gamma).min() < -1e-9:
            raise ValueError("Gamma must be symmetric positive semi-definite")
        self.lam_B_min = penalty_lower_bound(self.U)
        self.lam_B = float(margin * self.lam_B_min + 1e-3) if lam_B is None else float(lam_B)
        self.penalty_guaranteed = self.lam_B > self.lam_B_min

        self.slack_coef = slack_coefficients(self.B); self.K = len(self.slack_coef)
        self.n = self.N + self.K
        nA = int((self.groups == 0).sum()); nB = int((self.groups == 1).sum())
        if nA == 0 or nB == 0:
            raise ValueError("each cohort needs at least one applicant per group")
        self.n_A, self.n_B = nA, nB
        self.sigma = np.where(self.groups == 0, 1.0 / nA, -1.0 / nB)

        a = np.concatenate([self.w, self.slack_coef]).astype(float)
        s = np.concatenate([self.sigma, np.zeros(self.K)])
        M = np.zeros((self.n, self.n))
        M[:self.N, :self.N] -= np.diag(self.U)                  # y_i^2 = y_i: linear terms on the diagonal
        M += self.lam_B * (np.outer(a, a) - 2 * self.B * np.diag(a))
        M += self.lam_F * np.outer(s, s)
        M[:self.N, :self.N] += self.eta * self.Gamma
        self.M = (M + M.T) / 2                                   # symmetric:  E = y^T M y + offset
        self.offset = self.lam_B * self.B ** 2
        self._bits = None; self._E = None

    # ------------------------------------------------------------- evaluation
    def energy(self, y):
        Y = np.atleast_2d(np.asarray(y, float))
        E = np.einsum("ki,ij,kj->k", Y, self.M, Y) + self.offset
        return float(E[0]) if np.ndim(y) == 1 else E

    def objective_x(self, X):
        """Penalty-free objective J(x) = -U.x + eta x'Gamma x + lam_F d(x)^2 (batch)."""
        X = np.atleast_2d(np.asarray(X, float))
        return (-X @ self.U + self.eta * np.einsum("ki,ij,kj->k", X, self.Gamma, X)
                + self.lam_F * (X @ self.sigma) ** 2)

    def is_feasible(self, x) -> bool:
        return bool(self.w @ np.asarray(x, int)[: self.N] <= self.B)

    def complete_slack(self, x) -> np.ndarray:
        """Exact slack bits for a feasible x (greedy binary decomposition of B - w.x)."""
        r = self.B - int(self.w @ np.asarray(x, int)); assert r >= 0
        s = np.zeros(self.K, int)
        for k in np.argsort(-self.slack_coef):
            if self.slack_coef[k] <= r:
                s[k] = 1; r -= self.slack_coef[k]
        assert r == 0
        return s

    @property
    def all_bits(self):
        if self._bits is None:
            k = np.arange(2 ** self.n)[:, None]
            self._bits = ((k >> np.arange(self.n)) & 1).astype(np.int8)
        return self._bits

    def all_energies(self):
        if self._E is None:
            Y = self.all_bits.astype(float)
            self._E = np.einsum("ki,ij,kj->k", Y, self.M, Y, optimize=True) + self.offset
        return self._E

    def x_index(self, x) -> int:
        x = np.asarray(x, int)[: self.N]
        return int((x << np.arange(self.N)).sum())

    def marginal_x(self, probs):
        """Probability of each x pattern (slack qubits traced out)."""
        idx = np.arange(2 ** self.n) & (2 ** self.N - 1)
        return np.bincount(idx, weights=probs, minlength=2 ** self.N)

    # ------------------------------------------------------------- Ising
    def ising(self):
        """y = (1 - z)/2  ->  H(z) = C + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j,
        h = -1/2 M 1,  J_ij = M_ij / 2 (i < j, M symmetric)."""
        M = self.M
        h = -0.5 * M.sum(1)
        J = np.triu(0.5 * M, 1)
        C = 0.25 * (M.sum() - np.trace(M)) + 0.5 * np.trace(M) + self.offset
        return h, J, C

    def ising_energy(self, y):
        h, J, C = self.ising(); z = 1 - 2 * np.asarray(y, float)
        return float(h @ z + z @ J @ z + C)

    def summary(self) -> dict:
        return dict(N=self.N, K=self.K, qubits=self.n, B_units=self.B, lam_B=self.lam_B,
                    lam_B_min=self.lam_B_min, penalty_guaranteed=self.penalty_guaranteed,
                    lam_F=self.lam_F, eta=self.eta, slack_coef=self.slack_coef.tolist())
