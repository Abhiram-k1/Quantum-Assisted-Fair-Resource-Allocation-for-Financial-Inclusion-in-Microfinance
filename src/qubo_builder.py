"""
QUBO Matrix and Ising Spin Hamiltonian Builder for Fair Microfinance Allocation.
Maps the multi-objective constrained knapsack problem into:
1. Upper-triangular QUBO matrix Q
2. Pauli-Z Ising Hamiltonian parameters (h, J, offset)
"""

from typing import Dict, Tuple, Any
import numpy as np


class QUBOMapper:
    """
    Constructs the Quadratic Unconstrained Binary Optimization (QUBO) matrix
    and converts it to an Ising Spin Glass Hamiltonian:
        H_C = sum_i h_i Z_i + sum_{i < j} J_{ij} Z_i Z_j + offset * I
    """

    def __init__(
        self,
        composite_scores: np.ndarray,
        loan_amounts: np.ndarray,
        group_indicators: np.ndarray,
        total_budget: float,
        lambda_budget: float = 0.80,
        lambda_fairness: float = 1.50,
        scale_budget_weights: bool = True
    ):
        """
        Parameters:
        -----------
        composite_scores : np.ndarray
            Array of normalized composite benefit scores C_i in [0, 1].
        loan_amounts : np.ndarray
            Array of requested loan amounts w_i.
        group_indicators : np.ndarray
            Binary group labels (0: Group A / Marginalized, 1: Group B / General).
        total_budget : float
            Total capital budget limit B.
        lambda_budget : float
            Penalty weight for violating the budget capacity constraint.
        lambda_fairness : float
            Penalty weight for group demographic parity disparity.
        scale_budget_weights : bool
            Whether to normalize loan amounts and budget to maintain numerical stability.
        """
        self.C = np.asarray(composite_scores, dtype=float)
        self.w_raw = np.asarray(loan_amounts, dtype=float)
        self.G = np.asarray(group_indicators, dtype=int)
        self.B_raw = float(total_budget)
        self.lambda_B = float(lambda_budget)
        self.lambda_F = float(lambda_fairness)
        self.N = len(self.C)

        # Scale weights and budget to O(1) for numerical stability in QUBO
        if scale_budget_weights:
            self.w_scale = np.mean(self.w_raw)
            self.w = self.w_raw / self.w_scale
            self.B = self.B_raw / self.w_scale
        else:
            self.w_scale = 1.0
            self.w = self.w_raw
            self.B = self.B_raw

        # Precompute group sizes and fairness coefficients: sigma_i
        self.n_A = max(1, int(np.sum(self.G == 0)))
        self.n_B = max(1, int(np.sum(self.G == 1)))
        self.sigma = np.where(self.G == 0, 1.0 / self.n_A, -1.0 / self.n_B)

        self.Q = np.zeros((self.N, self.N), dtype=float)
        self.qubo_offset = 0.0

        # Ising parameters
        self.h = np.zeros(self.N, dtype=float)
        self.J = np.zeros((self.N, self.N), dtype=float)
        self.ising_offset = 0.0

        # Build matrices
        self._build_qubo()
        self._qubo_to_ising()

    def _build_qubo(self):
        """Constructs the upper-triangular QUBO matrix Q."""
        # 1. Diagonal terms: Q_ii = -C_i + lambda_B * (w_i^2 - 2 * B * w_i) + lambda_F * sigma_i^2
        for i in range(self.N):
            term_obj = - self.C[i]
            term_budget = self.lambda_B * (self.w[i] ** 2 - 2.0 * self.B * self.w[i])
            term_fair = self.lambda_F * (self.sigma[i] ** 2)
            self.Q[i, i] = term_obj + term_budget + term_fair

        # 2. Off-diagonal terms (i < j): Q_ij = 2 * lambda_B * w_i * w_j + 2 * lambda_F * sigma_i * sigma_j
        for i in range(self.N):
            for j in range(i + 1, self.N):
                term_budget_cross = 2.0 * self.lambda_B * (self.w[i] * self.w[j])
                term_fair_cross = 2.0 * self.lambda_F * (self.sigma[i] * self.sigma[j])
                self.Q[i, j] = term_budget_cross + term_fair_cross

        # Constant offset from budget penalty expansion (B^2)
        self.qubo_offset = self.lambda_B * (self.B ** 2)

    def _qubo_to_ising(self):
        """
        Maps QUBO to Ising Hamiltonian parameters via x_i = (I - Z_i) / 2:
        H_C = sum_i h_i Z_i + sum_{i < j} J_{ij} Z_i Z_j + ising_offset * I
        """
        # Symmetrized QUBO matrix representation for coupling conversion
        Q_sym = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            Q_sym[i, i] = self.Q[i, i]
            for j in range(i + 1, self.N):
                Q_sym[i, j] = self.Q[i, j] / 2.0
                Q_sym[j, i] = self.Q[i, j] / 2.0

        # Two-qubit exchange coupling J_ij = 0.25 * Q_ij
        for i in range(self.N):
            for j in range(i + 1, self.N):
                self.J[i, j] = 0.25 * self.Q[i, j]

        # Longitudinal fields h_i
        for i in range(self.N):
            row_sum = np.sum(Q_sym[i, :]) - Q_sym[i, i]
            self.h[i] = - 0.5 * self.Q[i, i] - 0.5 * row_sum

        # Constant offset
        total_q_sum = np.sum(Q_sym)
        self.ising_offset = self.qubo_offset + 0.5 * np.sum(np.diag(self.Q)) + 0.25 * np.sum(self.Q - np.diag(np.diag(self.Q)))

    def evaluate_bitstring_energy(self, x: np.ndarray) -> float:
        """
        Evaluates the exact classical objective energy for a binary vector x in {0, 1}^N:
        E(x) = x^T Q x + qubo_offset
        """
        x_vec = np.asarray(x, dtype=float)
        return float(x_vec @ self.Q @ x_vec + self.qubo_offset)

    def get_diagonal_hamiltonian(self) -> np.ndarray:
        """
        Computes the exact diagonal elements of H_C across all 2^N computational basis states.
        State index k corresponds to bitstring x_N ... x_1 in binary.
        """
        dim = 2 ** self.N
        diag_energies = np.zeros(dim, dtype=float)

        for k in range(dim):
            # Extract binary representation of length N
            # bit string: x[0] corresponds to qubit 0 (MSB or LSB, convention: qubit i is (k >> i) & 1)
            x_bits = np.array([(k >> i) & 1 for i in range(self.N)], dtype=float)
            diag_energies[k] = self.evaluate_bitstring_energy(x_bits)

        return diag_energies

    def get_summary(self) -> Dict[str, Any]:
        """Returns metadata summary of the QUBO and Ising formulation."""
        return {
            "num_variables": self.N,
            "lambda_budget": self.lambda_B,
            "lambda_fairness": self.lambda_F,
            "raw_budget": self.B_raw,
            "scaled_budget": self.B,
            "weight_scale": self.w_scale,
            "qubo_offset": self.qubo_offset,
            "ising_offset": self.ising_offset,
            "max_coupling_J": float(np.max(np.abs(self.J))),
            "max_field_h": float(np.max(np.abs(self.h)))
        }


if __name__ == "__main__":
    from src.data_generator import generate_microfinance_cohort
    from src.scoring import compute_multiobjective_scores

    df = generate_microfinance_cohort(n_applicants=6, seed=42)
    df_scored, arr = compute_multiobjective_scores(df)

    mapper = QUBOMapper(
        composite_scores=arr["composite_scores"],
        loan_amounts=arr["loan_amounts"],
        group_indicators=arr["group_indicators"],
        total_budget=80000.0,
        lambda_budget=0.8,
        lambda_fairness=1.5
    )

    print("QUBO Matrix Shape:", mapper.Q.shape)
    print("QUBO Offset:", mapper.qubo_offset)
    print("Ising h fields:", np.round(mapper.h, 3))
    print("Summary:", mapper.get_summary())
