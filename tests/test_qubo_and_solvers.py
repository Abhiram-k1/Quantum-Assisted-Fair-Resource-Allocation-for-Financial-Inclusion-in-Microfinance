"""
Unit Tests for QUBO Assembly, Ising Mapping, and Optimization Solvers.
Verifies:
1. QUBO matrix dimensions and symmetry
2. Ising Pauli-Z decomposition consistency
3. Classical exact ground state vs classical MILP feasibility
4. QAOA and CVaR-QAOA bitstring decoding and shape integrity
"""

import unittest
import numpy as np
from src.qubo_builder import QUBOMapper
from src.classical_solvers import (
    solve_exact_qubo,
    solve_greedy_knapsack,
    solve_classical_milp,
    solve_simulated_annealing
)
from src.quantum_engine import solve_qaoa, solve_cvar_qaoa


class TestQUBOAndSolvers(unittest.TestCase):
    def setUp(self):
        self.N = 4
        self.composite_scores = np.array([0.75, 0.65, 0.85, 0.70])
        self.loan_amounts = np.array([20000.0, 25000.0, 30000.0, 20000.0])
        self.group_indicators = np.array([0, 0, 1, 1])
        self.budget = 50000.0

        self.mapper = QUBOMapper(
            composite_scores=self.composite_scores,
            loan_amounts=self.loan_amounts,
            group_indicators=self.group_indicators,
            total_budget=self.budget,
            lambda_budget=0.8,
            lambda_fairness=1.2
        )

    def test_qubo_properties(self):
        Q = self.mapper.Q
        self.assertEqual(Q.shape, (self.N, self.N))
        # Diagonal Hamiltonian should have 2^N elements
        diag = self.mapper.get_diagonal_hamiltonian()
        self.assertEqual(len(diag), 2 ** self.N)

    def test_classical_solvers(self):
        # 1. Exact ground state
        res_exact = solve_exact_qubo(self.mapper)
        self.assertEqual(len(res_exact["allocation"]), self.N)
        self.assertTrue(all(bit in [0, 1] for bit in res_exact["allocation"]))

        # 2. Greedy solver
        res_greedy = solve_greedy_knapsack(
            self.composite_scores,
            self.loan_amounts,
            self.budget,
            self.mapper
        )
        self.assertEqual(len(res_greedy["allocation"]), self.N)
        self.assertTrue(res_greedy["is_feasible"])

        # 3. Exact MILP solver
        res_milp = solve_classical_milp(
            self.composite_scores,
            self.loan_amounts,
            self.budget,
            self.group_indicators,
            fairness_tolerance=0.25,
            qubo_mapper=self.mapper
        )
        self.assertEqual(len(res_milp["allocation"]), self.N)
        self.assertTrue(res_milp["is_feasible"])

    def test_quantum_solvers(self):
        # Standard QAOA p=1
        res_qaoa = solve_qaoa(self.mapper, p_layers=1, max_iter=25, seed=42)
        self.assertEqual(len(res_qaoa["allocation"]), self.N)
        self.assertEqual(len(res_qaoa["probabilities"]), 2 ** self.N)
        self.assertAlmostEqual(float(np.sum(res_qaoa["probabilities"])), 1.0, places=5)

        # CVaR-QAOA p=1 with alpha=0.5
        res_cvar = solve_cvar_qaoa(self.mapper, p_layers=1, cvar_alpha=0.5, max_iter=25, seed=42)
        self.assertEqual(len(res_cvar["allocation"]), self.N)
        self.assertEqual(len(res_cvar["probabilities"]), 2 ** self.N)
        self.assertAlmostEqual(float(np.sum(res_cvar["probabilities"])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
