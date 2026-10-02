"""
Unit Tests for Tail-Risk Simulation (VaR/CVaR) and CVaR-QAOA Hamiltonian Properties.
Verifies:
1. Monte Carlo default loss scenario generator
2. VaR and CVaR tail monotonicity (CVaR >= VaR >= Expected Loss)
3. CVaR quantum objective consistency: CVaR_{alpha=1.0}(state) == <psi | H_C | psi>
"""

import unittest
import numpy as np
from src.risk_engine import RiskEngine
from src.qubo_builder import QUBOMapper
from src.quantum_engine import QuantumSimulator


class TestCVaRRisk(unittest.TestCase):
    def setUp(self):
        self.loan_amounts = np.array([20000.0, 30000.0, 40000.0, 15000.0])
        self.default_probs = np.array([0.10, 0.25, 0.40, 0.05])
        self.risk_engine = RiskEngine(
            loan_amounts=self.loan_amounts,
            default_probs=self.default_probs,
            n_scenarios=2000,
            seed=42
        )

    def test_risk_monotonicity(self):
        # Allocation selecting all 4 applicants
        x = np.array([1, 1, 1, 1])
        res = self.risk_engine.evaluate_allocation_risk(x, alpha=0.95)
        
        exp_loss = res["expected_loss"]
        var_95 = res["var_95"]
        cvar_95 = res["cvar_95"]

        # Mathematical property: CVaR_alpha >= VaR_alpha >= Expected Loss
        self.assertGreaterEqual(cvar_95, var_95 - 1e-3)
        self.assertGreaterEqual(var_95, exp_loss - 1e-3)
        self.assertLessEqual(cvar_95, res["max_possible_loss"])

    def test_cvar_quantum_objective_limit(self):
        # Test that CVaR with alpha=1.0 exactly recovers standard expectation value
        mapper = QUBOMapper(
            composite_scores=np.array([0.7, 0.8, 0.6, 0.9]),
            loan_amounts=self.loan_amounts,
            group_indicators=np.array([0, 1, 0, 1]),
            total_budget=60000.0
        )
        sim = QuantumSimulator(mapper)
        state = sim.get_uniform_superposition()

        exp_energy = sim.compute_energy_expectation(state)
        cvar_1_energy = sim.compute_cvar_energy(state, alpha=1.0)

        self.assertAlmostEqual(exp_energy, cvar_1_energy, places=5)

    def test_cvar_tail_filtering(self):
        # Test that CVaR with alpha=0.25 concentrates on lower energies than standard expectation
        mapper = QUBOMapper(
            composite_scores=np.array([0.7, 0.8, 0.6, 0.9]),
            loan_amounts=self.loan_amounts,
            group_indicators=np.array([0, 1, 0, 1]),
            total_budget=60000.0
        )
        sim = QuantumSimulator(mapper)
        state = sim.get_uniform_superposition()

        exp_energy = sim.compute_energy_expectation(state)
        cvar_quarter_energy = sim.compute_cvar_energy(state, alpha=0.25)

        # In minimization, the 25% lowest quantile average must be <= overall mean
        self.assertLessEqual(cvar_quarter_energy, exp_energy)


if __name__ == "__main__":
    unittest.main()
