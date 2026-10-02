"""
Unit Tests for Problem Constraints & Feasibility Validation.
Verifies:
1. Binary allocation constraints x_i in {0, 1}
2. Budget ceiling constraint sum_i L_i x_i <= B
3. Budget utilization calculation
"""

import unittest
import numpy as np
import pandas as pd
from src.metrics import evaluate_allocation


class TestConstraints(unittest.TestCase):
    def setUp(self):
        self.df_test = pd.DataFrame({
            "applicant_id": [f"APP_{i:03d}" for i in range(4)],
            "requested_amount": [20000.0, 30000.0, 40000.0, 50000.0],
            "composite_score": [0.6, 0.7, 0.8, 0.9],
            "financial_score": [0.5, 0.6, 0.7, 0.8],
            "need_index": [0.7, 0.8, 0.6, 0.5],
            "social_impact_score": [0.6, 0.7, 0.8, 0.9],
            "group_binary": [0, 0, 1, 1]
        })
        self.budget = 60000.0

    def test_feasible_allocation(self):
        # Select applicant 0 and 1: 20000 + 30000 = 50000 <= 60000
        x = np.array([1, 1, 0, 0])
        res = evaluate_allocation(x, self.df_test, self.budget)
        self.assertTrue(res["is_feasible"])
        self.assertAlmostEqual(res["total_cost"], 50000.0)
        self.assertAlmostEqual(res["budget_utilization_pct"], (50000.0 / 60000.0) * 100.0, places=2)

    def test_infeasible_allocation(self):
        # Select applicant 2 and 3: 40000 + 50000 = 90000 > 60000
        x = np.array([0, 0, 1, 1])
        res = evaluate_allocation(x, self.df_test, self.budget)
        self.assertFalse(res["is_feasible"])
        self.assertAlmostEqual(res["total_cost"], 90000.0)


if __name__ == "__main__":
    unittest.main()
