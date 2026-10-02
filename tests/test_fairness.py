"""
Unit Tests for Demographic Fairness Metrics Engine.
Verifies:
1. Demographic Parity Difference (DPD) computation
2. Disparate Impact Ratio (DIR / 80% Rule)
3. Zero-division resilience and boundary conditions
"""

import unittest
import numpy as np
import pandas as pd
from src.metrics import evaluate_allocation


class TestFairness(unittest.TestCase):
    def setUp(self):
        # 4 applicants: 2 in Group A (Marginalized, binary 0), 2 in Group B (General, binary 1)
        self.df_test = pd.DataFrame({
            "applicant_id": [f"APP_{i:03d}" for i in range(4)],
            "requested_amount": [10000.0, 10000.0, 10000.0, 10000.0],
            "composite_score": [0.7, 0.7, 0.7, 0.7],
            "financial_score": [0.6, 0.6, 0.6, 0.6],
            "need_index": [0.8, 0.8, 0.8, 0.8],
            "social_impact_score": [0.7, 0.7, 0.7, 0.7],
            "group_binary": [0, 0, 1, 1]
        })
        self.budget = 50000.0

    def test_perfect_demographic_parity(self):
        # 1 approved from Group A (rate = 0.5), 1 approved from Group B (rate = 0.5)
        x = np.array([1, 0, 1, 0])
        res = evaluate_allocation(x, self.df_test, self.budget)
        self.assertAlmostEqual(res["demographic_parity_diff"], 0.0)
        self.assertAlmostEqual(res["disparate_impact_ratio"], 1.0)

    def test_complete_disparity(self):
        # 2 approved from Group B (rate = 1.0), 0 approved from Group A (rate = 0.0)
        x = np.array([0, 0, 1, 1])
        res = evaluate_allocation(x, self.df_test, self.budget)
        self.assertAlmostEqual(res["demographic_parity_diff"], 1.0)
        self.assertAlmostEqual(res["disparate_impact_ratio"], 0.0)

    def test_zero_denominator_handling(self):
        # 0 approved across all groups
        x = np.array([0, 0, 0, 0])
        res = evaluate_allocation(x, self.df_test, self.budget)
        self.assertAlmostEqual(res["demographic_parity_diff"], 0.0)
        self.assertAlmostEqual(res["disparate_impact_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
