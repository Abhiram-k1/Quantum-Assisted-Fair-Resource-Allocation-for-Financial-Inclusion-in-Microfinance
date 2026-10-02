"""
Risk Scenario Simulation and Tail-Risk (VaR & CVaR) Engine.
Computes:
1. Monte Carlo default scenarios under repayment uncertainty p_i
2. Portfolio Value-at-Risk (VaR_alpha)
3. Portfolio Conditional Value-at-Risk (CVaR_alpha / Expected Shortfall)
4. Portfolio Expected Loss vs Tail Loss profile
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class RiskEngine:
    """
    Simulates stochastic repayment/default outcomes and computes
    discrete tail-risk metrics (VaR, CVaR) for any loan allocation vector.
    """
    def __init__(self, loan_amounts: np.ndarray, default_probs: np.ndarray, n_scenarios: int = 2000, seed: int = 42):
        """
        Parameters:
        -----------
        loan_amounts : np.ndarray
            Loan principal requested by each applicant (L_i in INR).
        default_probs : np.ndarray
            Predicted probability of default for each applicant (q_i = 1 - p_i).
        n_scenarios : int
            Number of Monte Carlo scenarios for discrete distribution estimation.
        seed : int
            Random seed for scenario reproducibility.
        """
        self.loan_amounts = np.asarray(loan_amounts, dtype=float)
        self.default_probs = np.clip(np.asarray(default_probs, dtype=float), 0.0, 1.0)
        self.N = len(loan_amounts)
        self.n_scenarios = n_scenarios
        self.seed = seed
        
        # Pre-generate discrete scenario matrix: shape (n_scenarios, N)
        # Entry (s, i) = 1 if applicant i defaults in scenario s, 0 if repaid
        np.random.seed(seed)
        rand_draws = np.random.rand(n_scenarios, self.N)
        self.scenario_defaults = (rand_draws < self.default_probs).astype(float)
        
    def evaluate_allocation_risk(
        self,
        allocation: np.ndarray,
        alpha: float = 0.95
    ) -> Dict[str, float]:
        """
        Evaluates portfolio loss distribution for a given binary allocation x in {0, 1}^N.
        
        Parameters:
        -----------
        allocation : np.ndarray
            Binary allocation vector (1 if approved, 0 otherwise).
        alpha : float
            Confidence level for VaR and CVaR (e.g., 0.95 or 0.90).
            
        Returns:
        --------
        dict containing:
            - expected_loss: mean loss across all scenarios (INR)
            - var_alpha: Value-at-Risk at confidence level alpha (INR)
            - cvar_alpha: Conditional Value-at-Risk (tail average loss above VaR) (INR)
            - max_possible_loss: total approved capital at risk (INR)
            - loss_standard_dev: standard deviation of portfolio losses
        """
        x = np.asarray(allocation, dtype=float)
        
        # Loss in scenario s = sum_i L_i * x_i * default_{s, i}
        # Vectorized calculation: scenario_losses of shape (n_scenarios,)
        loss_per_applicant = self.loan_amounts * x  # shape (N,)
        scenario_losses = self.scenario_defaults @ loss_per_applicant  # shape (n_scenarios,)
        
        # Sort scenario losses in ascending order
        sorted_losses = np.sort(scenario_losses)
        
        # 1. Expected Loss (Mean)
        expected_loss = float(np.mean(sorted_losses))
        
        # 2. Value-at-Risk (VaR_alpha): alpha quantile
        # e.g., for alpha=0.95, 95% of scenario losses are below VaR_alpha
        var_idx = int(np.floor(alpha * self.n_scenarios))
        var_idx = min(var_idx, self.n_scenarios - 1)
        var_alpha = float(sorted_losses[var_idx])
        
        # 3. Conditional Value-at-Risk (CVaR_alpha): Expected loss in the worst (1 - alpha) tail
        tail_losses = sorted_losses[var_idx:]
        cvar_alpha = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_alpha
        
        max_possible_loss = float(np.sum(loss_per_applicant))
        loss_std = float(np.std(sorted_losses))
        
        return {
            "expected_loss": round(expected_loss, 2),
            f"var_{int(alpha*100)}": round(var_alpha, 2),
            f"cvar_{int(alpha*100)}": round(cvar_alpha, 2),
            "max_possible_loss": round(max_possible_loss, 2),
            "loss_std": round(loss_std, 2)
        }
