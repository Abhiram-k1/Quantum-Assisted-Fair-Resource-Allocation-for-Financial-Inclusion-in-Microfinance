"""
Metrics and Fair Allocation Evaluation Suite.
Evaluates:
1. Total Utility & Socio-Economic Impact
2. Budget Utilization & Feasibility
3. Demographic Parity Difference (DPD)
4. Disparate Impact Ratio (DIR / 80% Rule)
5. Approximation Ratio relative to Exact Classical Optimum
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.qubo_builder import QUBOMapper


def evaluate_allocation(
    allocation: np.ndarray,
    df_scored: pd.DataFrame,
    budget: float,
    qubo_mapper: Optional[QUBOMapper] = None,
    optimal_utility: Optional[float] = None,
    solver_name: str = "Unknown"
) -> Dict[str, Any]:
    """
    Computes financial, social, fairness, and quantum-approximation metrics
    for a given binary allocation vector x in {0, 1}^N.
    """
    x = np.asarray(allocation, dtype=int)
    N = len(x)

    w = df_scored["requested_amount"].values
    C = df_scored["composite_score"].values
    F = df_scored["financial_score"].values
    N_need = df_scored["need_index"].values
    S = df_scored["social_impact_score"].values
    groups = df_scored["group_binary"].values

    total_cost = float(np.sum(w * x))
    total_utility = float(np.sum(C * x))
    total_financial = float(np.sum(F * x))
    total_need = float(np.sum(N_need * x))
    total_social = float(np.sum(S * x))
    total_approved = int(np.sum(x))

    is_feasible = bool(total_cost <= budget + 1e-5)
    budget_utilization = float((total_cost / budget) * 100.0)

    # Group approval rates
    mask_A = (groups == 0)
    mask_B = (groups == 1)
    n_A = max(1, int(np.sum(mask_A)))
    n_B = max(1, int(np.sum(mask_B)))

    approved_A = int(np.sum(x[mask_A]))
    approved_B = int(np.sum(x[mask_B]))

    rate_A = float(approved_A / n_A)
    rate_B = float(approved_B / n_B)

    # Demographic Parity Difference: |rate_A - rate_B|
    dpd = float(abs(rate_A - rate_B))

    # Disparate Impact Ratio: rate_A / rate_B
    if rate_B > 0:
        dir_score = float(rate_A / rate_B)
    else:
        dir_score = 1.0 if rate_A == 0 else float("inf")

    energy = None
    if qubo_mapper is not None:
        energy = float(qubo_mapper.evaluate_bitstring_energy(x))

    approx_ratio = None
    if optimal_utility is not None and optimal_utility > 0:
        approx_ratio = float(total_utility / optimal_utility)

    return {
        "solver": solver_name,
        "total_approved": total_approved,
        "total_cost": round(total_cost, 2),
        "budget_limit": round(budget, 2),
        "budget_utilization_pct": round(budget_utilization, 2),
        "is_feasible": is_feasible,
        "total_utility": round(total_utility, 4),
        "financial_return": round(total_financial, 4),
        "social_need_met": round(total_need, 4),
        "social_impact": round(total_social, 4),
        "rate_group_A_marginalized": round(rate_A, 3),
        "rate_group_B_general": round(rate_B, 3),
        "demographic_parity_diff": round(dpd, 4),
        "disparate_impact_ratio": round(dir_score, 4),
        "qubo_energy": round(energy, 4) if energy is not None else None,
        "approximation_ratio": round(approx_ratio, 4) if approx_ratio is not None else None,
        "allocation_bitstring": "".join(map(str, x))
    }


def compile_comparison_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compiles a clean pandas DataFrame summarizing all solver performances.
    """
    display_cols = [
        "solver",
        "total_approved",
        "total_utility",
        "budget_utilization_pct",
        "is_feasible",
        "demographic_parity_diff",
        "disparate_impact_ratio",
        "approximation_ratio",
        "qubo_energy",
        "allocation_bitstring"
    ]
    df_res = pd.DataFrame(results_list)
    return df_res[display_cols]
