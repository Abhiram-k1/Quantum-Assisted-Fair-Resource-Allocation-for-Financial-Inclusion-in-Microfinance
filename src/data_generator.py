"""
Applicant Data Generator for Microfinance Capital Allocation.
Generates realistic borrower cohorts with financial attributes, social vulnerability,
demographic group labels, and requested loan amounts.
"""

from typing import Optional
import numpy as np
import pandas as pd


def generate_microfinance_cohort(
    n_applicants: int = 10,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generates a realistic cohort of microfinance applicants.

    Parameters:
    -----------
    n_applicants : int
        Number of applicants in the cohort (default: 10 for NISQ simulation).
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    pd.DataFrame:
        DataFrame containing applicant profiles:
        - applicant_id: Unique string identifier
        - monthly_income: Estimated household monthly income (INR)
        - dti_ratio: Debt-to-income ratio (0.1 to 0.8)
        - repayment_history: Historical repayment fidelity score (0.0 to 1.0)
        - poverty_index: Multidimensional poverty index (0.0: low, 1.0: extreme poverty)
        - dependents: Number of household dependents (1 to 6)
        - is_women_led: Whether the micro-enterprise is women-led (Boolean)
        - demographic_group: Group A (Rural / Marginalized) or Group B (Urban / General)
        - requested_amount: Requested loan principal in currency units (e.g., 10k to 50k INR)
    """
    if seed is not None:
        np.random.seed(seed)

    applicant_ids = [f"MFI_APP_{i+1:03d}" for i in range(n_applicants)]

    # Assign demographic groups: Group A (Marginalized/Rural) ~ 50%, Group B (General/Urban) ~ 50%
    demographic_groups = np.random.choice(
        ["Group_A_Marginalized", "Group_B_General"],
        size=n_applicants,
        p=[0.5, 0.5]
    )

    monthly_incomes = []
    dti_ratios = []
    repayment_histories = []
    poverty_indices = []
    dependents_list = []
    women_led_flags = []
    requested_amounts = []

    for group in demographic_groups:
        if group == "Group_A_Marginalized":
            # Typically lower formal income, higher poverty, higher dependents, higher women-led SHGs
            income = float(np.random.uniform(7000, 20000))
            dti = float(np.random.uniform(0.20, 0.65))
            repay = float(np.random.beta(a=6, b=2))  # generally reliable peer adherence
            poverty = float(np.random.uniform(0.55, 0.95))
            deps = int(np.random.choice([3, 4, 5, 6], p=[0.25, 0.35, 0.25, 0.15]))
            is_women = bool(np.random.choice([True, False], p=[0.80, 0.20]))
            amount = float(np.random.choice([15000, 20000, 25000, 30000], p=[0.3, 0.4, 0.2, 0.1]))
        else:
            # Group B: Urban / semi-formal, slightly higher income, lower poverty index
            income = float(np.random.uniform(18000, 45000))
            dti = float(np.random.uniform(0.15, 0.50))
            repay = float(np.random.beta(a=7, b=2))
            poverty = float(np.random.uniform(0.15, 0.50))
            deps = int(np.random.choice([1, 2, 3, 4], p=[0.35, 0.35, 0.20, 0.10]))
            is_women = bool(np.random.choice([True, False], p=[0.45, 0.55]))
            amount = float(np.random.choice([25000, 35000, 45000, 50000], p=[0.2, 0.4, 0.25, 0.15]))

        monthly_incomes.append(round(income, 2))
        dti_ratios.append(round(dti, 3))
        repayment_histories.append(round(repay, 3))
        poverty_indices.append(round(poverty, 3))
        dependents_list.append(deps)
        women_led_flags.append(is_women)
        requested_amounts.append(amount)

    df = pd.DataFrame({
        "applicant_id": applicant_ids,
        "demographic_group": demographic_groups,
        "monthly_income": monthly_incomes,
        "dti_ratio": dti_ratios,
        "repayment_history": repayment_histories,
        "poverty_index": poverty_indices,
        "dependents": dependents_list,
        "is_women_led": women_led_flags,
        "requested_amount": requested_amounts
    })

    return df


if __name__ == "__main__":
    df = generate_microfinance_cohort(n_applicants=8, seed=42)
    print("Generated Cohort Sample:")
    print(df[["applicant_id", "demographic_group", "requested_amount", "repayment_history", "poverty_index"]])
