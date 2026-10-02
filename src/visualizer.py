"""
Presentation-Ready Visual Analytics Engine for Microfinance Capital Allocation.
Generates:
1. visuals/qubo_matrix_heatmap.png
2. visuals/quantum_vs_classical_comparison.png
3. visuals/qaoa_convergence_profile.png
4. visuals/fairness_vs_budget_tradeoff.png
5. visuals/bitstring_probability_distribution.png
"""

import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_style():
    """Sets a clean, modern aesthetic for academic presentations."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_qubo_heatmap(Q: np.ndarray, output_path: str):
    """
    Plots the QUBO Matrix Heatmap showing diagonal linear utilities
    and off-diagonal quadratic constraints (budget + fairness).
    """
    setup_plot_style()
    N = Q.shape[0]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # Mask lower triangular part for clean upper-triangular visualization
    mask = np.tril(np.ones_like(Q, dtype=bool), k=-1)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        Q,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0.0,
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8, "label": "QUBO Coefficient Value"},
        ax=ax
    )

    ax.set_title(f"QUBO Interaction Matrix Q ({N}x{N})\nDiagonal: Linear Objective & Self-Penalties | Off-Diagonal: Pairwise Constraints",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Applicant Index (j)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Applicant Index (i)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_quantum_vs_classical(results_df: pd.DataFrame, output_path: str):
    """
    Multi-metric bar chart comparing solvers across:
    1. Composite Utility
    2. Demographic Parity Difference (DPD, lower is fairer)
    3. Budget Utilization (%)
    4. Approximation Ratio
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

    solvers = results_df["solver"].tolist()
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#8e44ad", "#f39c12"][:len(solvers)]

    # 1. Total Utility
    ax1 = axes[0, 0]
    bars1 = ax1.bar(solvers, results_df["total_utility"], color=colors, alpha=0.85, edgecolor="black")
    ax1.set_title("Total Multi-Objective Utility (Higher is Better)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Utility Score", fontsize=10)
    ax1.tick_params(axis="x", rotation=25)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}", ha="center", va="bottom", fontsize=8)

    # 2. Demographic Parity Difference (DPD)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(solvers, results_df["demographic_parity_diff"], color=colors, alpha=0.85, edgecolor="black")
    ax2.set_title("Demographic Parity Difference (Lower is Fairer)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Selection Rate Disparity |Rate_A - Rate_B|", fontsize=10)
    ax2.axhline(0.10, color="red", linestyle="--", alpha=0.7, label="Fairness Tolerance (0.10)")
    ax2.tick_params(axis="x", rotation=25)
    ax2.legend(loc="upper right", fontsize=8)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.2f}", ha="center", va="bottom", fontsize=8)

    # 3. Budget Utilization (%)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(solvers, results_df["budget_utilization_pct"], color=colors, alpha=0.85, edgecolor="black")
    ax3.set_title("Capital Budget Utilization (%)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Utilization %", fontsize=10)
    ax3.axhline(100.0, color="gray", linestyle=":", label="100% Budget Ceiling")
    ax3.tick_params(axis="x", rotation=25)
    ax3.legend(loc="upper right", fontsize=8)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

    # 4. Approximation Ratio
    ax4 = axes[1, 1]
    approx = results_df["approximation_ratio"].fillna(0.0)
    bars4 = ax4.bar(solvers, approx, color=colors, alpha=0.85, edgecolor="black")
    ax4.set_title("Approximation Ratio relative to Classical Exact", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Ratio (Utility_solver / Utility_exact)", fontsize=10)
    ax4.set_ylim(0.0, 1.15)
    ax4.tick_params(axis="x", rotation=25)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Q-FAR Benchmark: Classical vs. Quantum Solvers across Multiple Metrics",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_qaoa_convergence(
    qaoa_histories: Dict[str, List[float]],
    exact_energy: float,
    output_path: str
):
    """
    Plots the energy minimization trajectory of QAOA across optimization steps.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    palette = {
        "QAOA (p=1)": "#3498db",
        "QAOA (p=2)": "#9b59b6",
        "CVaR-QAOA (p=1, alpha=0.25)": "#e74c3c",
        "VQE": "#f39c12"
    }

    for name, history in qaoa_histories.items():
        color = palette.get(name, "#2c3e50")
        ax.plot(range(1, len(history) + 1), history, label=name, color=color, linewidth=2.0)

    ax.axhline(exact_energy, color="#2ecc71", linestyle="--", linewidth=1.8, label=f"Exact Ground State Energy ({exact_energy:.2f})")

    ax.set_title("Variational Quantum Energy Convergence Profile\nHybrid Quantum-Classical Optimization using COBYLA",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Optimizer Function Evaluation Step", fontsize=10, fontweight="bold")
    ax.set_ylabel("Expectation Energy <H_C>", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_fairness_vs_budget_tradeoff(
    tradeoff_records: List[Dict[str, float]],
    output_path: str
):
    """
    Plots the Pareto frontier of Financial Return vs Demographic Equity.
    """
    setup_plot_style()
    df_t = pd.DataFrame(tradeoff_records)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    scatter = ax.scatter(
        df_t["demographic_parity_diff"],
        df_t["financial_return"],
        c=df_t["lambda_fairness"],
        cmap="viridis",
        s=120,
        edgecolors="black",
        alpha=0.9
    )

    # Annotate points with lambda_F
    for _, row in df_t.iterrows():
        ax.annotate(
            f"λ_F={row['lambda_fairness']:.1f}",
            (row["demographic_parity_diff"], row["financial_return"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            fontweight="bold"
        )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Fairness Penalty Weight (λ_F)", fontsize=9, fontweight="bold")

    ax.set_title("Pareto Frontier: Multi-Objective Trade-Off\nFinancial Return vs. Demographic Parity Disparity",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Demographic Parity Disparity (Lower = Fairer)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Expected Financial Capital Return", fontsize=10, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_bitstring_distribution(
    probabilities: np.ndarray,
    ground_state_idx: int,
    num_qubits: int,
    top_k: int = 10,
    output_path: str = "visuals/bitstring_probability_distribution.png"
):
    """
    Plots a histogram of the top-k sampled computational basis states from the quantum statevector.
    Highlights the ground state bitstring in green.
    """
    setup_plot_style()
    sorted_indices = np.argsort(probabilities)[::-1][:top_k]

    bitstrings = []
    probs = []
    colors = []

    for idx in sorted_indices:
        bstr = "".join(str((idx >> i) & 1) for i in range(num_qubits))
        bitstrings.append(f"|{bstr}⟩")
        probs.append(probabilities[idx])
        if idx == ground_state_idx:
            colors.append("#2ecc71")  # Emerald green for ground state
        else:
            colors.append("#3498db")  # Blue for other states

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    bars = ax.bar(bitstrings, probs, color=colors, alpha=0.85, edgecolor="black")

    ax.set_title(f"Quantum Measurement Probability Distribution (Top {top_k} Basis States)\nConstructive Quantum Interference Peak at Optimal Loan Allocation",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Computational Basis State |x_0 x_1 ... x_{N-1}⟩", fontsize=10, fontweight="bold")
    ax.set_ylabel("Measurement Probability P(|x⟩)", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=35)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.3f}", ha="center", va="bottom", fontsize=8)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Optimal Allocation (Ground State)"),
        Patch(facecolor="#3498db", edgecolor="black", label="Sub-optimal Candidate States")
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
