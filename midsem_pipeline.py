"""
================================================================================
Q-FAR: Quantum-Assisted Fair Resource Allocation for Financial Inclusion
MID-SEMESTER BENCHMARK & EVALUATION PIPELINE (70% ACCOMPLISHMENT CHECKPOINT)
================================================================================
Executes the expanded Phase 1 & Phase 2 pipeline:
1. Microfinance applicant cohort generation & AI repayment prediction (Calibrated ML)
2. Multi-objective tri-scoring (Financial Return, Urgent Need, Social Inclusion)
3. Stochastic Monte Carlo scenario generation & discrete tail-risk modeling (VaR/CVaR)
4. QUBO matrix assembly & Ising spin Hamiltonian mapping
5. Classical solvers: Greedy Knapsack, Exact Combinatorial, Exact MILP, Simulated Annealing
6. Quantum variational algorithms: QAOA (p=1), QAOA (p=2), CVaR-QAOA (alpha=0.25), VQE
7. Multi-metric evaluation (Utility, Budget, Expected Loss, 95% CVaR, DPD, DIR, Approx Ratio)
8. High-resolution presentation visuals & Qiskit circuit export
================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_microfinance_cohort
from src.repayment_model import RepaymentPredictor, prepare_features
from src.scoring import compute_multiobjective_scores
from src.risk_engine import RiskEngine
from src.qubo_builder import QUBOMapper
from src.classical_solvers import (
    solve_greedy_knapsack,
    solve_exact_qubo,
    solve_classical_milp,
    solve_simulated_annealing
)
from src.quantum_engine import (
    solve_qaoa,
    solve_cvar_qaoa,
    solve_vqe,
    generate_qiskit_qaoa_circuit
)
from src.metrics import evaluate_allocation, compile_comparison_table
from src.visualizer import (
    plot_qubo_heatmap,
    plot_quantum_vs_classical,
    plot_qaoa_convergence,
    plot_fairness_vs_budget_tradeoff,
    plot_bitstring_distribution
)


def run_midsem_pipeline():
    start_time = time.time()
    print("=" * 90)
    print("  Q-FAR: QUANTUM-ASSISTED FAIR RESOURCE ALLOCATION IN MICROFINANCE")
    print("  MIDSEM EVALUATION PIPELINE - 70% ACCOMPLISHMENT CHECKPOINT")
    print("=" * 90)

    # Output directories
    os.makedirs("visuals", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: Applicant Cohort Generation & Supervised AI Model
    # ---------------------------------------------------------
    print("\n[STEP 1/8] Generating Microfinance Borrower Cohort & Training AI Repayment Model...")
    n_applicants = 8  # Optimal for fast, exact statevector quantum simulation
    budget = 110000.0  # Capital budget (INR)
    df_raw = generate_microfinance_cohort(n_applicants=n_applicants, seed=42)

    data_csv_path = os.path.join("data", "sample_applicants.csv")
    df_raw.to_csv(data_csv_path, index=False)
    print(f" -> Generated {n_applicants} applicants. Total requested capital: INR {df_raw['requested_amount'].sum():,.2f}")
    print(f" -> Available lending budget ceiling: INR {budget:,.2f}")

    # Train calibrated AI repayment predictor
    X_feat, y_rep = prepare_features(df_raw)
    predictor = RepaymentPredictor(model_type="calibrated_rf", seed=42)
    train_metrics = predictor.fit(X_feat, y_rep)
    p_repay = predictor.predict_proba(X_feat)
    df_raw["repayment_prob"] = np.round(p_repay, 4)
    print(f" -> Calibrated AI Repayment Model Trained (ROC-AUC: {train_metrics['roc_auc']:.3f}, Brier Loss: {train_metrics['brier_score']:.3f})")

    # ---------------------------------------------------------
    # STEP 2: Multi-Objective Tri-Scoring & Risk Engine Setup
    # ---------------------------------------------------------
    print("\n[STEP 2/8] Computing Multi-Objective Tri-Scoring & Initializing Risk Engine...")
    weights = (0.35, 0.35, 0.30)  # Financial Viability, Urgent Need, Social Impact
    df_scored, arr = compute_multiobjective_scores(df_raw, weights=weights)
    df_scored["repayment_prob"] = df_raw["repayment_prob"]

    # Monte Carlo Risk Engine (2,500 scenarios for tail-loss / VaR / CVaR estimation)
    risk_engine = RiskEngine(
        loan_amounts=arr["loan_amounts"],
        default_probs=1.0 - p_repay,
        n_scenarios=2500,
        seed=42
    )
    print(f" -> Initialized Monte Carlo Tail-Risk Engine (2,500 scenarios under uncertainty)")

    print(" -> Applicant Profiles & Composite Utility:")
    summary_cols = ["applicant_id", "demographic_group", "requested_amount", "repayment_prob",
                    "financial_score", "need_index", "social_impact_score", "composite_score"]
    print(df_scored[summary_cols].to_string(index=False))

    # ---------------------------------------------------------
    # STEP 3: QUBO Matrix & Ising Hamiltonian Mapping
    # ---------------------------------------------------------
    print("\n[STEP 3/8] Assembling QUBO Matrix & Mapping to Ising Spin Glass...")
    lambda_budget = 0.85
    lambda_fairness = 1.60

    qubo_mapper = QUBOMapper(
        composite_scores=arr["composite_scores"],
        loan_amounts=arr["loan_amounts"],
        group_indicators=arr["group_indicators"],
        total_budget=budget,
        lambda_budget=lambda_budget,
        lambda_fairness=lambda_fairness
    )

    info = qubo_mapper.get_summary()
    print(f" -> QUBO Matrix Shape: {qubo_mapper.Q.shape}")
    print(f" -> Budget Penalty Weight (lambda_B): {info['lambda_budget']}")
    print(f" -> Fairness Penalty Weight (lambda_F): {info['lambda_fairness']}")
    print(f" -> QUBO Constant Offset: {info['qubo_offset']:.3f}")
    print(f" -> Ising Spin Hamiltonian Offset: {info['ising_offset']:.3f}")
    print(f" -> Max 2-Qubit Coupling J_ij: {info['max_coupling_J']:.4f}")

    # ---------------------------------------------------------
    # STEP 4: Solving via Classical Solvers (Baselines & Exact MILP)
    # ---------------------------------------------------------
    print("\n[STEP 4/8] Executing Classical Solvers...")

    # A. Greedy Knapsack Heuristic
    res_greedy = solve_greedy_knapsack(
        composite_scores=arr["composite_scores"],
        loan_amounts=arr["loan_amounts"],
        budget=budget,
        qubo_mapper=qubo_mapper
    )
    print(" -> [1/4] Greedy Ratio Knapsack: Done.")

    # B. Exact Combinatorial Solver (Global Ground Truth)
    res_exact = solve_exact_qubo(qubo_mapper)
    print(" -> [2/4] Classical Exact Combinatorial Ground State: Done.")

    # C. Classical Exact MILP Solver (scipy.optimize.milp)
    res_milp = solve_classical_milp(
        composite_scores=arr["composite_scores"],
        loan_amounts=arr["loan_amounts"],
        budget=budget,
        group_indicators=arr["group_indicators"],
        fairness_tolerance=0.15,
        qubo_mapper=qubo_mapper
    )
    print(" -> [3/4] Classical Exact MILP Solver (scipy.optimize.milp): Done.")

    # D. Simulated Annealing Heuristic
    res_sa = solve_simulated_annealing(qubo_mapper, max_steps=1200, seed=42)
    print(" -> [4/4] Classical Simulated Annealing (MCMC): Done.")

    # ---------------------------------------------------------
    # STEP 5: Solving via Quantum Variational Algorithms
    # ---------------------------------------------------------
    print("\n[STEP 5/8] Executing Quantum Variational Algorithms (QAOA, CVaR-QAOA, VQE)...")

    # A. QAOA (p=1 layer, standard expectation)
    print(" -> Running Standard QAOA (p=1 layer, COBYLA)...")
    res_qaoa_p1 = solve_qaoa(qubo_mapper, p_layers=1, cvar_alpha=1.0, max_iter=100, seed=42)

    # B. QAOA (p=2 layers, standard expectation)
    print(" -> Running Standard QAOA (p=2 layers, COBYLA)...")
    res_qaoa_p2 = solve_qaoa(qubo_mapper, p_layers=2, cvar_alpha=1.0, max_iter=130, seed=42)

    # C. CVaR-QAOA (p=1 layer, tail risk alpha=0.25)
    print(" -> Running CVaR-QAOA (p=1 layer, alpha=0.25 tail quantile, COBYLA)...")
    res_cvar_qaoa = solve_cvar_qaoa(qubo_mapper, p_layers=1, cvar_alpha=0.25, max_iter=110, seed=42)

    # D. VQE (Hardware-Efficient TwoLocal Ansatz, L=1 layer)
    print(" -> Running VQE (Hardware-Efficient TwoLocal Ansatz, COBYLA)...")
    res_vqe = solve_vqe(qubo_mapper, ansatz_layers=1, max_iter=120, seed=42)

    # ---------------------------------------------------------
    # STEP 6: Multi-Metric Evaluation & Comprehensive Table
    # ---------------------------------------------------------
    print("\n[STEP 6/8] Computing Multi-Objective Fairness & Tail-Risk Metrics across All Solvers...")

    optimal_utility = float(np.sum(df_scored["composite_score"].values * res_exact["allocation"]))

    all_solvers = [
        ("Classical Greedy (No Fairness)", res_greedy["allocation"]),
        ("Classical Exact (Ground Truth)", res_exact["allocation"]),
        ("Classical Exact MILP", res_milp["allocation"]),
        ("Simulated Annealing", res_sa["allocation"]),
        ("Standard QAOA (p=1)", res_qaoa_p1["allocation"]),
        ("Standard QAOA (p=2)", res_qaoa_p2["allocation"]),
        ("Tail-Risk CVaR-QAOA (p=1, alpha=0.25)", res_cvar_qaoa["allocation"]),
        ("Quantum VQE (Hardware-Ansatz)", res_vqe["allocation"])
    ]

    metrics_records = []
    for name, alloc in all_solvers:
        met = evaluate_allocation(
            allocation=alloc,
            df_scored=df_scored,
            budget=budget,
            qubo_mapper=qubo_mapper,
            optimal_utility=optimal_utility,
            solver_name=name,
            risk_engine=risk_engine
        )
        metrics_records.append(met)

    df_comparison = compile_comparison_table(metrics_records)
    print("\n" + "=" * 105)
    print("                        BENCHMARK PERFORMANCE COMPARISON TABLE (70% CHECKPOINT)")
    print("=" * 105)
    display_cols = ["solver", "total_approved", "total_utility", "total_cost", "budget_utilization_pct",
                    "expected_loss", "cvar_95_loss", "demographic_parity_diff", "disparate_impact_ratio", "approximation_ratio"]
    print(df_comparison[display_cols].to_string(index=False))
    print("=" * 105)

    # ---------------------------------------------------------
    # STEP 7: Presentation Visuals Generation
    # ---------------------------------------------------------
    print("\n[STEP 7/8] Generating Presentation-Ready Figures in 'visuals/'...")

    # 1. QUBO Heatmap
    heatmap_path = os.path.join("visuals", "qubo_matrix_heatmap.png")
    plot_qubo_heatmap(qubo_mapper.Q, heatmap_path)
    print(f" -> [1/5] QUBO Heatmap: {heatmap_path}")

    # 2. Solver Comparison Plot
    comp_plot_path = os.path.join("visuals", "quantum_vs_classical_comparison.png")
    plot_quantum_vs_classical(pd.DataFrame(metrics_records), comp_plot_path)
    print(f" -> [2/5] Comparative Multi-Metric Plot: {comp_plot_path}")

    # 3. QAOA, CVaR-QAOA & VQE Convergence Profile
    conv_plot_path = os.path.join("visuals", "qaoa_convergence_profile.png")
    histories = {
        "QAOA (p=1)": res_qaoa_p1["convergence_history"],
        "QAOA (p=2)": res_qaoa_p2["convergence_history"],
        "CVaR-QAOA (p=1, alpha=0.25)": res_cvar_qaoa["convergence_history"],
        "VQE": res_vqe["convergence_history"]
    }
    plot_qaoa_convergence(histories, res_exact["energy"], conv_plot_path)
    print(f" -> [3/5] Convergence Curves: {conv_plot_path}")

    # 4. Pareto Frontier Sweep
    print(" -> Generating Pareto Frontier Sweep (varying lambda_fairness)...")
    pareto_records = []
    for l_f in [0.0, 0.5, 1.0, 1.6, 2.5, 3.5]:
        mapper_sweep = QUBOMapper(
            composite_scores=arr["composite_scores"],
            loan_amounts=arr["loan_amounts"],
            group_indicators=arr["group_indicators"],
            total_budget=budget,
            lambda_budget=lambda_budget,
            lambda_fairness=l_f
        )
        sol_sweep = solve_exact_qubo(mapper_sweep)
        x_sw = sol_sweep["allocation"]
        m_sw = evaluate_allocation(x_sw, df_scored, budget, mapper_sweep, risk_engine=risk_engine)
        pareto_records.append({
            "lambda_fairness": l_f,
            "financial_return": m_sw["financial_return"],
            "demographic_parity_diff": m_sw["demographic_parity_diff"],
            "total_utility": m_sw["total_utility"]
        })
    df_pareto = pd.DataFrame(pareto_records)
    pareto_plot_path = os.path.join("visuals", "fairness_vs_budget_tradeoff.png")
    plot_fairness_vs_budget_tradeoff(df_pareto, pareto_plot_path)
    print(f" -> [4/5] Pareto Frontier Curve: {pareto_plot_path}")

    # 5. Quantum Statevector Measurement Distribution
    dist_plot_path = os.path.join("visuals", "bitstring_probability_distribution.png")
    plot_bitstring_distribution(res_qaoa_p2["probabilities"], res_exact["ground_state_index"], qubo_mapper.N, top_k=10, output_path=dist_plot_path)
    print(f" -> [5/5] Quantum Probability Histogram: {dist_plot_path}")

    # ---------------------------------------------------------
    # STEP 8: Qiskit QAOA Circuit Export
    # ---------------------------------------------------------
    print("\n[STEP 8/8] Exporting Native Qiskit QAOA Circuit...")
    _, circuit_str = generate_qiskit_qaoa_circuit(qubo_mapper, p_layers=1)
    circuit_file = os.path.join("visuals", "qiskit_qaoa_circuit.txt")
    with open(circuit_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Q-FAR: QISKIT QAOA PARAMETERIZED QUANTUM CIRCUIT (p=1 LAYER)\n")
        f.write("=" * 80 + "\n\n")
        f.write(circuit_str)
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("OPENQASM 2.0 / 3.0 EQUIVALENT SPECIFICATION AVAILABLE\n")
        f.write("=" * 80 + "\n")
    print(f" -> Circuit Diagram successfully exported to: {circuit_file}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 90)
    print(f"  [SUCCESS] 70% ACCOMPLISHMENT CHECKPOINT REACHED in {elapsed:.2f} seconds.")
    print("  - Phase 1 (Mathematical Modeling, ML Repayment, Classical Baselines & Exact MILP): 100% DONE")
    print("  - Phase 2 (QAOA, CVaR-QAOA, VQE, Monte Carlo Risk Scenarios & Visuals): 75% DONE")
    print("  - TOTAL WEIGHTED PROGRESS: 40% + 30% = 70.0% COMPLETED")
    print("=" * 90)

    return df_comparison


if __name__ == "__main__":
    run_midsem_pipeline()
