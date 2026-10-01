"""
================================================================================
Q-FAR: Quantum-Assisted Fair Resource Allocation for Financial Inclusion
MID-SEMESTER BENCHMARK & EVALUATION PIPELINE (60% ACCOMPLISHMENT CHECKPOINT)
================================================================================
Executes the Phase 1 & Phase 2 pipeline:
1. Microfinance applicant cohort generation & tri-objective scoring
2. QUBO matrix assembly & Ising spin Hamiltonian mapping
3. Classical baselines execution (Greedy, Exact, Simulated Annealing)
4. Quantum variational algorithms execution (QAOA p=1, QAOA p=2, VQE)
5. Comprehensive metrics evaluation (Fairness, Budget, Utility, Approx Ratio)
6. High-resolution presentation visuals generation
7. Qiskit QAOA circuit export
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
from src.scoring import compute_multiobjective_scores
from src.qubo_builder import QUBOMapper
from src.classical_solvers import (
    solve_greedy_knapsack,
    solve_exact_qubo,
    solve_simulated_annealing
)
from src.quantum_engine import (
    solve_qaoa,
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
    print("=" * 80)
    print("  Q-FAR: QUANTUM-ASSISTED FAIR RESOURCE ALLOCATION IN MICROFINANCE")
    print("  MIDSEM EVALUATION PIPELINE - 60% ACCOMPLISHMENT CHECKPOINT")
    print("=" * 80)

    # Output directories
    os.makedirs("visuals", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: Applicant Cohort Generation (Phase 1)
    # ---------------------------------------------------------
    print("\n[STEP 1/7] Generating Microfinance Borrower Cohort...")
    n_applicants = 8  # Optimal for swift NISQ exact statevector simulation
    budget = 110000.0  # Capital budget (INR)
    df_raw = generate_microfinance_cohort(n_applicants=n_applicants, seed=42)

    data_csv_path = os.path.join("data", "sample_applicants.csv")
    df_raw.to_csv(data_csv_path, index=False)
    print(f" -> Generated {n_applicants} applicants. Total requested capital: INR {df_raw['requested_amount'].sum():,.2f}")
    print(f" -> Available lending budget ceiling: INR {budget:,.2f}")
    print(f" -> Raw applicant cohort saved to: {data_csv_path}")

    # ---------------------------------------------------------
    # STEP 2: Multi-Objective Tri-Scoring (Phase 1)
    # ---------------------------------------------------------
    print("\n[STEP 2/7] Computing Multi-Objective Tri-Scoring...")
    # Weights: 35% Financial Viability, 35% Urgent Need, 30% Social Impact
    weights = (0.35, 0.35, 0.30)
    df_scored, arr = compute_multiobjective_scores(df_raw, weights=weights)

    print(" -> Applicant Profiles & Composite Utility:")
    summary_cols = ["applicant_id", "demographic_group", "requested_amount",
                    "financial_score", "need_index", "social_impact_score", "composite_score"]
    print(df_scored[summary_cols].to_string(index=False))

    # ---------------------------------------------------------
    # STEP 3: QUBO Matrix & Ising Hamiltonian Mapping (Phase 1 & 2)
    # ---------------------------------------------------------
    print("\n[STEP 3/7] Assembling QUBO Matrix & Mapping to Ising Spin Glass...")
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
    print(f" -> Max 2-Qubit Exchange Coupling J_ij: {info['max_coupling_J']:.4f}")

    # ---------------------------------------------------------
    # STEP 4: Solving via Classical Baselines (Phase 1)
    # ---------------------------------------------------------
    print("\n[STEP 4/7] Executing Classical Solvers...")

    # A. Greedy Knapsack Heuristic
    res_greedy = solve_greedy_knapsack(
        composite_scores=arr["composite_scores"],
        loan_amounts=arr["loan_amounts"],
        budget=budget,
        qubo_mapper=qubo_mapper
    )
    print(" -> [1/3] Greedy Ratio Knapsack: Done.")

    # B. Exact Combinatorial Solver (Global Ground Truth)
    res_exact = solve_exact_qubo(qubo_mapper)
    print(" -> [2/3] Classical Exact Combinatorial Ground State: Done.")

    # C. Simulated Annealing Heuristic
    res_sa = solve_simulated_annealing(qubo_mapper, max_steps=1200, seed=42)
    print(" -> [3/3] Classical Simulated Annealing (MCMC): Done.")

    # ---------------------------------------------------------
    # STEP 5: Solving via Quantum Variational Algorithms (Phase 2)
    # ---------------------------------------------------------
    print("\n[STEP 5/7] Executing Quantum Variational Algorithms...")

    # A. QAOA (p=1 layer)
    print(" -> Running QAOA (p=1 layer, COBYLA optimizer)...")
    res_qaoa_p1 = solve_qaoa(qubo_mapper, p_layers=1, max_iter=100, seed=42)

    # B. QAOA (p=2 layers)
    print(" -> Running QAOA (p=2 layers, COBYLA optimizer)...")
    res_qaoa_p2 = solve_qaoa(qubo_mapper, p_layers=2, max_iter=130, seed=42)

    # C. VQE (Hardware-Efficient RealAmplitudes ansatz, L=1 layer)
    print(" -> Running VQE (Hardware-Efficient TwoLocal Ansatz, COBYLA)...")
    res_vqe = solve_vqe(qubo_mapper, ansatz_layers=1, max_iter=120, seed=42)

    # ---------------------------------------------------------
    # STEP 6: Multi-Metric Evaluation & Comparison Table
    # ---------------------------------------------------------
    print("\n[STEP 6/7] Computing Fairness & Efficiency Metrics across All Solvers...")

    # Compute optimal utility from exact solver
    optimal_utility = float(np.sum(df_scored["composite_score"].values * res_exact["allocation"]))

    all_solvers = [
        ("Classical Greedy (No Fairness)", res_greedy["allocation"]),
        ("Classical Exact (Ground Truth)", res_exact["allocation"]),
        ("Simulated Annealing", res_sa["allocation"]),
        ("Quantum QAOA (p=1)", res_qaoa_p1["allocation"]),
        ("Quantum QAOA (p=2)", res_qaoa_p2["allocation"]),
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
            solver_name=name
        )
        metrics_records.append(met)

    df_comparison = compile_comparison_table(metrics_records)
    print("\n" + "=" * 95)
    print("                 BENCHMARK PERFORMANCE COMPARISON TABLE")
    print("=" * 95)
    print(df_comparison.to_string(index=False))
    print("=" * 95)

    # ---------------------------------------------------------
    # STEP 7: Presentation Visuals Generation (Phase 2)
    # ---------------------------------------------------------
    print("\n[STEP 7/7] Generating Presentation-Ready Figures in 'visuals/'...")

    # 1. QUBO Heatmap
    heatmap_path = os.path.join("visuals", "qubo_matrix_heatmap.png")
    plot_qubo_heatmap(qubo_mapper.Q, heatmap_path)
    print(f" -> [1/5] QUBO Heatmap: {heatmap_path}")

    # 2. Solver Comparison Plot
    comp_plot_path = os.path.join("visuals", "quantum_vs_classical_comparison.png")
    plot_quantum_vs_classical(pd.DataFrame(metrics_records), comp_plot_path)
    print(f" -> [2/5] Comparative Multi-Metric Plot: {comp_plot_path}")

    # 3. QAOA & VQE Convergence Profile
    conv_plot_path = os.path.join("visuals", "qaoa_convergence_profile.png")
    histories = {
        "QAOA (p=1)": res_qaoa_p1["convergence_history"],
        "QAOA (p=2)": res_qaoa_p2["convergence_history"],
        "VQE": res_vqe["convergence_history"]
    }
    plot_qaoa_convergence(histories, res_exact["energy"], conv_plot_path)
    print(f" -> [3/5] Convergence Curves: {conv_plot_path}")

    # 4. Pareto Frontier Sweep (Fairness vs Return)
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
        m_sw = evaluate_allocation(x_sw, df_scored, budget, mapper_sweep)
        pareto_records.append({
            "lambda_fairness": l_f,
            "financial_return": m_sw["financial_return"],
            "demographic_parity_diff": m_sw["demographic_parity_diff"],
            "total_utility": m_sw["total_utility"]
        })
    pareto_plot_path = os.path.join("visuals", "fairness_vs_budget_tradeoff.png")
    plot_fairness_vs_budget_tradeoff(pareto_records, pareto_plot_path)
    print(f" -> [4/5] Pareto Frontier Plot: {pareto_plot_path}")

    # 5. Quantum State Measurement Distribution
    dist_plot_path = os.path.join("visuals", "bitstring_probability_distribution.png")
    plot_bitstring_distribution(
        probabilities=res_qaoa_p2["probabilities"],
        ground_state_idx=res_exact["ground_state_index"],
        num_qubits=n_applicants,
        top_k=10,
        output_path=dist_plot_path
    )
    print(f" -> [5/5] Quantum Measurement Distribution: {dist_plot_path}")

    # 6. Qiskit QAOA Circuit Export
    qc, circuit_str = generate_qiskit_qaoa_circuit(qubo_mapper, p_layers=1)
    circuit_txt_path = os.path.join("visuals", "qiskit_qaoa_circuit.txt")
    with open(circuit_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Q-FAR: QISKIT QAOA PARAMETERIZED QUANTUM CIRCUIT (p=1 LAYER)\n")
        f.write("=" * 80 + "\n\n")
        f.write(circuit_str)
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("OPENQASM 2.0 / 3.0 EQUIVALENT SPECIFICATION AVAILABLE\n")
        f.write("=" * 80 + "\n")
    print(f" -> [Bonus] Qiskit QAOA Circuit Diagram saved to: {circuit_txt_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  MIDSEM PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("  STATUS: EXACTLY 60% ACCOMPLISHMENT MILESTONE ACHIEVED")
    print("=" * 80)

    return df_comparison


if __name__ == "__main__":
    run_midsem_pipeline()
