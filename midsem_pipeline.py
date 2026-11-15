"""
================================================================================
Q-FAR: Quantum-Assisted Fair Resource Allocation for Financial Inclusion
MID-SEMESTER PROTOTYPE PIPELINE (v1 — 70% Accomplishment Checkpoint)
================================================================================

NOTE: This file is the v1 midsem prototype retained for historical reference.
      The final, production-grade pipeline is in:

        qfar/scripts/run_qfar.py          ← unified CLI runner
        qfar/src/qfar/pipeline.py         ← run_allocation() entry-point
        qfar/src/qfar/quantum.py          ← QAOA / Warm-Start QAOA / CVaR-QAOA
        qfar/src/qfar/qiskit_backend.py   ← Qiskit circuit synthesis + ZNE

      To run the final pipeline:
        cd qfar
        pip install -e ".[full]"
        python scripts/run_qfar.py

      To use programmatically:
        from qfar import run_allocation, AllocationResult
        result: AllocationResult = run_allocation(df, budget=50_000)

================================================================================
WHAT THIS MIDSEM PROTOTYPE DOES (v1):
  1. Microfinance cohort generation & AI repayment prediction (Calibrated ML)
  2. Multi-objective tri-scoring (Financial Return, Urgent Need, Social Inclusion)
  3. Stochastic Monte Carlo scenario generation & discrete tail-risk (VaR/CVaR)
  4. QUBO matrix assembly & Ising spin Hamiltonian mapping
  5. Classical solvers: Greedy, Exact Combinatorial, Exact MILP, Simulated Annealing
  6. Quantum algorithms: QAOA (p=1,2), CVaR-QAOA (alpha=0.25), VQE
  7. Multi-metric evaluation (Utility, Budget, CVaR, DPD, DIR, Approx Ratio)
  8. Visualisations & Qiskit circuit export
================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# NOTE: v1 imports — these reference the root-level src/ prototype modules.
# The final implementation uses qfar/src/qfar/* instead.
# ---------------------------------------------------------------------------

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.data_generator import generate_microfinance_cohort
    from src.repayment_model import RepaymentPredictor, prepare_features
    from src.scoring import compute_multiobjective_scores
    from src.risk_engine import RiskEngine
    from src.qubo_builder import QUBOMapper
    from src.classical_solvers import (
        solve_greedy_knapsack,
        solve_exact_qubo,
        solve_classical_milp,
        solve_simulated_annealing,
    )
    from src.quantum_engine import (
        solve_qaoa,
        solve_cvar_qaoa,
        solve_vqe,
        generate_qiskit_qaoa_circuit,
    )
    from src.metrics import evaluate_allocation, compile_comparison_table
    from src.visualizer import (
        plot_qubo_heatmap,
        plot_quantum_vs_classical,
        plot_qaoa_convergence,
        plot_fairness_vs_budget_tradeoff,
        plot_bitstring_distribution,
    )
    _V1_AVAILABLE = True
except ImportError:
    _V1_AVAILABLE = False


def run_midsem_pipeline():
    """
    Execute the v1 midsem benchmark pipeline.

    For the final Q-FAR v2 pipeline, use:
        from qfar import run_allocation
        result = run_allocation(df, budget=50_000)
    """
    if not _V1_AVAILABLE:
        print(
            "\n[NOTICE] v1 prototype modules (src/) not importable in this environment.\n"
            "         Please use the final Q-FAR v2 pipeline:\n\n"
            "           cd qfar && pip install -e '.[full]'\n"
            "           python scripts/run_qfar.py\n"
        )
        return

    start_time = time.time()
    print("=" * 90)
    print("  Q-FAR: QUANTUM-ASSISTED FAIR RESOURCE ALLOCATION IN MICROFINANCE")
    print("  MIDSEM EVALUATION PIPELINE — v1 PROTOTYPE (70% ACCOMPLISHMENT CHECKPOINT)")
    print("=" * 90)
    print()

    # ── 1. Generate cohort ───────────────────────────────────────────────────
    print("[1/8] Generating microfinance applicant cohort …")
    rng = np.random.default_rng(42)
    df = generate_microfinance_cohort(n=8, seed=42)
    print(f"      Generated {len(df)} applicants.")

    # ── 2. Repayment prediction ──────────────────────────────────────────────
    print("[2/8] Training repayment predictor …")
    predictor = RepaymentPredictor()
    X = prepare_features(df)
    predictor.fit(X, df["repaid"].values)
    df["p_repay"] = predictor.predict_proba(X)
    print(f"      PD model fitted. Sample probabilities: {df['p_repay'].values[:4]}")

    # ── 3. Scoring ───────────────────────────────────────────────────────────
    print("[3/8] Computing tri-objective scores …")
    df = compute_multiobjective_scores(df)

    # ── 4. Risk engine ───────────────────────────────────────────────────────
    print("[4/8] Running Monte Carlo tail-risk engine …")
    risk_engine = RiskEngine(n_scenarios=2500, alpha=0.95, seed=42)
    df["cvar_contribution"] = risk_engine.compute_cvar_contributions(df)

    # ── 5. QUBO ──────────────────────────────────────────────────────────────
    print("[5/8] Building QUBO matrix …")
    budget = int(df["loan_amount"].sum() * 0.6)
    mapper = QUBOMapper(lambda_budget=10.0, lambda_fairness=5.0)
    Q = mapper.build(df, budget=budget)
    print(f"      QUBO shape: {Q.shape}, budget: {budget:,} INR")

    # ── 6. Classical solvers ─────────────────────────────────────────────────
    print("[6/8] Running classical solvers …")
    results = {}
    results["Greedy"]  = solve_greedy_knapsack(df, budget)
    results["Exact"]   = solve_exact_qubo(Q, df, budget)
    results["MILP"]    = solve_classical_milp(df, budget)
    results["SA"]      = solve_simulated_annealing(Q, df, budget, seed=42)

    # ── 7. Quantum solvers ───────────────────────────────────────────────────
    print("[7/8] Running quantum solvers …")
    results["QAOA_p1"]     = solve_qaoa(Q, df, budget, p=1, seed=42)
    results["QAOA_p2"]     = solve_qaoa(Q, df, budget, p=2, seed=42)
    results["CVaR_QAOA"]   = solve_cvar_qaoa(Q, df, budget, alpha=0.25, seed=42)
    results["VQE"]         = solve_vqe(Q, df, budget, seed=42)

    # ── 8. Evaluation ────────────────────────────────────────────────────────
    print("[8/8] Evaluating and printing results …")
    table = compile_comparison_table(results, df, budget, risk_engine)
    print()
    print(table.to_string(index=False))
    print()
    print(f"Pipeline complete in {time.time() - start_time:.1f}s.")
    print()
    print("─" * 90)
    print("  For the FINAL Q-FAR v2 pipeline with Warm-Start QAOA, Qiskit NISQ backend,")
    print("  ZNE noise mitigation, and interactive dashboard, run:")
    print()
    print("    cd qfar && pip install -e '[full]' && python scripts/run_qfar.py")
    print("─" * 90)


if __name__ == "__main__":
    run_midsem_pipeline()
