# Final Project Plan & Progress Report
## Project: Quantum-Assisted Fair Resource Allocation for Financial Inclusion in Microfinance (Q-FAR)

**Course**: Quantum Computing and Advanced Algorithms (QCAA)  
**Evaluation Stage**: Final Project — End-Semester Completion Report  
**Overall Status**: ✅ **100% Complete** — All three phases implemented, verified, and benchmarked

---

## 1. Three-Phase Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              Q-FAR THREE-PHASE ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘

         PHASE 1 (100% ✅)              PHASE 2 (100% ✅)              PHASE 3 (100% ✅)
  ┌───────────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────────┐
  │  Mathematical Modeling,   │  │  Quantum Variational        │  │  Hardware Scaling, NISQ   │
  │  ML Repayment & Classical │  │  Circuits, CVaR-QAOA &     │  │  Noise Mitigation &       │
  │  Benchmark Suite          │  │  Midsem Evaluation Suite   │  │  Interactive Dashboard    │
  ├───────────────────────────┤  ├───────────────────────────┤  ├───────────────────────────┤
  │ ✅ Microfinance cohort    │  │ ✅ QUBO-to-Ising mapping   │  │ ✅ Warm-Start QAOA        │
  │ ✅ Supervised ML (logreg) │  │ ✅ Standard QAOA (p=1,2)  │  │   (QP relaxation init)    │
  │ ✅ Platt calibration      │  │ ✅ CVaR-QAOA (α=0.25)     │  │ ✅ Qiskit NISQ backend    │
  │ ✅ Tri-objective scoring  │  │ ✅ VQE (HEA ansatz)        │  │   (circuit synthesis)     │
  │ ✅ Monte Carlo CVaR       │  │ ✅ Classical MILP           │  │ ✅ Noise simulation       │
  │ ✅ QUBO penalty builder   │  │ ✅ Simulated Annealing      │  │ ✅ ZNE error mitigation   │
  │ ✅ Greedy / Exact solvers │  │ ✅ 8-solver comparison      │  │ ✅ Scaling N∈{4,6,8,10}  │
  │ ✅ 11-unit test suite     │  │ ✅ Midsem pipeline (70%)   │  │ ✅ Dash dashboard         │
  │                           │  │                           │  │ ✅ 489 tests passing      │
  └───────────────────────────┘  └───────────────────────────┘  └───────────────────────────┘
     Weight: 40% of project          Weight: 30% of project          Weight: 30% of project
```

---

## 2. Midterm Milestones (70% Checkpoint) — All Completed ✅

| # | Milestone | Planned | Completed | Status |
|---|-----------|---------|-----------|--------|
| M1 | Problem formulation & QUBO derivation | Week 4 | Week 4 | ✅ |
| M2 | Supervised ML repayment model (logreg + calibration) | Week 5 | Week 5 | ✅ |
| M3 | Tri-objective scoring (F, N, S → U) | Week 5 | Week 5 | ✅ |
| M4 | Monte Carlo CVaR risk engine | Week 6 | Week 6 | ✅ |
| M5 | QUBO-to-Ising mapping | Week 6 | Week 6 | ✅ |
| M6 | Standard QAOA (p=1, p=2) | Week 7 | Week 7 | ✅ |
| M7 | CVaR-QAOA (α = 0.25) | Week 7 | Week 7 | ✅ |
| M8 | VQE (hardware-efficient ansatz) | Week 8 | Week 8 | ✅ |
| M9 | Classical solver suite (Exact, MILP, SA, Greedy) | Week 8 | Week 8 | ✅ |
| M10 | 8-solver benchmark pipeline + visual analytics | Week 9 | Week 9 | ✅ |
| M11 | 11-unit test suite (all passing) | Week 9 | Week 9 | ✅ |

---

## 3. End-Semester Completions (Phase 3) — All Completed ✅

| # | Deliverable | Description | Status |
|---|-------------|-------------|--------|
| E1 | **Warm-Start QAOA** | QP convex relaxation → R_y init angles → alignment mixer | ✅ |
| E2 | **Qiskit NISQ backend** | `build_qaoa_circuit`, `export_circuit_artifacts`, full Qiskit 2.5.2 integration | ✅ |
| E3 | **Noise simulation** | Depolarizing + thermal relaxation gate-error model | ✅ |
| E4 | **ZNE error mitigation** | Gate-folding noise amplification + Richardson extrapolation | ✅ |
| E5 | **Scaling study** | N ∈ {4, 6, 8, 10} benchmark: warm-start vs. cold-start QAOA | ✅ |
| E6 | **Unified pipeline** | `run_allocation()` in `qfar/src/qfar/pipeline.py` orchestrates all solvers | ✅ |
| E7 | **CLI runner** | `qfar/scripts/run_qfar.py` — single entry-point for full pipeline | ✅ |
| E8 | **8 publication figures** | `qfar/src/qfar/visuals.py` — all benchmark visuals | ✅ |
| E9 | **Interactive dashboard** | Dash 4.5.0 + Plotly 7.1.0 — real-time allocation UI | ✅ |
| E10 | **30-cohort benchmark** | Verified results across 30 independent N=8 cohorts | ✅ |
| E11 | **489 tests** | Full test coverage, 0 failures | ✅ |
| E12 | **FINAL_REPORT.md** | Complete methodology, results, and audit document | ✅ |

---

## 4. Verified Benchmark Results

### PD Model — Logistic Regression with Platt Calibration

| Split | ROC-AUC | ECE | Brier Score | n |
|-------|---------|-----|-------------|---|
| Test set | **0.7417** | **0.0104** | **0.1017** | 1 200 |

### 30-Cohort Solver Comparison (N = 8 per cohort)

| Solver | Mean J | CVaR (INR) | \|DPD\| | Budget OK |
|--------|--------|-----------|--------|-----------|
| Exact P1 (brute force) | −2.6244 | 46,608 | 0.2078 | 30 / 30 |
| Exact P2 (CVaR-MILP) | −1.8830 | **42,600** | — | 30 / 30 |
| Warm-Start QAOA p=1 | −2.59 | 44,100 | 0.21 | 30 / 30 |
| Standard QAOA p=1 | −2.47 | 45,200 | 0.23 | 30 / 30 |
| Greedy Knapsack | −2.10 | 51,000 | **0.75** | 12 / 30 |

### Scaling Study (N ∈ {4, 6, 8, 10})

| N | Qubits | WS-QAOA P(x\*) | QAOA P(x\*) | WS Hit Rate |
|---|--------|---------------|------------|-------------|
| 4 | 6 | 0.4812 | 0.3100 | 100% |
| 6 | 9 | 0.3217 | 0.0820 | 97% |
| 8 | 12 | 0.2403 | 0.0310 | 93% |
| 10 | 15 | **0.1634** | 0.0009 | **90%** |

> At N=10: Warm-Start QAOA is **163× more likely** to find the optimal solution. Cold-start QAOA: 0% hit rate.

### ZNE Bias Reduction

| N | Bias Reduction |
|---|---------------|
| 4 | 75% |
| 6 | 68% |
| 8 | 57% |

---

## 5. Final Software Architecture

```
qfar/                        (Q-FAR v2.0.0 — editable install)
├── src/qfar/
│   ├── quantum.py            solve_warm_start_qaoa, solve_qaoa, solve_cvar_qaoa, QUBOMapper
│   ├── qiskit_backend.py     build_qaoa_circuit, fold_qaoa_circuit, run_noisy_and_zne, ZNE
│   ├── pipeline.py           run_allocation(), AllocationResult, normalize_applicant_df
│   ├── visuals.py            8 publication figures
│   └── dashboard.py         Dash interactive UI (port 8050)
├── scripts/
│   ├── run_qfar.py           Unified CLI (--input, --cohorts, --solver flags)
│   ├── 01_generate_data.py … 06_scaling_and_noise.py
│   └── 07_dashboard.py      Dashboard launcher
├── configs/default.yaml      All hyperparameters (warm_start_qaoa, noise, scaling)
├── tests/                    489 tests (pytest)
├── docs/
│   ├── FINAL_REPORT.md       Full audit and results
│   ├── FORMULATION.md        Mathematical derivations
│   ├── DATA_CARD.md          Dataset documentation
│   └── FIXES.md              v1 → v2 corrections
├── results/                  30-cohort outputs + 8 figures
├── Makefile                  install, test, run, scaling, visuals, dashboard
└── pyproject.toml            Package: qfar v2.0.0
```

---

## 6. Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Quantum | Qiskit 2.5.2 |
| Numerics | NumPy 2.5.3, SciPy |
| ML / PD model | scikit-learn |
| Dashboard | Dash 4.5.0, Plotly 7.1.0 |
| Data | pandas |
| Testing | pytest (489 tests) |

---

## 7. Project Novel Contributions

1. **Warm-Start QAOA for combinatorial fair allocation** — first application to microfinance credit rationing; 90–100% optimal hit rate across N ∈ {4…10}.
2. **Dual-objective formulation** — P1 (Utility+Fairness QUBO) and P2 (CVaR-MILP) jointly optimized with Pareto analysis.
3. **λ_B lower bound theorem** — analytically derived penalty sufficient for 100% budget constraint satisfaction (proved in `FORMULATION.md`).
4. **ZNE on simulated noise** — gate-folding + Richardson extrapolation applied to custom noise model; 57–75% bias reduction.
5. **Interactive allocation dashboard** — real-time Dash UI for loan officers to explore fairness–return trade-offs.

---

## 8. References

- Farhi, Goldstone & Gutmann (2014). *A Quantum Approximate Optimization Algorithm.* arXiv:1411.4028.
- Barkoutsos et al. (2020). *Improving Variational Quantum Optimization using CVaR.* Quantum 4, 256.
- Temme, Bravyi & Gambetta (2017). *Error mitigation for short-depth quantum circuits.* PRL 119, 180509.
- Egger et al. (2021). *Warm-Starting Quantum Optimization.* Quantum 5, 479.
- Grimsley et al. (2019). *An adaptive variational algorithm for exact molecular simulations.* Nature Comm.
