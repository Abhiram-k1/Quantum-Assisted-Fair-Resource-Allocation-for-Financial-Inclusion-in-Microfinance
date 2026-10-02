# Comprehensive Algorithmic Specification & Pipeline Architecture (`pseudo.md`)
## Project: Quantum-Assisted Fair Resource Allocation for Financial Inclusion in Microfinance (Q-FAR)
**Course**: Quantum Computing and Advanced Algorithms (QCAA)  
**Document Standard**: High-Performance Computing (HPC), Computer Vision (CV) & Quantum Systems Architecture  
**Evaluation Milestone**: 70% Accomplishment Milestone Checkpoint (Verified across 8 Solvers)  
**Methodological Paradigm**: Explicit **WHY, WHAT, and HOW** Unified Framework  

---

## Master Table of Contents
1. [Section 1: Master Architectural Paradigm — The WHY, WHAT, and HOW of Q-FAR](#1-master-architectural-paradigm--the-why-what-and-how-of-q-far)
   - [1.1 The WHY: Foundational Problem, Real-World Crisis & Scientific Gaps](#11-the-why-foundational-problem-real-world-crisis--scientific-gaps)
   - [1.2 The WHAT: System Scope, Input-Output Contract & Target Deliverables](#12-the-what-system-scope-input-output-contract--target-deliverables)
   - [1.3 The HOW: End-to-End Computational Pipeline & Mathematical Architecture](#13-the-how-end-to-end-computational-pipeline--mathematical-architecture)
   - [1.4 Algorithmic Step Quick Reference Index](#14-algorithmic-step-quick-reference-index)
2. [Section 2: Granular Algorithmic Specifications (WHY, WHAT, HOW)](#2-granular-algorithmic-specifications-why-what-how)
   - [Algorithm 1: Demographic & Microfinance Cohort Generation](#algorithm-1-demographic--microfinance-cohort-generation)
   - [Algorithm 2: Supervised AI Repayment Modeling & Probability Calibration](#algorithm-2-supervised-ai-repayment-modeling--probability-calibration)
   - [Algorithm 3: Multi-Objective Tri-Scoring & Welfare Synthesis](#algorithm-3-multi-objective-tri-scoring--welfare-synthesis)
   - [Algorithm 4: Monte Carlo Tail-Risk Engine (VaR & CVaR Simulation)](#algorithm-4-monte-carlo-tail-risk-engine-var--cvar-simulation)
   - [Algorithm 5: Multi-Objective Constrained Knapsack to QUBO Mapping](#algorithm-5-multi-objective-constrained-knapsack-to-qubo-mapping)
   - [Algorithm 6: QUBO to Ising Spin Glass Transformation](#algorithm-6-qubo-to-ising-spin-glass-transformation)
   - [Algorithm 7: Classical Benchmark Solvers Suite (Greedy, Exact, MILP, SA)](#algorithm-7-classical-benchmark-solvers-suite)
   - [Algorithm 8: Standard Quantum Approximate Optimization Algorithm (QAOA)](#algorithm-8-standard-quantum-approximate-optimization-algorithm-qaoa)
   - [Algorithm 9: Tail-Risk Conditional Value-at-Risk QAOA (CVaR-QAOA)](#algorithm-9-tail-risk-conditional-value-at-risk-qaoa-cvar-qaoa)
   - [Algorithm 10: Variational Quantum Eigensolver (VQE) with Hardware-Efficient Ansatz](#algorithm-10-variational-quantum-eigensolver-vqe-with-hardware-efficient-ansatz)
   - [Algorithm 11: Native Qiskit Circuit Synthesis & Gate Decomposition](#algorithm-11-native-qiskit-circuit-synthesis--gate-decomposition)
   - [Algorithm 12: Multi-Objective Fairness & Tail-Risk Evaluation](#algorithm-12-multi-objective-fairness--tail-risk-evaluation)
   - [Algorithm 13: End-to-End Benchmark Pipeline Orchestrator](#algorithm-13-end-to-end-benchmark-pipeline-orchestrator)
3. [Section 3: Master Algorithm Complexity & Hardware Resource Matrix](#3-master-algorithm-complexity--hardware-resource-matrix)

---

# 1. Master Architectural Paradigm — The WHY, WHAT, and HOW of Q-FAR

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       THE Q-FAR CORE METHODOLOGICAL TRIAD                                              │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  1. THE WHY (The Purpose)         2. THE WHAT (The System)            3. THE HOW (The Machinery)                      │
│  - Capital Rationing Dilemma     - Supervised AI Risk Engine         - Multi-Objective Normalization                  │
│  - Commercial "Mission Drift"    - Monte Carlo Tail-Risk (CVaR)      - Constrained-to-QUBO Penalty Mapping            │
│  - Systemic Demographic Bias     - Knapsack-to-Ising Compiler        - Affine Pauli-Z Spin Operator Transformation    │
│  - Severe Tail Default Catastrophe- 8 Classical & Quantum Solvers   - Quantum Variational Ansätze (QAOA/VQE)         │
│  - Combinatorial NP-Hard Scalability- Rigorous Multi-Metric Auditor   - Tail-Energy CVaR Optimization (alpha=0.25)     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.1 The WHY: Foundational Problem, Real-World Crisis & Scientific Gaps

### 1. The Microfinance Credit Rationing Crisis
Microfinance institutions (MFIs) serve unbanked, low-income, and marginalized communities who lack formal credit scores (FICO/CIBIL) and physical collateral. In real-world microcredit operations, **loan demand invariably exceeds available lendable capital**:
$$\sum_{i=1}^N w_i > B$$
Because capital is strictly bounded by budget $B$, the institution cannot approve every applicant. It must decide who receives credit and who is rejected.

### 2. The Trap of "Mission Drift" and Demographic Disparity
When microfinance institutions adopt standard commercial credit algorithms or naive greedy heuristics, they optimize exclusively for immediate financial repayment:
$$\max \sum_{i=1}^N \text{Profit}_i \cdot x_i$$
This produces catastrophic socio-economic failure known in development economics as **"mission drift"**:
- Lower-income, rural, and historically marginalized borrowers (Group A) have higher debt-to-income (DTI) ratios and smaller cash reserves.
- Standard profit maximization systematically approves wealthier, urban borrowers (Group B) and rejects marginalized borrowers.
- In our empirical baseline, greedy heuristics produce a **Demographic Parity Difference of 75%** and a **Disparate Impact Ratio of 0.25**, severely violating the internationally recognized **Four-Fifths (80%) Rule** of fair lending.

### 3. The Fallacy of Expected Loss and the Need for Tail-Risk (CVaR)
Traditional credit risk models rely on Expected Loss:
$$\mathbb{E}[\text{Loss}] = \sum_{i=1}^N w_i (1 - p_i) x_i$$
However, in emerging markets, microfinance defaults do not occur independently—they occur in **clustered, systemic waves** caused by localized monsoon floods, macroeconomic inflation, or crop disease shocks. 
- While average losses may appear manageable (e.g., INR 5,000), a 95% tail default event can wipe out an MFI's entire liquidity reserve (e.g., INR 30,000+).
- **Conditional Value-at-Risk ($\text{CVaR}_{0.95}$)** is mathematically proven to be a **coherent risk measure** (subadditive and convex) that explicitly quantifies the expected loss in the worst 5% of crisis scenarios.

### 4. Why Quantum Variational Optimization?
Embedding fairness balance terms across demographic groups transforms the weakly NP-complete 0-1 Knapsack problem into a **Quadratic Knapsack Problem (QKP)**:
$$\min_x \left( \frac{1}{|G_A|} \sum_{i \in G_A} x_i - \frac{1}{|G_B|} \sum_{j \in G_B} x_j \right)^2 \implies \sum_{i, j} \sigma_i \sigma_j x_i x_j$$
- Quadratic Knapsack is **strongly NP-hard**.
- Exact classical solvers (Branch-and-Bound / MILP) require linearizing all $\mathcal{O}(N^2)$ cross-terms using auxiliary continuous variables, encountering combinatorial bottlenecks as $N$ scales.
- Noisy Intermediate-Scale Quantum (NISQ) algorithms—specifically the **Quantum Approximate Optimization Algorithm (QAOA)** and the **Variational Quantum Eigensolver (VQE)**—exploit quantum superposition across the $2^N$ Hilbert state space, quantum entanglement, and quantum tunneling through barrier-laden penalty landscapes to find high-utility ground states.

---

## 1.2 The WHAT: System Scope, Input-Output Contract & Target Deliverables

### 1. Concrete System Inputs
The Q-FAR engine ingests a cohort of $N$ microfinance loan applicants, represented by a multidimensional feature matrix:
$$\mathbf{X} \in \mathbb{R}^{N \times d}, \quad \vec{w} \in \mathbb{R}^N, \quad \vec{g} \in \{G_A, G_B\}^N$$
- **Financial Features**: Monthly income ($I_i$), debt-to-income ratio ($\text{DTI}_i$), historical repayment discipline ($H_i$), requested loan amount ($w_i$).
- **Social & Humanitarian Features**: Multidimensional poverty index ($P_i$), dependent family members ($D_i$), women-led micro-enterprise flag ($W_i$).
- **Institutional Parameters**: Total lending budget ceiling $B$, fairness penalty multiplier $\lambda_F$, budget penalty multiplier $\lambda_B$, multi-objective weights $\vec{\omega} = (\omega_F, \omega_N, \omega_S)$.

### 2. The 6 Core Computational Subsystems
1. **Supervised AI Probability Calibrator (`src/repayment_model.py`)**: Predicts true posterior repayment probabilities $p_i = \mathbb{P}(\text{Repay}=1 \mid \vec{x}_i)$ using Platt/Isotonic calibrated classifiers with strict demographic feature isolation.
2. **Multi-Objective Tri-Scoring Engine (`src/scoring.py`)**: Synthesizes Financial Viability ($F_i$), Urgent Need ($N_i$), and Social Impact ($S_i$) into a convex composite utility score $C_i \in [0, 1]$.
3. **Stochastic Monte Carlo Risk Simulator (`src/risk_engine.py`)**: Simulates $S = 2,500$ joint default scenarios to evaluate Portfolio Value-at-Risk ($\text{VaR}_{0.95}$) and Conditional Value-at-Risk ($\text{CVaR}_{0.95}$).
4. **QUBO & Ising Hamiltonian Compiler (`src/qubo_builder.py`)**: Encodes knapsack budget and demographic parity quadratic constraints into an unconstrained matrix $Q \in \mathbb{R}^{N \times N}$ and affine Pauli-Z spin Hamiltonian $H_C$.
5. **8-Solver Comparative Benchmark Suite**:
   - Classical Solvers: Greedy Heuristic, Exact Brute Force, Exact MILP (`scipy.optimize.milp`), Simulated Annealing (MCMC).
   - Quantum Variational Solvers: Standard QAOA ($p=1$), Standard QAOA ($p=2$), Tail-Risk CVaR-QAOA ($\alpha=0.25$), Hardware-Efficient VQE ($L=1$).
6. **Multi-Metric Fairness & Financial Auditor (`src/metrics.py`)**: Audits all solutions across 10 evaluation dimensions including Demographic Parity Difference (DPD), Disparate Impact Ratio (DIR), budget utilization, and approximation ratio $\alpha$.

### 3. Concrete System Outputs & Deliverables
- **Optimal Allocation Vector**: Binary decision bitstring $x^* = (x_1^*, \dots, x_N^*) \in \{0, 1\}^N$, indicating approved ($1$) or rejected ($0$) loans.
- **Published Benchmark Comparison Table**: Formatted multi-metric table comparing all 8 solvers on identical inputs.
- **5 High-Resolution Presentation Visuals**:
  1. `visuals/qubo_matrix_heatmap.png`: Pairwise budget and fairness interaction energy matrix.
  2. `visuals/quantum_vs_classical_comparison.png`: 4-panel multi-metric comparative bar chart.
  3. `visuals/qaoa_convergence_profile.png`: Variational energy optimization trajectory across iterations.
  4. `visuals/fairness_vs_budget_tradeoff.png`: Pareto frontier mapping utility cost vs demographic equity.
  5. `visuals/bitstring_probability_distribution.png`: Sampled quantum computational basis state distribution.
- **Hardware-Ready Qiskit Circuit Diagram**: `visuals/qiskit_qaoa_circuit.txt`, showing gate decomposition into native $H$, $R_Z$, $R_{ZZ}$, and $R_X$ gates.

---

## 1.3 The HOW: End-to-End Computational Pipeline & Mathematical Architecture

### High-Level Dataflow Diagram
```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             Q-FAR END-TO-END COMPUTATIONAL PIPELINE                                     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                            │
    [STEP 1] Data Generation & Feature Ingestion            │  Algorithm 1: Demographic & Credit Cohort Simulation
             (Demographics, income, DTI, loan requests)     ▼  (src/data_generator.py)
    [STEP 2] Supervised AI Repayment Modeling               │  Algorithm 2: Supervised Probability Calibration
             (CalibratedClassifierCV, Platt/Isotonic)       ▼  (src/repayment_model.py)
    [STEP 3] Multi-Objective Tri-Scoring                   │  Algorithm 3: Multi-Objective Utility Synthesis
             (Financial F_i, Need N_i, Social S_i)          ▼  (src/scoring.py)
    [STEP 4] Stochastic Monte Carlo Risk Engine             │  Algorithm 4: Portfolio Tail-Risk Simulation (VaR/CVaR)
             (2,500 default scenarios under uncertainty)    ▼  (src/risk_engine.py)
    [STEP 5] QUBO Assembly & Penalty Engineering            │  Algorithm 5: Constrained Knapsack to QUBO Mapping
             (Budget capacity + Demographic parity terms)   ▼  (src/qubo_builder.py)
    [STEP 6] Ising Spin Hamiltonian Mapping                 │  Algorithm 6: Pauli-Z Spin Glass Transformation
             (Affine substitution x_i = (I - Z_i)/2)        ▼  (src/qubo_builder.py)
                                                            │
                            ┌───────────────────────────────┴───────────────────────────────┐
                            ▼                                                               ▼
            [CLASSICAL BENCHMARK SOLVERS]                                   [QUANTUM VARIATIONAL ENGINES]
    Algorithm 7A: Greedy Ratio Knapsack (src/classical_solvers.py)     Algorithm 8: Standard QAOA (src/quantum_engine.py)
    Algorithm 7B: Exact Combinatorial (src/classical_solvers.py)       Algorithm 9: Tail-Risk CVaR-QAOA (src/quantum_engine.py)
    Algorithm 7C: Exact Classical MILP (src/classical_solvers.py)      Algorithm 10: Hardware VQE (src/quantum_engine.py)
    Algorithm 7D: Simulated Annealing (src/classical_solvers.py)       Algorithm 11: Qiskit Circuit Synthesis (src/quantum_engine.py)
                            │                                                               │
                            └───────────────────────────────┬───────────────────────────────┘
                                                            ▼
    [STEP 7] Multi-Metric Auditing & Fairness Analysis      │  Algorithm 12: Comprehensive Fairness & Risk Evaluation
             (DPD, DIR, CVaR 95%, Approx Ratio alpha)       ▼  (src/metrics.py)
    [STEP 8] End-to-End Benchmark Orchestration             │  Algorithm 13: Pipeline Execution & Visualization
             (Automated comparative evaluation)             ▼  (midsem_pipeline.py)
```

### Mathematical Translation Pipeline
The computational core translates high-level social and economic policies into physical quantum states in 4 rigorous mathematical steps:

#### Stage A: Constrained Mathematical Program
$$\max_{x \in \{0, 1\}^N} \sum_{i=1}^N C_i x_i \quad \text{s.t.} \quad \sum_{i=1}^N w_i x_i \le B, \quad \left| \frac{\sum_{i \in G_A} x_i}{|G_A|} - \frac{\sum_{j \in G_B} x_j}{|G_B|} \right| \le \epsilon$$

#### Stage B: Unconstrained Quadratic Formulation (QUBO)
Convert inequalities and equality penalties into squared energy penalties:
$$E(x) = -\sum_{i=1}^N C_i x_i + \lambda_B \left(\sum_{i=1}^N w_i x_i - B\right)^2 + \lambda_F \left(\sum_{i=1}^N \sigma_i x_i\right)^2$$
Expanding and applying the idempotent identity $x_i^2 = x_i$ yields the standard upper-triangular QUBO matrix $Q \in \mathbb{R}^{N \times N}$:
$$E(x) = \sum_{i=1}^N Q_{ii} x_i + \sum_{i < j} Q_{ij} x_i x_j + E_{\text{offset}} = x^T Q x + E_{\text{offset}}$$

#### Stage C: Quantum Spin-1/2 Ising Transformation
Substitute binary decision variables $x_i$ with Pauli-Z spin operators $Z_i = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$:
$$x_i = \frac{I - Z_i}{2}$$
This maps the classical quadratic energy to a physical diagonal Ising Hamiltonian:
$$H_C = \sum_{i=1}^N h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \text{offset} \cdot I$$
where:
$$J_{ij} = \frac{1}{4} Q_{ij}, \quad h_i = -\frac{1}{2} Q_{ii} - \frac{1}{4} \left( \sum_{j > i} Q_{ij} + \sum_{j < i} Q_{ji} \right)$$

#### Stage D: Quantum Variational Circuit Execution
1. **QAOA**: Prepares uniform superposition $|+\rangle^{\otimes N}$, then alternates $p$ layers of problem evolution $U_C(\gamma) = e^{-i \gamma H_C}$ and transverse mixer evolution $U_M(\beta) = e^{-i \beta \sum X_i}$:
   $$|\psi(\vec{\gamma}, \vec{\beta})\rangle = \prod_{k=1}^p e^{-i \beta_k \sum_{i=1}^N X_i} e^{-i \gamma_k H_C} |+\rangle^{\otimes N}$$
2. **CVaR-QAOA**: Replaces standard expectation loss with the **Conditional Value-at-Risk** $\text{CVaR}_\alpha$ over the lowest $\alpha = 0.25$ quantile of the energy distribution, eliminating high-energy infeasible tail states from biasing the classical optimizer.
3. **Classical Optimization Loop**: A classical optimizer (COBYLA) iteratively tunes variational angles $(\vec{\gamma}, \vec{\beta})$ or ansatz parameters $\vec{\theta}$ until the ground state energy is minimized.
4. **Basis Sampling**: Measuring the final quantum statevector yields probability distribution $P(x) = |\langle x | \psi \rangle|^2$. The maximum-likelihood bitstring $x^* = \arg\max P(x)$ is extracted as the optimal loan allocation.

---

## 1.4 Algorithmic Step Quick Reference Index

| Step | Algorithm Name | Primary Role | Paradigm |
|:---:|---|---|---|
| **Alg 1** | [Demographic & Microfinance Cohort Generation](#algorithm-1-demographic--microfinance-cohort-generation) | Synthesizes realistic applicant records | Statistical Sampling |
| **Alg 2** | [Supervised AI Repayment Modeling](#algorithm-2-supervised-ai-repayment-modeling--probability-calibration) | Predicts calibrated posterior probabilities $p_i$ | Calibrated ML |
| **Alg 3** | [Multi-Objective Tri-Scoring](#algorithm-3-multi-objective-tri-scoring--welfare-synthesis) | Synthesizes financial, need & social welfare $C_i$ | Convex Synthesis |
| **Alg 4** | [Monte Carlo Tail-Risk Engine](#algorithm-4-monte-carlo-tail-risk-engine-var--cvar-simulation) | Evaluates $\text{VaR}_{0.95}$ & $\text{CVaR}_{0.95}$ losses | Stochastic MCMC |
| **Alg 5** | [Knapsack to QUBO Mapping](#algorithm-5-multi-objective-constrained-knapsack-to-qubo-mapping) | Compiles constraints to quadratic penalty matrix $Q$ | Penalty Compilation |
| **Alg 6** | [QUBO to Ising Spin Transformation](#algorithm-6-qubo-to-ising-spin-glass-transformation) | Maps binary variables to Pauli-Z spin operators | Affine Spin Mapping |
| **Alg 7** | [Classical Benchmark Solvers Suite](#algorithm-7-classical-benchmark-solvers-suite) | Executes Greedy, Exact, MILP, and SA baselines | Classical Optimization |
| **Alg 8** | [Standard QAOA](#algorithm-8-standard-quantum-approximate-optimization-algorithm-qaoa) | Solves Ising Hamiltonian via $p$-layer phase/mixer | Quantum Variational |
| **Alg 9** | [Tail-Risk CVaR-QAOA](#algorithm-9-tail-risk-conditional-value-at-risk-qaoa-cvar-qaoa) | Filters tail states via $\alpha$-quantile expectation | Risk-Aware Quantum |
| **Alg 10**| [Hardware-Efficient VQE](#algorithm-10-variational-quantum-eigensolver-vqe-with-hardware-efficient-ansatz) | Optimizes native $R_y(\theta)$ + CNOT ansatz | Device-Native Quantum |
| **Alg 11**| [Qiskit Circuit Synthesis](#algorithm-11-native-qiskit-circuit-synthesis--gate-decomposition) | Compiles to native hardware basis gates | Circuit Compilation |
| **Alg 12**| [Fairness & Risk Auditing](#algorithm-12-multi-objective-fairness--tail-risk-evaluation) | Vectorized auditing of DPD, DIR, CVaR, ROI | Empirical Verification |
| **Alg 13**| [End-to-End Benchmark Orchestrator](#algorithm-13-end-to-end-benchmark-pipeline-orchestrator) | Runs 8-solver benchmark & exports 5 visuals | Orchestration |

---

# 2. Granular Algorithmic Specifications (WHY, WHAT, HOW)

---

### Algorithm 1: Demographic & Microfinance Cohort Generation
* **Target Module**: [`src/data_generator.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/data_generator.py)
* **Pipeline Stage**: Step 1 of 13

#### 1. 💡 WHY We Need This Step (Motivation & Problem Context)
1. **Financial Privacy & Data Scarcity**: Real-world microcredit datasets with verified protected demographic attributes are heavily restricted under banking privacy regulations (GDPR, Fair Lending Acts).
2. **Controlled Benchmark Environments**: To rigorously test quantum optimization across scaling instances ($N = 4, 6, 8, 10, \dots, 50$), we require an exact, reproducible data generator where group imbalances, poverty levels, and loan sizes can be systematically varied.
3. **Preventing Algorithmic Blindness**: Without demographic tags in synthetic data, it is impossible to evaluate Demographic Parity Difference (DPD) or Disparate Impact Ratio (DIR).

#### 2. 🔍 WHAT This Step Does (Specification & Input-Output Mapping)
A statistical cohort generator that synthesizes realistic, heterogeneous microfinance applicant profiles, producing financial attributes (income, debt-to-income ratio, credit history), social attributes (poverty index, dependents, women-led enterprise status), and protected demographic labels ($G_A$: Rural/Marginalized, $G_B$: Urban/General).

#### 3. ⚙️ HOW It Works (Mathematical Engine & Mechanics)
1. Demographic group labels are sampled from a Bernoulli/Categorical prior: $g_i \sim \text{Categorical}([p_A, 1 - p_A])$.
2. Conditioned on group $g_i$, income is drawn from a uniform interval $I_i \sim \mathcal{U}(I_{\min}^{(g)}, I_{\max}^{(g)})$.
3. Repayment fidelity score is drawn from a Beta distribution $H_i \sim \text{Beta}(a_g, b_g)$, modeling observed microfinance group repayment discipline.
4. Loan request amounts are drawn from discrete capital tier choices: $L_i \in \{15k, 20k, 25k, \dots, 50k\}$ INR.

#### 4. 📝 Formal Algorithmic Pseudocode
```text
Algorithm 1: GenerateMicrofinanceCohort
Input:
  - n_applicants: Number of loan applicants N in N_+
  - seed: Pseudorandom generator seed (default: 42)
  - group_split: Demographic proportion vector [p_A, p_B] where p_A + p_B = 1.0

Output:
  - DataFrame D containing N applicant records with fields:
    {applicant_id, demographic_group, monthly_income, dti_ratio, repayment_history,
     poverty_index, dependents, is_women_led, requested_amount}

Procedure:
  1: Initialize RandomNumberGenerator with seed
  2: D ← empty table with N rows
  3: For i ← 1 to N do:
  4:    group_label ← Sample from {"Group_A_Marginalized", "Group_B_General"} with weights [p_A, p_B]
  5:    
  6:    If group_label == "Group_A_Marginalized" then:
  7:        income_i ← Uniform(7000.0, 20000.0)
  8:        dti_i ← Uniform(0.20, 0.65)
  9:        repay_i ← Beta(alpha=6.0, beta=2.0)
 10:        poverty_i ← Uniform(0.55, 0.95)
 11:        dependents_i ← DiscreteChoice([3, 4, 5, 6], weights=[0.25, 0.35, 0.25, 0.15])
 12:        women_led_i ← Bernoulli(p=0.80)
 13:        amount_i ← DiscreteChoice([15000, 20000, 25000, 30000], weights=[0.3, 0.4, 0.2, 0.1])
 14:    Else:
 15:        income_i ← Uniform(18000.0, 45000.0)
 16:        dti_i ← Uniform(0.15, 0.50)
 17:        repay_i ← Beta(alpha=7.0, beta=2.0)
 18:        poverty_i ← Uniform(0.15, 0.50)
 19:        dependents_i ← DiscreteChoice([1, 2, 3, 4], weights=[0.35, 0.35, 0.20, 0.10])
 20:        women_led_i ← Bernoulli(p=0.45)
 21:        amount_i ← DiscreteChoice([25000, 35000, 45000, 50000], weights=[0.2, 0.4, 0.25, 0.15])
 22:    End If
 23:
 24:    D[i] ← { "applicant_id": Format("MFI_APP_{:03d}", i),
 25:             "demographic_group": group_label,
 26:             "monthly_income": Round(income_i, 2),
 27:             "dti_ratio": Round(dti_i, 3),
 28:             "repayment_history": Round(repay_i, 3),
 29:             "poverty_index": Round(poverty_i, 3),
 30:             "dependents": dependents_i,
 31:             "is_women_led": women_led_i,
 32:             "requested_amount": amount_i }
 33: End For
 34: Return D
```

#### 5. 📊 Computational Complexity & Hardware Resource Footprint
- **Time Complexity**: $\mathcal{O}(N)$ sequential stochastic draws.
- **Space / Memory Complexity**: $\mathcal{O}(N \times d)$ tabular memory footprint where $d=9$ features.
- **Downstream Dependency**: Feeds applicant attributes into Algorithm 2 and Algorithm 3.

---

### Algorithm 2: Supervised AI Repayment Modeling & Probability Calibration
* **Target Module**: [`src/repayment_model.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/repayment_model.py)
* **Pipeline Stage**: Step 2 of 13

#### 1. 💡 WHY We Need This Step (Motivation & Problem Context)
1. **Uncalibrated Scores Fail Risk Modeling**: Standard raw classification scores are uncalibrated ordinal values, not true probabilities. To calculate Expected Financial Loss ($R_i = L_i (1-p_i)$) and Monte Carlo scenario default distributions (Algorithm 4), mathematically sound probabilities are mandatory.
2. **Fairness Isolation Principle**: Regulated credit algorithms must not use protected attributes (caste, gender, race) as predictive features to lower creditworthiness. Our ML model is trained with **strict feature isolation** (excluding demographic group labels from $X$).
3. **Platt/Isotonic Reliability**: Calibration minimizes Brier loss $\frac{1}{N}\sum (p_i - y_i)^2$, ensuring a predicted probability of 80% means exactly 80 out of 100 borrowers repay.

#### 2. 🔍 WHAT This Step Does (Specification & Input-Output Mapping)
A supervised machine learning classification and calibration pipeline that estimates posterior repayment probabilities $p_i = \mathbb{P}(\text{Repayment}=1 \mid \vec{x}_i^{\text{feat}})$ using Random Forest and Logistic Regression models fitted with cross-validated Platt scaling (sigmoid calibration).

#### 3. ⚙️ HOW It Works (Mathematical Engine & Mechanics)
1. Extracts financial and behavioral predictors: $X = [\text{Amount}, \text{Income}, \text{DTI}, \text{RepaymentHistory}, \text{PovertyIndex}, \text{Dependents}]$.
2. Fits an ensemble of Decision Trees (Random Forest) or regularized Logistic Regression.
3. Wraps the estimator in `CalibratedClassifierCV` using sigmoid calibration: $P(y=1 \mid \hat{f}) = \frac{1}{1 + \exp(A \hat{f} + B)}$.
4. Evaluates probability calibration via ROC-AUC, Brier score loss, Accuracy, and F1 score. Includes a pure-NumPy analytical fallback for lightweight deployment.

#### 4. 📝 Formal Algorithmic Pseudocode
```text
Algorithm 2: TrainAndCalibrateRepaymentModel
Input:
  - Dataset D: Applicant cohort table from Algorithm 1
  - model_type: Classifier architecture ("logistic" or "calibrated_rf")
  - calibration_method: "sigmoid" (Platt scaling) or "isotonic"

Output:
  - RepaymentProbabilities: Vector p in [0, 1]^N
  - ModelMetrics: Evaluation dictionary { roc_auc, brier_score, accuracy, f1_score }

Procedure:
  1: FeatureCols ← ["requested_amount", "monthly_income", "dti_ratio",
  2:                 "repayment_history", "poverty_index", "dependents"]
  3: X ← D[FeatureCols].to_numpy()
  4: 
  5: If "repaid" in D then:
  6:     y ← D["repaid"].to_numpy()
  7: Else:
  8:     income_ratio ← D["monthly_income"] / Median(D["monthly_income"])
  9:     z ← 2.5 * D["repayment_history"] - 1.8 * D["dti_ratio"] + 0.6 * income_ratio - 1.0 * D["poverty_index"] - 0.2 * (D["dependents"] / 6.0)
 10:     p_latent ← 1.0 / (1.0 + exp(-z))
 11:     y ← (UniformRandom(N) < p_latent).astype(int)
 12:     Ensure both classes {0, 1} are populated
 13: End If
 14:
 15: If scikit-learn is available then:
 16:     If model_type == "logistic" then:
 17:         base_estimator ← LogisticRegression(max_iter=500, random_state=42)
 18:     Else:
 19:         base_estimator ← RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42)
 20:     End If
 21:
 22:     min_class_count ← min(count(y == 0), count(y == 1))
 23:     If min_class_count >= 3 then:
 24:         model ← CalibratedClassifierCV(base_estimator, method=calibration_method, cv=min(3, min_class_count))
 25:     Else:
 26:         model ← base_estimator
 27:     End If
 28:     model.fit(X, y)
 29:     p_repay ← model.predict_proba(X)[:, 1]
 30: Else:
 31:     // Pure-NumPy analytical logistic fallback
 32:     X_norm ← (X - mean(X, axis=0)) / (std(X, axis=0) + 1e-6)
 33:     w_fallback ← [-0.4, 0.6, -1.2, 2.0, -1.0, -0.2]
 34:     p_repay ← 1.0 / (1.0 + exp(-(X_norm @ w_fallback)))
 35: End If
 36:
 37: brier_loss ← mean((p_repay - y)^2)
 38: roc_auc ← ComputeAreaUnderROC(y, p_repay)
 39: Return p_repay, { "roc_auc": roc_auc, "brier_score": brier_loss }
```

#### 5. 📊 Computational Complexity & Hardware Resource Footprint
- **Time Complexity**: $\mathcal{O}(K \cdot N \cdot d)$ where $K$ is the number of trees and $d=6$ is feature dimension.
- **Space / Memory Complexity**: $\mathcal{O}(N \cdot d)$ model training buffers.
- **Downstream Dependency**: Feeds $p_i$ into Algorithm 3 (Expected Return) and Algorithm 4 (Monte Carlo Risk Scenarios).

---

### Algorithm 3: Multi-Objective Tri-Scoring & Welfare Synthesis
* **Target Module**: [`src/scoring.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/scoring.py)
* **Pipeline Stage**: Step 3 of 13

#### 1. 💡 WHY We Need This Step (Motivation & Problem Context)
1. **Mission Drift Dilemma**: Commercial microcredit institutions that optimize solely for financial return ($F_i$) systematically deny loans to poorest borrowers, abandoning their core poverty-alleviation mission.
2. **Institutional Insolvency**: Pure humanitarian charity that ignores financial repayment risk ($F_i$) collapses due to loan defaults and capital depletion.
3. **Holistic Policy Control**: MFI managers can tune policy weights $\vec{\omega} = (\omega_F, \omega_N, \omega_S)$ to trace operational trade-offs across financial return and demographic inclusion.

#### 2. 🔍 WHAT This Step Does (Specification & Input-Output Mapping)
A mathematical normalization and convex synthesis engine that combines three competing microfinance objectives into a single normalized composite utility score $C_i \in [0, 1]$:
$$C_i = \omega_F F_i + \omega_N N_i + \omega_S S_i, \quad \sum \omega = 1.0$$
where $F_i$ is Financial Viability, $N_i$ is Urgent Humanitarian Need, and $S_i$ is Social Impact Multiplier.

#### 3. ⚙️ HOW It Works (Mathematical Engine & Mechanics)
1. **Financial Viability ($F_i$)**: Reward lower DTI, high repayment fidelity history, and stable income relative to median.
2. **Urgent Need ($N_i$)**: Reward multidimensional poverty index, higher dependent count, and lower absolute income.
3. **Social Impact ($S_i$)**: Provide explicit multipliers for women-led enterprises, marginalized communities, and micro-loan leverage.
4. Clamps all components to $[0, 1]$ and evaluates the convex linear combination.

#### 4. 📝 Formal Algorithmic Pseudocode
```text
Algorithm 3: ComputeMultiobjectiveTriScoring
Input:
  - Applicant Table D of size N
  - Calibrated repayment probabilities p in [0, 1]^N
  - Policy weights omega = (omega_F, omega_N, omega_S) with sum(omega) == 1.0

Output:
  - Enriched Table D_scored
  - Composite utility vector C in [0, 1]^N
  - Binary group membership vector G in {0, 1}^N (0: Marginalized, 1: General)

Procedure:
  1: income_median ← Median(D["monthly_income"])
  2: C ← zeros(N); F ← zeros(N); N_need ← zeros(N); S ← zeros(N); G ← zeros(N)
  3:
  4: For i ← 1 to N do:
  5:    // 1. Financial Viability Score F_i
  6:    F[i] ← 0.40 * (1.0 - Clamp(D["dti_ratio"][i], 0.0, 1.0))
  7:           + 0.40 * Clamp(D["repayment_history"][i], 0.0, 1.0)
  8:           + 0.20 * Clamp(D["monthly_income"][i] / (income_median * 1.5), 0.0, 1.0)
  9:    F[i] ← Clamp(F[i], 0.0, 1.0)
 10:
 11:    // 2. Urgent Need Index N_i
 12:    N_need[i] ← 0.50 * Clamp(D["poverty_index"][i], 0.0, 1.0)
 13:                + 0.30 * Clamp(D["dependents"][i] / 6.0, 0.0, 1.0)
 14:                + 0.20 * Clamp(1.0 - (D["monthly_income"][i] / (income_median * 2.0)), 0.0, 1.0)
 15:    N_need[i] ← Clamp(N_need[i], 0.0, 1.0)
 16:
 17:    // 3. Social Impact Multiplier S_i
 18:    women_boost ← 1.0 if D["is_women_led"][i] == True else 0.25
 19:    group_boost ← 1.0 if D["demographic_group"][i] == "Group_A_Marginalized" else 0.35
 20:    size_factor ← Clamp(1.0 - (D["requested_amount"][i] / 60000.0), 0.20, 1.0)
 21:    S[i] ← 0.45 * women_boost + 0.35 * group_boost + 0.20 * size_factor
 22:    S[i] ← Clamp(S[i], 0.0, 1.0)
 23:
 24:    // 4. Composite Utility Synthesis
 25:    C[i] ← omega_F * F[i] + omega_N * N_need[i] + omega_S * S[i]
 26:    G[i] ← 0 if D["demographic_group"][i] == "Group_A_Marginalized" else 1
 27: End For
 28:
 29: D_scored ← Copy(D)
 30: D_scored["financial_score"] ← F
 31: D_scored["need_index"] ← N_need
 32: D_scored["social_impact_score"] ← S
 33: D_scored["composite_score"] ← C
 34: D_scored["group_binary"] ← G
 35: Return D_scored, C, G
```

#### 5. 📊 Computational Complexity & Hardware Resource Footprint
- **Time Complexity**: $\mathcal{O}(N)$ vectorized floating point operations.
- **Space / Memory Complexity**: $\mathcal{O}(N)$ arrays.
- **Downstream Dependency**: Feeds $C_i$ into Algorithm 5 (QUBO linear terms) and Algorithm 7C (MILP objective).

---

### Algorithm 4: Monte Carlo Tail-Risk Engine (VaR & CVaR Simulation)
* **Target Module**: [`src/risk_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/risk_engine.py)
* **Pipeline Stage**: Step 4 of 13

#### 1. 💡 WHY We Need This Step (Motivation & Problem Context)
1. **Average Loss Hides Ruin**: Mean expected loss $\sum L_i (1-p_i) x_i$ tells an MFI its average default rate, but financial ruin is driven by the extreme tail (economic recessions, disease outbreaks, localized monsoon crop failures).
2. **Coherent Risk Metric**: Standard deviation of loss penalizes positive upside as well as downside. CVaR is mathematically proven to be a **coherent risk measure** (subadditive, convex, positive homogeneous) that directly quantifies average loss during the worst 5% of economic crises.
3. **Cross-Method Risk Benchmarking**: Provides an identical, objective risk evaluator across all classical and quantum solvers.

#### 2. 🔍 WHAT This Step Does (Specification & Input-Output Mapping)
A high-throughput Monte Carlo risk simulation engine that samples $S = 2,500$ stochastic default scenarios to evaluate discrete portfolio **Value-at-Risk ($\text{VaR}_{0.95}$)** and **Conditional Value-at-Risk ($\text{CVaR}_{0.95}$ / Expected Shortfall)** for any candidate binary allocation vector $x \in \{0, 1\}^N$.

#### 3. ⚙️ HOW It Works (Mathematical Engine & Mechanics)
1. Pre-generates an $S \times N$ discrete scenario default matrix where element $(s, i) = 1$ if applicant $i$ defaults in scenario $s$ (sampled via Bernoulli($1 - p_i$)).
2. For any allocation $x$, computes scenario loss vector $\vec{\mathcal{L}} = \text{DefaultMatrix} \times (\vec{w} \odot \vec{x})$.
3. Sorts losses ascendingly: $\mathcal{L}_{(1)} \le \mathcal{L}_{(2)} \le \dots \le \mathcal{L}_{(S)}$.
4. Evaluates $\text{VaR}_{0.95} = \mathcal{L}_{(\lceil 0.95 S \rceil)}$ and $\text{CVaR}_{0.95} = \frac{1}{S - \lceil 0.95 S \rceil + 1} \sum_{s \ge \lceil 0.95 S \rceil} \mathcal{L}_{(s)}$.

#### 4. 📝 Formal Algorithmic Pseudocode
```text
Algorithm 4: EvaluatePortfolioTailRisk
Input:
  - Loan amounts vector w = [L_1, ..., L_N] in R^N
  - Default probabilities vector q = 1 - p in [0, 1]^N
  - Allocation decision vector x in {0, 1}^N
  - Number of Monte Carlo scenarios S (default: 2,500)
  - Confidence level alpha in (0, 1) (default: 0.95)

Output:
  - RiskMetrics: { expected_loss, var_alpha, cvar_alpha, max_loss, loss_std }

Procedure:
  1: Initialize RandomNumberGenerator with fixed seed
  2: RandomMatrix ← Matrix of Uniform(0, 1) of shape (S, N)
  3: DefaultMatrix ← (RandomMatrix < q).astype(float)
  4:
  5: CapitalAtRisk ← w * x
  6: ScenarioLosses ← DefaultMatrix @ CapitalAtRisk
  7:
  8: SortedLosses ← SortAscending(ScenarioLosses)
  9:
 10: expected_loss ← Mean(SortedLosses)
 11: var_index ← Floor(alpha * S)
 12: var_alpha ← SortedLosses[var_index]
 13:
 14: TailLosses ← SortedLosses[var_index : S]
 15: cvar_alpha ← Mean(TailLosses)
 16: max_loss ← sum(CapitalAtRisk)
 17: loss_std ← StandardDeviation(SortedLosses)
 18:
 19: Return { "expected_loss": Round(expected_loss, 2),
 20:          "var_95": Round(var_alpha, 2),
 21:          "cvar_95": Round(cvar_alpha, 2),
 22:          "max_loss": Round(max_loss, 2),
 23:          "loss_std": Round(loss_std, 2) }
```

#### 5. 📊 Computational Complexity & Hardware Resource Footprint
- **Time Complexity**: $\mathcal{O}(S \cdot N + S \log S)$ for scenario matrix-vector multiplication and quicksort.
- **Space / Memory Complexity**: $\mathcal{O}(S \cdot N)$ scenario storage buffer ($2,500 \times 8 \times 8$ bytes $\approx 160$ KB).
- **Downstream Dependency**: Evaluates allocations in Algorithm 12 and Algorithm 13.

---

### Algorithm 5: Multi-Objective Constrained Knapsack to QUBO Mapping
* **Target Module**: [`src/qubo_builder.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/qubo_builder.py)
* **Pipeline Stage**: Step 5 of 13

#### 1. 💡 WHY We Need This Step (Motivation & Problem Context)
1. **Quantum Unconstrained Paradigm**: Quantum variational algorithms (QAOA, VQE) and physical quantum annealers cannot natively evaluate inequality boundaries ($\sum w_i x_i \le B$) or equality constraints.
2. **Quadratic Complexity of Demographic Equity**: Imposing demographic parity creates quadratic interaction terms:
   $$\left( \frac{\sum_{i \in G_A} x_i}{|G_A|} - \frac{\sum_{j \in G_B} x_j}{|G_B|} \right)^2 = \sum_{i, j} \sigma_i \sigma_j x_i x_j$$
   This proves that Fair Knapsack is a **Quadratic Knapsack Problem (QKP)**, which is strongly NP-hard. QUBO is the exact mathematical formalism designed for quadratic binary interactions.

#### 2. 🔍 WHAT This Step Does (Specification & Input-Output Mapping)
An exact mathematical compiler that translates the constrained multi-objective knapsack problem into an unconstrained Quadratic Unconstrained Binary Optimization (QUBO) matrix $Q \in \mathbb{R}^{N \times N}$, such that minimizing $E(x) = x^T Q x + E_{\text{offset}}$ solves the original constrained resource allocation problem.

#### 3. ⚙️ HOW It Works (Mathematical Engine & Mechanics)
1. The unconstrained objective penalty functional is defined as:
   $$E(x) = -\sum_{i=1}^N C_i x_i + \lambda_B \left(\sum_{i=1}^N w_i x_i - B\right)^2 + \lambda_F \left(\sum_{i=1}^N \sigma_i x_i\right)^2$$
2. Expands the squared penalties:
   - Budget linear: $\lambda_B \sum_i (w_i^2 - 2 B w_i) x_i$; Budget cross: $2 \lambda_B \sum_{i < j} w_i w_j x_i x_j$.
   - Fairness linear: $\lambda_F \sum_i \sigma_i^2 x_i$; Fairness cross: $2 \lambda_F \sum_{i < j} \sigma_i \sigma_j x_i x_j$.
3. Idempotent Boolean identity ($x_i^2 = x_i$) absorbs linear terms into the diagonal $Q_{ii}$.
4. Off-diagonal $Q_{ij}$ ($i < j$) stores the sum of pairwise budget and fairness cross-penalties.

#### 4. 📝 Formal Algorithmic Pseudocode
```text
Algorithm 5: BuildQUBOMatrix
Input:
  - Composite utility vector C in R^N
  - Requested loan amounts w in R^N
  - Group indicators G in {0, 1}^N
  - Budget capacity B in R_+
  - Penalty weights lambda_B, lambda_F in R_+

Output:
  - Upper-triangular matrix Q in R^{N x N}
  - Constant energy scalar offset E_offset

Procedure:
  1: N ← length(C)
  2: Q ← zeros(N, N)
  3: N_A ← count(i for which G_i == 0)
  4: N_B ← count(i for which G_i == 1)
  5:
  6: sigma ← zeros(N)
  7: For i ← 1 to N do:
  8:    sigma[i] ← (+ 1.0 / N_A) if G[i] == 0 else (- 1.0 / N_B)
  9: End For
 10:
 11: For i ← 1 to N do:
 12:    term_utility ← - C[i]
 13:    term_budget_linear ← lambda_B * (w[i]^2 - 2.0 * B * w[i])
 14:    term_fairness_linear ← lambda_F * (sigma[i]^2)
 15:    Q[i, i] ← term_utility + term_budget_linear + term_fairness_linear
 16: End For
 17:
 18: For i ← 1 to N do:
 19:    For j ← (i + 1) to N do:
 20:        cross_budget ← 2.0 * lambda_B * w[i] * w[j]
 21:        cross_fairness ← 2.0 * lambda_F * sigma[i] * sigma[j]
 22:        Q[i, j] ← cross_budget + cross_fairness
 23:    End For
 24: End For
 25:
 26: E_offset ← lambda_B * (B^2)
 27: Return Q, E_offset
```

#### 5. 📊 Computational Complexity & Hardware Resource Footprint
- **Time Complexity**: $\mathcal{O}(N^2)$ matrix element calculations.
- **Space / Memory Complexity**: $\mathcal{O}(N^2)$ float64 matrix ($8 \times 8$ elements $\approx 512$ bytes).
- **Downstream Dependency**: Feeds $Q$ into Algorithm 6 (Ising mapping), Algorithm 7B (Exact), and Algorithm 7D (Simulated Annealing).

---

### Algorithm 6: QUBO to Ising Spin Glass Transformation
* **Target Module**: [`src/qubo_builder.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/qubo_builder.py)
* **Pipeline Stage**: Step 6 of 13

#### 1. 💡 WHY We Need This Step (Motivation & Problem Context)
1. **Quantum Hardware Substrate**: Superconducting quantum processors (IBM Quantum) execute gates on qubits whose physical eigenstates $|0\rangle$ and $|1\rangle$ have Pauli-Z eigenvalues $+1$ and $-1$.
2. **Unitary Evolution Generator**: In quantum mechanics, state evolution is generated by applying unitary operators $e^{-i \theta H}$. To evolve the quantum wave function toward optimal loan allocations, the cost functional must be an operator Hermitian matrix $H_C$.

#### 2. 🔍 WHAT This Step Does (Specification & Input-Output Mapping)
Transforms the classical binary optimization problem into a quantum spin-half Ising Hamiltonian:
$$H_C = \sum_{i=1}^N h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \text{offset} \cdot I$$
where $Z_i$ is the Pauli-Z operator acting on qubit $i$, $h_i$ is the single-qubit longitudinal magnetic field, and $J_{ij}$ is the two-qubit exchange coupling.

#### 3. ⚙️ HOW It Works (Mathematical Engine & Mechanics)
1. Uses the affine transformation: $x_i = \frac{I - Z_i}{2}$.
2. Substituting into quadratic form $x^T Q x$:
   $$x_i x_j = \frac{I - Z_i - Z_j + Z_i Z_j}{4}$$
3. Longitudinal fields $h_i$ accumulate coefficients of linear $Z_i$ terms:
   $$h_i = -\frac{1}{2} Q_{ii} - \frac{1}{4} \left( \sum_{j > i} Q_{ij} + \sum_{j < i} Q_{ji} \right)$$
4. Interaction couplings $J_{ij}$ collect bilinear $Z_i Z_j$ cross-terms: $J_{ij} = \frac{1}{4} Q_{ij}$.
5. Scalar identity offset shifts the overall energy spectrum.

#### 4. 📝 Formal Algorithmic Pseudocode
```text
Algorithm 6: TransformQUBOToIsing
Input:
  - QUBO Matrix Q in R^{N x N}
  - Constant scalar offset E_offset

Output:
  - Single-qubit longitudinal field vector h in R^N
  - Two-qubit exchange coupling matrix J in R^{N x N} (strictly upper-triangular)
  - Ising Hamiltonian scalar shift ising_offset

Procedure:
  1: N ← dimension(Q, 1)
  2: h ← zeros(N); J ← zeros(N, N)
  3: ising_offset ← E_offset
  4:
  5: For i ← 1 to N do:
  6:    For j ← (i + 1) to N do:
  7:        J[i, j] ← 0.25 * Q[i, j]
  8:    End For
  9: End For
 10:
 11: For i ← 1 to N do:
 12:    sum_couplings ← sum(Q[i, j] for j > i) + sum(Q[j, i] for j < i)
 13:    h[i] ← - 0.5 * Q[i, i] - 0.25 * sum_couplings
 14: End For
 15:
 16: sum_diag ← sum(Q[i, i] for i in 1..N)
 17: sum_upper ← sum(Q[i, j] for i in 1..N, j > i)
 18: ising_offset ← ising_offset + 0.5 * sum_diag + 0.25 * sum_upper
 19:
 20: Return h, J, ising_offset
```

#### 5. 📊 Computational Complexity & Hardware Resource Footprint
- **Time Complexity**: $\mathcal{O}(N^2)$ matrix transformation.
- **Space / Memory Complexity**: $\mathcal{O}(N^2)$ arrays for $h$ and $J$.
- **Downstream Dependency**: Feeds $h_i, J_{ij}$ into Algorithm 8 (QAOA), Algorithm 9 (CVaR-QAOA), Algorithm 10 (VQE), and Algorithm 11 (Qiskit Circuit Export).

---

### Algorithm 7: Classical Benchmark Solvers Suite
* **Target Module**: [`src/classical_solvers.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/classical_solvers.py)
* **Pipeline Stage**: Step 7 of 13

#### 1. 💡 WHY We Need This Step (Motivation & Problem Context)
1. **Scientific Validation**: Quantum claims of performance or approximation ratio ($\alpha = \frac{\text{Utility}_{\text{Quantum}}}{\text{Utility}_{\text{Exact}}}$) are scientifically invalid without comparing against identical classical problem instances.
2. **Exposing Greedy Bias**: Demonstrates that naive classical heuristics (Greedy) maximize immediate value-to-cost ratio but induce massive demographic bias (high DPD, failing the 80% rule).
3. **Establishing Theoretical Optimum**: Exact Combinatorial and MILP solvers establish the provable theoretical ceiling of attainable utility and fairness.

#### 2. 🔍 WHAT This Step Does (Specification & Input-Output Mapping)
A suite of four classical solvers establishing algorithmic baselines across the spectrum of speed, heuristic approximation, and global mathematical exactness:
1. **Greedy Ratio Knapsack**: Standard value/cost heuristic.
2. **Exact Combinatorial Solver**: Exhaustive evaluation over all $2^N$ states (ground-truth reference).
3. **Classical Mixed-Integer Linear Program (MILP)**: `scipy.optimize.milp` with exact simplex branch-and-cut.
4. **Simulated Annealing (SA)**: Classical Markov Chain Monte Carlo (MCMC) thermal spin-flip metaheuristic.

#### 3. ⚙️ HOW It Works (Mathematical Engine & Mechanics)
- **Greedy**: Sorts ratios $r_i = C_i / w_i$ descendingly and packs applicants until budget exhausted.
- **Exact**: Vectorized diagonal Hamiltonian evaluation finding $\arg\min_{k} \text{diag}(H_C)_k$.
- **MILP**: Solves $\min -C^T x$ subject to $0 \le w^T x \le B$, $-\epsilon \le \text{FairnessDiff}(x) \le \epsilon$, $x \in \{0, 1\}^N$.
- **Simulated Annealing**: Proposes single-bit flips, accepts energy improvements unconditionally, and accepts uphill transitions with thermal Metropolis probability $\exp(-\Delta E / T)$, lowering temperature by $T \leftarrow T \cdot \gamma_{\text{cool}}$.

#### 4. 📝 Formal Algorithmic Pseudocode
```text
Algorithm 7: ClassicalSolversSuite

// Subroutine 7A: Greedy Ratio Knapsack Heuristic
Procedure SolveGreedyKnapsack(C, w, B):
  1: ratios ← C / w
  2: sorted_order ← SortIndicesDescending(ratios)
  3: x_greedy ← zeros(N); current_spent ← 0.0
  4: For each index i in sorted_order do:
  5:     If current_spent + w[i] <= B then:
  6:         x_greedy[i] ← 1
  7:         current_spent ← current_spent + w[i]
  8:     End If
  9: End For
 10: Return x_greedy, current_spent

// Subroutine 7B: Exact Combinatorial Ground-State Solver
Procedure SolveExactCombinatorial(diag_energies):
  1: best_idx ← ArgMin(diag_energies)
  2: x_exact ← BinaryVectorFromIndex(best_idx, length=N)
  3: Return x_exact, diag_energies[best_idx]

// Subroutine 7C: Exact Classical Mixed-Integer Linear Programming (MILP)
Procedure SolveClassicalMILP(C, w, B, G, tolerance_epsilon):
  1: Objective: Minimize - C^T x
  2: Constraints:
  3:    0 <= w^T x <= B
  4:    -epsilon <= (1/N_A sum_{i in A} x_i) - (1/N_B sum_{j in B} x_j) <= epsilon
  5:    x_i in {0, 1} for all i in 1..N
  6: res ← ScipyMILP(c=-C, constraints, integrality=ones(N))
  7: Return res.x.astype(int), res.fun

// Subroutine 7D: Classical Simulated Annealing (Metropolis-Hastings MCMC)
Procedure SolveSimulatedAnnealing(Q, E_offset, T_start=10.0, T_min=0.01, cooling=0.985, max_steps=1200):
  1: x_curr ← RandomBinaryVector(N); E_curr ← Energy(x_curr, Q, E_offset)
  2: x_best ← x_curr; E_best ← E_curr; T ← T_start
  3: While T > T_min and step < max_steps do:
  4:     k ← RandomInteger(1, N)
  5:     x_cand ← x_curr with bit k flipped
  6:     E_cand ← Energy(x_cand, Q, E_offset)
  7:     delta_E ← E_cand - E_curr
  8:     If delta_E < 0 or RandomUniform(0, 1) < exp(-delta_E / T) then:
  9:         x_curr ← x_cand; E_curr ← E_cand
 10:         If E_curr < E_best then:
 11:             x_best ← x_curr; E_best ← E_curr
 12:         End If
 13:     End If
 14:     T ← T * cooling
 15: End While
 16: Return x_best, E_best
```

#### 5. 📊 Computational Complexity & Hardware Resource Footprint
- **Greedy**: Time $\mathcal{O}(N \log N)$, Space $\mathcal{O}(N)$.
- **Exact Combinatorial**: Time $\mathcal{O}(2^N)$, Space $\mathcal{O}(2^N)$.
- **MILP**: Time $\mathcal{O}(2^N)$ worst-case, practical $\mathcal{O}(N)$ via branch-and-cut, Space $\mathcal{O}(N)$.
- **Simulated Annealing**: Time $\mathcal{O}(\text{steps} \cdot N)$, Space $\mathcal{O}(N)$.

---

### Algorithm 8: Standard Quantum Approximate Optimization Algorithm (QAOA)
* **Target Module**: [`src/quantum_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/quantum_engine.py#L110)
* **Pipeline Stage**: Step 8 of 13

#### 1. 💡 WHY We Need This Step (Motivation & Problem Context)
1. **Problem-Tailored Architecture**: Unlike generic neural networks or black-box heuristics, QAOA directly encodes the problem's graph structure into the unitary circuit $U_C(\gamma)$, requiring only $2p$ variational angles.
2. **Quantum Tunneling**: Classical heuristics get trapped in high-energy barriers created by strict knapsack penalties; quantum superposition and transverse mixing ($U_M$) allow wave packets to tunnel through barriers.
3. **Provable Convergence**: Farhi et al. (2014) proved that as circuit depth $p \to \infty$, QAOA convergence to global optimality is mathematically guaranteed.

#### 2. 🔍 WHAT This Step Does (Specification & Input-Output Mapping)
A hybrid quantum-classical variational algorithm that alternates application of the Problem Cost Unitary $U_C(\gamma) = e^{-i \gamma H_C}$ and Transverse Mixer Unitary $U_M(\beta) = e^{-i \beta \sum X_i}$ over $p$ layers, tuning $2p$ parameters to minimize the global energy expectation $\langle \psi(\vec{\gamma}, \vec{\beta}) | H_C | \psi(\vec{\gamma}, \vec{\beta}) \rangle$.

#### 3. ⚙️ HOW It Works (Mathematical Engine & Mechanics)
1. Initializes uniform superposition state $|+\rangle^{\otimes N} = \frac{1}{\sqrt{2^N}} \sum_{x} |x\rangle$ using Hadamard gates.
2. Alternates $p$ layers:
   - Apply diagonal phase rotation: $|\psi\rangle \leftarrow e^{-i \gamma_k H_C} |\psi\rangle$ (implemented via $R_Z$ and $R_{ZZ}$ gates).
   - Apply transverse mixer: $|\psi\rangle \leftarrow e^{-i \beta_k \sum X_i} |\psi\rangle$ (implemented via $R_X(2 \beta_k)$ gates).
3. Evaluates expectation: $E(\vec{\gamma}, \vec{\beta}) = \sum_{k=0}^{2^N-1} |\psi_k|^2 E_k$.
4. A classical optimizer (COBYLA) updates angles until convergence.
5. Samples the final statevector and selects the highest-probability allocation bitstring.

#### 4. 📝 Formal Algorithmic Pseudocode
```text
Algorithm 8: StandardQAOA
Input:
  - Hamiltonian parameters (h, J, ising_offset)
  - Number of circuit layers p in {1, 2}
  - Max optimizer iterations max_iter (default: 120)

Output:
  - x_optimal: Sampled allocation bitstring in {0, 1}^N
  - E_optimal: Optimized expectation energy
  - StateProbabilities: Measurement distribution over 2^N computational basis states

Procedure:
  1: |psi_0> ← (1 / sqrt(2^N)) * sum_{k=0}^{2^N - 1} |k>
  2: gamma_init ← UniformRandom(0.1, pi, size=p)
  3: beta_init ← UniformRandom(0.1, pi / 2.0, size=p)
  4: params_init ← Concatenate([gamma_init, beta_init])
  5:
  6: Define ObjectiveFunction(params):
  7:     gamma_vec ← params[0 : p]
  8:     beta_vec ← params[p : 2p]
  9:     |psi> ← |psi_0>
 10:     For k ← 0 to (p - 1) do:
 11:         |psi> ← ApplyDiagonalCostUnitary(|psi>, gamma_vec[k])
 12:         |psi> ← ApplyTransverseMixerUnitary(|psi>, beta_vec[k])
 13:     End For
 14:     probs ← |psi|^2
 15:     expectation_energy ← sum(probs[k] * diag_energies[k] for k in 0..2^N - 1)
 16:     Return expectation_energy
 17:
 18: opt_result ← Minimize(ObjectiveFunction, params_init, method="COBYLA", maxiter=max_iter)
 19: |psi_opt> ← EvolveState(|psi_0>, opt_result.x[:p], opt_result.x[p:])
 20: probs_opt ← |psi_opt|^2
 21: best_state_idx ← ArgMax(probs_opt)
 22: x_optimal ← BitstringFromInteger(best_state_idx, length=N)
 23: Return x_optimal, opt_result.fun, probs_opt
```

#### 5. 📊 Computational Complexity & Hardware Resource Footprint
- **Time Complexity**: $\mathcal{O}(\text{evals} \cdot p \cdot 2^N)$ statevector simulation steps.
- **Space / Memory Complexity**: $\mathcal{O}(2^N)$ complex128 statevector amplitudes ($2^8 \times 16$ bytes $= 4$ KB).
- **Quantum Hardware Footprint**: $N$ qubits, $2p$ classical variational angles, $\mathcal{O}(p \cdot N^2)$ entangling gates.

---

### Algorithm 9: Tail-Risk Conditional Value-at-Risk QAOA (CVaR-QAOA)
* **Target Module**: [`src/quantum_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/quantum_engine.py#L110-L140)
* **Pipeline Stage**: Step 9 of 13

#### 1. 💡 WHY We Need This Step (Motivation & Problem Context)
1. **Expectation Values Dilute Ground States**: In standard QAOA, high-energy infeasible bitstrings in the state distribution pull the mean upward, forcing the classical optimizer to compromise across the entire Hilbert space.
2. **Tail-Risk Filtering**: In loan portfolio allocation, we do not care about the average quality of infeasible states; we care strictly about the best feasible ground states. CVaR acts as a risk filter that instructs the classical optimizer to ignore poor states and concentrate probability amplitude exclusively on top-tier candidate bitstrings.
3. **Faster NISQ Convergence**: Barkoutsos et al. (2020) demonstrated that for shallow circuits ($p=1, 2$), CVaR-QAOA converges significantly faster than standard QAOA on combinatorial landscapes.

#### 2. 🔍 WHAT This Step Does (Specification & Input-Output Mapping)
A risk-aware quantum variational optimization algorithm based on [Barkoutsos et al. (2020)](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/algorithmic_study.md#L275) that modifies the classical feedback loop of QAOA: instead of minimizing the global mean expectation $\langle H_C \rangle$, it minimizes the Conditional Value-at-Risk ($\text{CVaR}_\alpha$) over the lowest $\alpha$-quantile ($\alpha = 0.25$) of the energy distribution.

#### 3. ⚙️ HOW It Works (Mathematical Engine & Mechanics)
1. Evolves statevector $|\psi(\vec{\gamma}, \vec{\beta})\rangle$ through $p$ layers.
2. Evaluates measurement probability vector $P_k = |\psi_k|^2$.
3. Sorts Hamiltonian eigenenergies in ascending order: $E_{(0)} \le E_{(1)} \le \dots \le E_{(2^N-1)}$.
4. Finds cutoff index $K_\alpha$ where cumulative probability reaches quantile $\alpha$:
   $$\sum_{j=0}^{K_\alpha - 1} P_{(j)} < \alpha \le \sum_{j=0}^{K_\alpha} P_{(j)}$$
5. Computes analytical CVaR:
   $$\text{CVaR}_\alpha = \frac{1}{\alpha} \left[ \sum_{j=0}^{K_\alpha - 1} P_{(j)} E_{(j)} + \left(\alpha - \sum_{j=0}^{K_\alpha - 1} P_{(j)}\right) E_{(K_\alpha)} \right]$$
6. Optimizer updates $(\vec{\gamma}, \vec{\beta})$ using $\text{CVaR}_\alpha$ as the loss function.

#### 4. 📝 Formal Algorithmic Pseudocode
```text
Algorithm 9: CVaR_QAOA
Input:
  - Hamiltonian parameters (h, J, ising_offset)
  - Circuit depth p in N_+
  - CVaR confidence quantile alpha in (0, 1] (default: 0.25)
  - Optimizer settings (COBYLA, max_iter=120)

Output:
  - x_cvar: Optimal sampled allocation vector in {0, 1}^N
  - CVaR_Energy: Optimized tail expectation value
  - probs: Output state measurement distribution

Procedure:
  1: |psi_0> ← UniformSuperposition(N)
  2: params_init ← InitializeVariationalAngles(p)
  3: 
  4: Define CVaR_Objective(params):
  5:     |psi> ← EvolveCircuit(|psi_0>, params, p)
  6:     probs ← |psi|^2
  7:     
  8:     If alpha >= 1.0 then:
  9:         Return ExpectationEnergy(probs, diag_energies)
 10:     End If
 11:     
 12:     sort_order ← ArgsortAscending(diag_energies)
 13:     sorted_E ← diag_energies[sort_order]
 14:     sorted_P ← probs[sort_order]
 15:     
 16:     cum_P ← CumulativeSum(sorted_P)
 17:     k_alpha ← BinarySearch(cum_P, alpha)
 18:     
 19:     P_head ← sorted_P[0 : k_alpha]
 20:     E_head ← sorted_E[0 : k_alpha]
 21:     residual_P ← alpha - sum(P_head)
 22:     
 23:     cvar_val ← (sum(P_head * E_head) + max(0.0, residual_P) * sorted_E[k_alpha]) / alpha
 24:     Return cvar_val
 25:
 26: opt_res ← ClassicalOptimizer(CVaR_Objective, params_init, method="COBYLA")
 27: |psi_final> ← EvolveCircuit(|psi_0>, opt_res.x, p)
 28: x_cvar ← ArgMaxBasisState(|psi_final|^2)
 29: Return x_cvar, opt_res.fun, |psi_final|^2
```

#### 5. 📊 Computational Complexity & Hardware Resource Footprint
- **Time Complexity**: $\mathcal{O}(\text{evals} \cdot (p \cdot 2^N + 2^N \log(2^N)))$ including quantile sorting.
- **Space / Memory Complexity**: $\mathcal{O}(2^N)$ statevector memory.
- **Quantum Advantage Potential**: When $\alpha \to 1$, matches Standard QAOA; when $\alpha = 0.25$, suppresses non-optimal tail states.

---

### Algorithm 10: Variational Quantum Eigensolver (VQE) with Hardware-Efficient Ansatz
* **Target Module**: [`src/quantum_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/quantum_engine.py#L240)
* **Pipeline Stage**: Step 10 of 13

#### 1. 💡 WHY We Need This Step (Motivation & Problem Context)
1. **Architectural Comparison**: QAOA circuits are structured according to the problem graph; VQE circuits are structured according to the physical quantum processor's native connectivity. Evaluating both side-by-side demonstrates whether problem-tailored circuits outperform device-native circuits.
2. **Expressive Power**: VQE uses $N \times (L + 1)$ parameters (where $L$ is layer repetitions), exploring a broader manifold of states than QAOA's $2p$ parameters.

#### 2. 🔍 WHAT This Step Does (Specification & Input-Output Mapping)
An unconstrained variational eigensolver employing a Hardware-Efficient TwoLocal ansatz consisting of single-qubit parameterized $R_y(\theta)$ rotations and circular Controlled-NOT (CNOT) entangling gates to explore the Hilbert space and minimize ground state energy.

#### 3. ⚙️ HOW It Works (Mathematical Engine & Mechanics)
1. Begins in ground state $|0\rangle^{\otimes N}$.
2. In each repetition layer $l \in \{1, \dots, L\}$:
   - Rotates all qubits around Y-axis: $\bigotimes_{i=1}^N R_y(\theta_{i, l})$.
   - Applies circular entangling gates: $\text{CNOT}(i, (i + 1) \bmod N)$.
3. Applies a final rotation layer $\bigotimes_{i=1}^N R_y(\theta_{i, L+1})$.
4. Evaluates expectation $\langle 0 | U^\dagger(\vec{\theta}) H_C U(\vec{\theta}) | 0 \rangle$.
5. Classical optimizer (COBYLA) tunes $\vec{\theta}$ to minimize energy.

#### 4. 📝 Formal Algorithmic Pseudocode
```text
Algorithm 10: HardwareEfficientVQE
Input:
  - Diagonal problem Hamiltonian H_C
  - Number of ansatz repetitions L (default: 1)
  - Number of qubits N

Output:
  - x_vqe: Ground state binary allocation bitstring
  - E_vqe: Minimal variational ground state energy

Procedure:
  1: num_parameters ← N * (L + 1)
  2: theta_init ← UniformRandom(0.05, 0.5, size=num_parameters)
  3: 
  4: Define VQE_Ansatz(|0>^N, theta):
  5:     |psi> ← |0>^N
  6:     For l ← 0 to (L - 1) do:
  7:         Apply single-qubit rotations Ry(theta[l*N + i]) to all qubits i in 0..N-1
  8:         Apply entangling CNOT gates in circular topology: CNOT(i, (i + 1) % N)
  9:     End For
 10:     Apply final rotation layer Ry(theta[L*N + i]) to all qubits i in 0..N-1
 11:     Return |psi>
 12:
 13: Define VQE_Energy(theta):
 14:     |psi> ← VQE_Ansatz(|0>^N, theta)
 15:     probs ← |psi|^2
 16:     Return sum(probs[k] * diag_energies[k] for k in 0..2^N - 1)
 17:
 18: opt_res ← Minimize(VQE_Energy, theta_init, method="COBYLA", maxiter=120)
 19: |psi_opt> ← VQE_Ansatz(|0>^N, opt_res.x)
 20: x_vqe ← ArgMaxBasisState(|psi_opt|^2)
 21: Return x_vqe, opt_res.fun
```

#### 5. 📊 Computational Complexity & Hardware Resource Footprint
- **Time Complexity**: $\mathcal{O}(\text{evals} \cdot L \cdot 2^N)$.
- **Space / Memory Complexity**: $\mathcal{O}(2^N)$ complex statevector amplitudes.
- **Ansatz Parameters**: $N \times (L + 1) = 8 \times 2 = 16$ variational parameters.

---

### Algorithm 11: Native Qiskit Circuit Synthesis & Gate Decomposition
* **Target Module**: [`src/quantum_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/quantum_engine.py#L300), [`visuals/qiskit_qaoa_circuit.txt`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/visuals/qiskit_qaoa_circuit.txt)
* **Pipeline Stage**: Step 11 of 13

#### 1. 💡 WHY We Need This Step (Motivation & Problem Context)
1. **Bridging Theory and Hardware**: Theoretical spin Hamiltonians cannot be executed directly on physical QPUs; they must be decomposed into sequences of calibrated hardware basis gates.
2. **Qiskit Runtime Compatibility**: Prepares the project for Phase 3 execution on real 127-qubit IBM Quantum superconducting processors (e.g. `ibm_brisbane`, `ibm_kyoto`).
3. **Visual Circuit Verification**: Provides clear ASCII gate diagrams for project presentations and thesis defense.

#### 2. 🔍 WHAT This Step Does (Specification & Input-Output Mapping)
A quantum circuit compiler that transforms mathematical Hamiltonian parameters ($h_i, J_{ij}$) into native Qiskit `QuantumCircuit` objects, decomposing operators into standard physical quantum gates ($H, R_Z, R_{ZZ}, R_X$, Measurement) and exporting OpenQASM/ASCII circuit representations.

#### 3. ⚙️ HOW It Works (Mathematical Engine & Mechanics)
1. Instantiates a `QuantumRegister` and `ClassicalRegister` of size $N$.
2. Applies a layer of Hadamard gates ($H$) across all qubits to initialize $|+\rangle^{\otimes N}$.
3. For each QAOA layer $k \in \{1, \dots, p\}$:
   - Synthesizes problem unitary: applies single-qubit $R_Z(2 \gamma_k h_i)$ for non-zero fields, and two-qubit entangling $R_{ZZ}(2 \gamma_k J_{ij})$ for non-zero couplings.
   - Synthesizes mixer unitary: applies $R_X(2 \beta_k)$ to each qubit.
4. Appends barrier and measurement operations, compiling text/OpenQASM output.

#### 4. 📝 Formal Algorithmic Pseudocode
```text
Algorithm 11: SynthesizeQiskitCircuit
Input:
  - Longitudinal fields h in R^N
  - Coupling matrix J in R^{N x N}
  - Circuit depth p in N_+

Output:
  - qc: Parameterized Qiskit QuantumCircuit
  - ascii_diagram: Human-readable gate level circuit string

Procedure:
  1: qreg ← QuantumRegister(N, "q"); creg ← ClassicalRegister(N, "meas")
  2: qc ← QuantumCircuit(qreg, creg)
  3: 
  4: For each qubit q in 0 to (N - 1) do:
  5:     qc.h(q)
  6: End For
  7: qc.barrier()
  8:
  9: For layer k ← 1 to p do:
 10:     gamma_k ← Parameter(Format("gamma_{}", k))
 11:     beta_k ← Parameter(Format("beta_{}", k))
 12:     
 13:     For i ← 0 to (N - 1) do:
 14:         If abs(h[i]) > 1e-5 then:
 15:             qc.rz(2.0 * h[i] * gamma_k, i)
 16:         End If
 17:     End For
 18:     For i ← 0 to (N - 1) do:
 19:         For j ← (i + 1) to (N - 1) do:
 20:             If abs(J[i, j]) > 1e-5 then:
 21:                 qc.rzz(2.0 * J[i, j] * gamma_k, i, j)
 22:             End If
 23:         End For
 24:     End For
 25:     qc.barrier()
 26:     
 27:     For i ← 0 to (N - 1) do:
 28:         qc.rx(2.0 * beta_k, i)
 29:     End For
 30:     qc.barrier()
 31: End For
 32:
 33: qc.measure(qreg, creg)
 34: ascii_diagram ← qc.draw(output="text")
 35: Return qc, ascii_diagram
```

#### 5. 📊 Computational Complexity & Hardware Resource Footprint
- **Time Complexity**: $\mathcal{O}(p \cdot N^2)$ gate construction time.
- **Circuit Depth**: $\mathcal{O}(p \cdot N)$ depth depending on qubit connectivity topology.
- **Two-Qubit Gate Count**: Up to $\frac{p \cdot N(N-1)}{2}$ $R_{ZZ}$ two-qubit entangling gates.

---

### Algorithm 12: Multi-Objective Fairness & Tail-Risk Evaluation
* **Target Module**: [`src/metrics.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/metrics.py)
* **Pipeline Stage**: Step 12 of 13

#### 1. 💡 WHY We Need This Step (Motivation & Problem Context)
1. **Regulatory Compliance (Four-Fifths Rule)**: The US EEOC and international fair lending guidelines mandate that selection rates of protected classes must achieve at least $80\%$ of the reference group ($\text{DIR} = \frac{SR_A}{SR_B} \ge 0.80$).
2. **Multi-Faceted Audit**: A solution that achieves 100% budget utilization but has $\text{DPD} = 0.75$ is socially unacceptable; a solution that has $\text{DPD} = 0.00$ but allocates 0 capital is financially useless. We must measure all dimensions simultaneously.

#### 2. 🔍 WHAT This Step Does (Specification & Input-Output Mapping)
A multi-metric fairness auditing engine that evaluates any candidate allocation $x \in \{0, 1\}^N$ across financial metrics (total cost, budget utilization, expected repayment), demographic fairness metrics (selection rates $SR_g$, Demographic Parity Difference $\text{DPD}$, Disparate Impact Ratio $\text{DIR}$), tail-risk metrics ($\text{CVaR}_{0.95}$), and quantum approximation ratio ($\alpha$).

#### 3. ⚙️ HOW It Works (Mathematical Engine & Mechanics)
1. Evaluates total approved capital $\sum w_i x_i$ and verifies feasibility against budget $B$.
2. Computes group approval rates: $\text{Rate}_A = \frac{\sum_{i \in G_A} x_i}{|G_A|}$, $\text{Rate}_B = \frac{\sum_{j \in G_B} x_j}{|G_B|}$.
3. Computes Demographic Parity Difference: $\text{DPD} = |\text{Rate}_A - \text{Rate}_B|$.
4. Computes Disparate Impact Ratio: $\text{DIR} = \frac{\text{Rate}_A}{\text{Rate}_B}$ with explicit zero-division protection.
5. Queries Algorithm 4 for portfolio Expected Loss and $\text{CVaR}_{0.95}$.
6. Computes Approximation Ratio: $\alpha = \frac{\sum C_i x_i}{\text{OptimalUtility}_{\text{Exact}}}$.

#### 4. 📝 Formal Algorithmic Pseudocode
```text
Algorithm 12: EvaluateAllocationMetrics
Input:
  - Allocation vector x in {0, 1}^N
  - Applicant Table D_scored
  - Budget ceiling B
  - RiskEngine instance from Algorithm 4
  - Optimal classical utility C_optimal

Output:
  - MetricsRecord M

Procedure:
  1: TotalCost ← sum(w[i] * x[i] for i in 1..N)
  2: TotalUtility ← sum(C[i] * x[i] for i in 1..N)
  3: TotalApproved ← sum(x[i] for i in 1..N)
  4: is_feasible ← (TotalCost <= B + 1e-5)
  5: budget_utilization ← (TotalCost / B) * 100.0
  6:
  7: mask_A ← (G == 0); mask_B ← (G == 1)
  8: n_A ← max(1, count(mask_A)); n_B ← max(1, count(mask_B))
  9: rate_A ← sum(x[mask_A]) / n_A
 10: rate_B ← sum(x[mask_B]) / n_B
 11: 
 12: DPD ← abs(rate_A - rate_B)
 13:
 14: If rate_B > 0 then:
 15:     DIR ← rate_A / rate_B
 16: Else:
 17:     DIR ← 1.0 if rate_A == 0 else +infinity
 18: End If
 19:
 20: risk_stats ← RiskEngine.Evaluate(x, alpha=0.95)
 21: approx_ratio ← TotalUtility / C_optimal if C_optimal > 0 else 1.0
 22:
 23: Return { "solver": solver_name,
 24:          "total_approved": TotalApproved,
 25:          "total_cost": TotalCost,
 26:          "budget_utilization_pct": budget_utilization,
 27:          "is_feasible": is_feasible,
 28:          "total_utility": TotalUtility,
 29:          "expected_loss": risk_stats["expected_loss"],
 30:          "cvar_95_loss": risk_stats["cvar_95"],
 31:          "rate_group_A": rate_A,
 32:          "rate_group_B": rate_B,
 33:          "demographic_parity_diff": DPD,
 34:          "disparate_impact_ratio": DIR,
 35:          "approximation_ratio": approx_ratio }
```

#### 5. 📊 Computational Complexity & Hardware Resource Footprint
- **Time Complexity**: $\mathcal{O}(N + S)$ linear evaluation.
- **Space / Memory Complexity**: $\mathcal{O}(1)$ metrics dictionary.
- **Downstream Dependency**: Compiles rows for the Master Benchmark Table in Algorithm 13.

---

### Algorithm 13: End-to-End Benchmark Pipeline Orchestrator
* **Target Module**: [`midsem_pipeline.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/midsem_pipeline.py)
* **Pipeline Stage**: Step 13 of 13

#### 1. 💡 WHY We Need This Step (Motivation & Problem Context)
1. **Reproducibility Guarantee**: In computational research, results must be reproducible from raw data to final plots with a single execution command (`python midsem_pipeline.py`).
2. **Automated Verification**: Reaches the verified **70% Accomplishment Milestone Checkpoint** without manual intervention.

#### 2. 🔍 WHAT This Step Does (Specification & Input-Output Mapping)
The master execution orchestrator that unifies all 12 prior modules, executing the end-to-end experimental benchmark comparing 8 classical and quantum solvers, printing the publication comparison table, and generating 5 high-resolution visuals.

#### 3. ⚙️ HOW It Works (Mathematical Engine & Mechanics)
1. Sequences cohort generation (Alg 1), AI calibration (Alg 2), tri-scoring (Alg 3), and risk engine setup (Alg 4).
2. Synthesizes QUBO (Alg 5) and Ising Hamiltonian (Alg 6).
3. Executes 4 classical solvers (Alg 7A-D) and 4 quantum variational engines (Alg 8-10).
4. Evaluates all metrics (Alg 12) and compiles the Master Comparison Table.
5. Synthesizes 5 visual figures in `visuals/` and exports native Qiskit circuit (Alg 11).

#### 4. 📝 Formal Algorithmic Pseudocode
```text
Algorithm 13: RunMidsemBenchmarkPipeline
Procedure:
  1: cohort ← GenerateMicrofinanceCohort(N=8, seed=42)
  2: p_repay, ml_metrics ← TrainAndCalibrateRepaymentModel(cohort)
  3: scored_cohort, C, G ← ComputeMultiobjectiveTriScoring(cohort, p_repay)
  4: risk_engine ← InitializeRiskEngine(cohort.amounts, 1.0 - p_repay)
  5: Q, E_offset ← BuildQUBOMatrix(C, cohort.amounts, G, Budget=110k)
  6: h, J, ising_shift ← TransformQUBOToIsing(Q, E_offset)
  7: 
  8: // Execute 8-Solver Comparative Benchmark
  9: res_greedy   ← SolveGreedyKnapsack(C, cohort.amounts, Budget=110k)
 10: res_exact    ← SolveExactCombinatorial(GetDiagonal(Q))
 11: res_milp     ← SolveClassicalMILP(C, cohort.amounts, Budget=110k, G)
 12: res_sa       ← SolveSimulatedAnnealing(Q, E_offset)
 13: res_qaoa_p1  ← StandardQAOA(h, J, ising_shift, p=1)
 14: res_qaoa_p2  ← StandardQAOA(h, J, ising_shift, p=2)
 15: res_cvar     ← CVaR_QAOA(h, J, ising_shift, p=1, alpha=0.25)
 16: res_vqe      ← HardwareEfficientVQE(h, J, L=1)
 17:
 18: For each solver in {Greedy, Exact, MILP, SA, QAOA_p1, QAOA_p2, CVaR_QAOA, VQE} do:
 19:     metrics[solver] ← EvaluateAllocationMetrics(solver.allocation, scored_cohort, risk_engine)
 20: End For
 21:
 22: PlotQUBOHeatmap(Q, "visuals/qubo_matrix_heatmap.png")
 23: PlotMultiMetricComparison(metrics, "visuals/quantum_vs_classical_comparison.png")
 24: PlotConvergenceProfile(histories, "visuals/qaoa_convergence_profile.png")
 25: PlotParetoFrontier(varying lambda_F, "visuals/fairness_vs_budget_tradeoff.png")
 26: PlotQuantumProbability(res_qaoa_p2.probs, "visuals/bitstring_probability_distribution.png")
 27: ExportQiskitCircuit(h, J, p=1, "visuals/qiskit_qaoa_circuit.txt")
 28:
 29: PrintBenchmarkTable(metrics)
 30: Print("[SUCCESS] 70% Accomplishment Milestone Checkpoint Complete")
```

#### 5. 📊 Computational Complexity & Hardware Resource Footprint
- **Total Runtime**: Under 7.0 seconds on standard CPU simulation ($N=8$, $S=2,500$).
- **Output Artifacts**: 5 presentation figures (`visuals/*.png`), 1 circuit diagram (`qiskit_qaoa_circuit.txt`), 1 benchmark cohort (`data/sample_applicants.csv`).

---

# 3. Master Algorithm Complexity & Hardware Resource Matrix

| Alg # | Algorithm Name | Method / Paradigm | Time Complexity | Memory / Space | Target Source File |
|:---:|---|---|:---:|:---:|---|
| **1** | `GenerateMicrofinanceCohort` | Statistical Distribution Sampling | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ | [`src/data_generator.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/data_generator.py) |
| **2** | `TrainAndCalibrateRepaymentModel` | Supervised Random Forest + Platt Scaling | $\mathcal{O}(K \cdot N \cdot d)$ | $\mathcal{O}(N \cdot d)$ | [`src/repayment_model.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/repayment_model.py) |
| **3** | `ComputeMultiobjectiveTriScoring` | Convex Objective Synthesis | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ | [`src/scoring.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/scoring.py) |
| **4** | `EvaluatePortfolioTailRisk` | Monte Carlo Stochastic Simulation ($S=2,500$) | $\mathcal{O}(S \cdot N + S \log S)$ | $\mathcal{O}(S \cdot N)$ | [`src/risk_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/risk_engine.py) |
| **5** | `BuildQUBOMatrix` | Quadratic Penalty Formulation | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ | [`src/qubo_builder.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/qubo_builder.py) |
| **6** | `TransformQUBOToIsing` | Affine Pauli-Z Operator Mapping | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ | [`src/qubo_builder.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/qubo_builder.py) |
| **7A**| `SolveGreedyKnapsack` | Value-to-Cost Ratio Heuristic | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N)$ | [`src/classical_solvers.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/classical_solvers.py) |
| **7B**| `SolveExactCombinatorial` | Exhaustive State Evaluation ($2^N$) | $\mathcal{O}(2^N)$ | $\mathcal{O}(2^N)$ | [`src/classical_solvers.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/classical_solvers.py) |
| **7C**| `SolveClassicalMILP` | Exact Branch-and-Cut Simplex | $\mathcal{O}(2^N)$ worst-case | $\mathcal{O}(N)$ | [`src/classical_solvers.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/classical_solvers.py) |
| **7D**| `SolveSimulatedAnnealing` | Metropolis-Hastings MCMC | $\mathcal{O}(\text{steps} \cdot N)$ | $\mathcal{O}(N)$ | [`src/classical_solvers.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/classical_solvers.py) |
| **8** | `StandardQAOA` | Expectation Variational Optimization ($p=1, 2$) | $\mathcal{O}(\text{evals} \cdot p \cdot 2^N)$ | $N$ Qubits / $2^N$ Floats | [`src/quantum_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/quantum_engine.py) |
| **9** | `CVaR_QAOA` | Tail-Risk Quantile Optimization ($\alpha=0.25$) | $\mathcal{O}(\text{evals} \cdot 2^N \log(2^N))$ | $N$ Qubits / $2^N$ Floats | [`src/quantum_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/quantum_engine.py) |
| **10**| `HardwareEfficientVQE` | Parameterized TwoLocal Ansatz ($R_y$ + CNOT) | $\mathcal{O}(\text{evals} \cdot L \cdot 2^N)$ | $N$ Qubits / $2^N$ Floats | [`src/quantum_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/quantum_engine.py) |
| **11**| `SynthesizeQiskitCircuit` | Native Circuit Synthesis & Decomposition | $\mathcal{O}(p \cdot N^2)$ | $\mathcal{O}(p \cdot N^2)$ Gates | [`visuals/qiskit_qaoa_circuit.txt`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/visuals/qiskit_qaoa_circuit.txt) |
| **12**| `EvaluateAllocationMetrics` | Multi-Metric Fairness & Tail-Risk Auditing | $\mathcal{O}(N + S)$ | $\mathcal{O}(1)$ | [`src/metrics.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/metrics.py) |
| **13**| `RunMidsemBenchmarkPipeline`| Master End-to-End Orchestrator | $\mathcal{O}(2^N + S \cdot N)$ | $\mathcal{O}(2^N)$ | [`midsem_pipeline.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/midsem_pipeline.py) |
