# Comprehensive Algorithmic Specification & Pipeline Architecture (`pseudo.md`)
## Project: Quantum-Assisted Fair Resource Allocation for Financial Inclusion in Microfinance (Q-FAR)
**Course**: Quantum Computing and Advanced Algorithms (QCAA)  
**Document Type**: End-to-End Computational Pipeline Specification (HPC / AI / Quantum Systems Standard)  
**Current Milestone**: 70% Accomplishment Milestone  

---

## 1. System Pipeline Overview & Computational Architecture

The Q-FAR pipeline unifies supervised machine learning, credit risk analytics, classical combinatorial optimization, and parameterized quantum variational circuits into an integrated algorithmic framework. 

The complete execution flow from raw borrower features to Pareto-optimal, fair capital allocations is structured across **13 core algorithmic procedures**:

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

---

## 2. Granular Algorithmic Specifications

---

### Algorithm 1: Demographic & Microfinance Cohort Generation
* **Module**: [`src/data_generator.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/data_generator.py)
* **Purpose**: Generates realistic microfinance borrower profiles reflecting empirical microcredit distributions in developing economies (bimodal income, debt-to-income ratios, multidimensional poverty indicators, and protected demographic group labels).
* **Mathematical Model**:
  - Group assignment: $g_i \sim \text{Categorical}(\{\text{Marginalized}, \text{General}\}, [p_A, 1 - p_A])$
  - Household Income: $I_i \sim \mathcal{U}(I_{\min}^{(g)}, I_{\max}^{(g)})$
  - Debt-to-Income (DTI): $\text{DTI}_i \sim \mathcal{U}(\text{DTI}_{\min}^{(g)}, \text{DTI}_{\max}^{(g)})$
  - Repayment History: $H_i \sim \text{Beta}(a_g, b_g)$
  - Requested Loan Principal: $L_i \sim \text{DiscreteUniform}(\mathcal{S}_L^{(g)})$

```text
Algorithm 1: GenerateMicrofinanceCohort
Input:
  - n_applicants: Total number of loan applicants N in N_+
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
  4:    // Assign demographic group membership
  5:    group_label ← Sample from {"Group_A_Marginalized", "Group_B_General"} with weights [p_A, p_B]
  6:    
  7:    If group_label == "Group_A_Marginalized" then:
  8:        income_i ← Uniform(7000.0, 20000.0)
  9:        dti_i ← Uniform(0.20, 0.65)
 10:        repay_i ← Beta(alpha=6.0, beta=2.0)
 11:        poverty_i ← Uniform(0.55, 0.95)
 12:        dependents_i ← DiscreteChoice([3, 4, 5, 6], weights=[0.25, 0.35, 0.25, 0.15])
 13:        women_led_i ← Bernoulli(p=0.80)
 14:        amount_i ← DiscreteChoice([15000, 20000, 25000, 30000], weights=[0.3, 0.4, 0.2, 0.1])
 15:    Else:
 16:        income_i ← Uniform(18000.0, 45000.0)
 17:        dti_i ← Uniform(0.15, 0.50)
 18:        repay_i ← Beta(alpha=7.0, beta=2.0)
 19:        poverty_i ← Uniform(0.15, 0.50)
 20:        dependents_i ← DiscreteChoice([1, 2, 3, 4], weights=[0.35, 0.35, 0.20, 0.10])
 21:        women_led_i ← Bernoulli(p=0.45)
 22:        amount_i ← DiscreteChoice([25000, 35000, 45000, 50000], weights=[0.2, 0.4, 0.25, 0.15])
 23:    End If
 24:
 25:    D[i] ← { "applicant_id": Format("MFI_APP_{:03d}", i),
 26:             "demographic_group": group_label,
 27:             "monthly_income": Round(income_i, 2),
 28:             "dti_ratio": Round(dti_i, 3),
 29:             "repayment_history": Round(repay_i, 3),
 30:             "poverty_index": Round(poverty_i, 3),
 31:             "dependents": dependents_i,
 32:             "is_women_led": women_led_i,
 33:             "requested_amount": amount_i }
 34: End For
 35: Return D
```
* **Complexity**: Time $\mathcal{O}(N)$, Space $\mathcal{O}(N)$.

---

### Algorithm 2: Supervised AI Repayment Modeling & Probability Calibration
* **Module**: [`src/repayment_model.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/repayment_model.py)
* **Purpose**: Estimates the empirical posterior probability of loan repayment $p_i = \mathbb{P}(\text{Repayment}=1 \mid \vec{x}_i^{\text{feat}})$ using supervised classifiers with Platt/Isotonic probability calibration.
* **Fairness Guardrail**: Protected demographic attributes (gender, marginalized group) are **strictly isolated and excluded** from feature matrix $X$ to prevent direct algorithmic discrimination.

```text
Algorithm 2: TrainAndCalibrateRepaymentModel
Input:
  - Dataset D: Applicant cohort table from Algorithm 1
  - model_type: Classifier architecture ("logistic" or "calibrated_rf")
  - calibration_method: "sigmoid" (Platt scaling) or "isotonic"
  - cv_folds: Number of cross-validation splits for calibration

Output:
  - RepaymentProbabilities: Vector p in [0, 1]^N
  - ModelMetrics: Evaluation dictionary (ROC-AUC, Brier Score, Accuracy, F1)

Procedure:
  1: // Step A: Feature extraction with fairness isolation
  2: FeatureCols ← ["requested_amount", "monthly_income", "dti_ratio",
  3:                 "repayment_history", "poverty_index", "dependents"]
  4: X ← D[FeatureCols].to_numpy()
  5: 
  6: // Step B: Derive latent ground-truth repayment target if not present
  7: If "repaid" in D then:
  8:     y ← D["repaid"].to_numpy()
  9: Else:
 10:     income_ratio ← D["monthly_income"] / Median(D["monthly_income"])
 11:     z ← 2.5 * D["repayment_history"] - 1.8 * D["dti_ratio"] + 0.6 * income_ratio - 1.0 * D["poverty_index"] - 0.2 * (D["dependents"] / 6.0)
 12:     p_latent ← 1.0 / (1.0 + exp(-z))
 13:     y ← (UniformRandom(N) < p_latent).astype(int)
 14:     Ensure both classes {0, 1} are populated
 15: End If
 16:
 17: // Step C: Train supervised classifier with calibration
 18: If scikit-learn is available then:
 19:     If model_type == "logistic" then:
 20:         base_estimator ← LogisticRegression(max_iter=500, random_state=42)
 21:     Else:
 22:         base_estimator ← RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42)
 23:     End If
 24:
 25:     min_class_count ← min(count(y == 0), count(y == 1))
 26:     If min_class_count >= 3 then:
 27:         model ← CalibratedClassifierCV(base_estimator, method=calibration_method, cv=min(3, min_class_count))
 28:     Else:
 29:         model ← base_estimator
 30:     End If
 31:     model.fit(X, y)
 32:     p_repay ← model.predict_proba(X)[:, 1]
 33: Else:
 34:     // High-performance pure-NumPy analytical logistic fallback
 35:     X_norm ← (X - mean(X, axis=0)) / (std(X, axis=0) + 1e-6)
 36:     w_fallback ← [-0.4, 0.6, -1.2, 2.0, -1.0, -0.2]
 37:     p_repay ← 1.0 / (1.0 + exp(-(X_norm @ w_fallback)))
 38: End If
 39:
 40: // Step D: Evaluate calibration quality
 41: brier_loss ← mean((p_repay - y)^2)
 42: roc_auc ← ComputeAreaUnderROC(y, p_repay)
 43: Return p_repay, { "roc_auc": roc_auc, "brier_score": brier_loss }
```
* **Complexity**: Time $\mathcal{O}(K \cdot N \cdot d)$ (where $K$ = trees, $d$ = features), Space $\mathcal{O}(N \cdot d)$.

---

### Algorithm 3: Multi-Objective Tri-Scoring & Welfare Synthesis
* **Module**: [`src/scoring.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/scoring.py)
* **Purpose**: Synthesizes the tri-objective utility score $C_i$ balancing Financial Return ($F_i$), Humanitarian Need ($N_i$), and Social Impact Multiplier ($S_i$):
  $$C_i = \omega_F F_i + \omega_N N_i + \omega_S S_i, \quad \sum \omega = 1.0$$
* **Expected Return & Loss Formulation**:
  - Expected Repayment Value: $V_i = L_i \cdot p_i$
  - Expected Default Loss: $R_i = L_i \cdot (1 - p_i)$

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
  5:    // 1. Financial Viability Score F_i (Higher credit fidelity, moderate income, lower DTI)
  6:    F[i] ← 0.40 * (1.0 - Clamp(D["dti_ratio"][i], 0.0, 1.0))
  7:           + 0.40 * Clamp(D["repayment_history"][i], 0.0, 1.0)
  8:           + 0.20 * Clamp(D["monthly_income"][i] / (income_median * 1.5), 0.0, 1.0)
  9:    F[i] ← Clamp(F[i], 0.0, 1.0)
 10:
 11:    // 2. Urgent Need Index N_i (Higher poverty index, dependent count, lower income)
 12:    N_need[i] ← 0.50 * Clamp(D["poverty_index"][i], 0.0, 1.0)
 13:                + 0.30 * Clamp(D["dependents"][i] / 6.0, 0.0, 1.0)
 14:                + 0.20 * Clamp(1.0 - (D["monthly_income"][i] / (income_median * 2.0)), 0.0, 1.0)
 15:    N_need[i] ← Clamp(N_need[i], 0.0, 1.0)
 16:
 17:    // 3. Social Impact Multiplier S_i (Women-led enterprise, marginalized community, micro-scale)
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
* **Complexity**: Time $\mathcal{O}(N)$, Space $\mathcal{O}(N)$.

---

### Algorithm 4: Monte Carlo Tail-Risk Engine (VaR & CVaR Simulation)
* **Module**: [`src/risk_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/risk_engine.py)
* **Purpose**: Simulates stochastic default losses under credit uncertainty across $S = 2,500$ discrete Monte Carlo scenarios to estimate portfolio **Value-at-Risk ($\text{VaR}_\alpha$)** and **Conditional Value-at-Risk ($\text{CVaR}_\alpha$)**.
* **Mathematical Definition**:
  $$\text{Loss}^{(s)}(x) = \sum_{i=1}^N L_i \cdot x_i \cdot \delta_i^{(s)}, \quad \delta_i^{(s)} \sim \text{Bernoulli}(1 - p_i)$$
  $$\text{VaR}_\alpha(x) = \text{Quantile}_\alpha(\{\text{Loss}^{(s)}(x)\}_{s=1}^S)$$
  $$\text{CVaR}_\alpha(x) = \mathbb{E}[\text{Loss}(x) \mid \text{Loss}(x) \ge \text{VaR}_\alpha(x)]$$

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
  2: // Pre-generate discrete scenario default indicators (S x N matrix)
  3: RandomMatrix ← Matrix of Uniform(0, 1) of shape (S, N)
  4: DefaultMatrix ← (RandomMatrix < q).astype(float)  // (s, i) = 1 if applicant i defaults
  5:
  6: // Vectorized computation of portfolio losses across all S scenarios
  7: CapitalAtRisk ← w * x  // element-wise product of shape (N,)
  8: ScenarioLosses ← DefaultMatrix @ CapitalAtRisk  // matrix-vector product of shape (S,)
  9:
 10: // Sort scenario losses in ascending order
 11: SortedLosses ← SortAscending(ScenarioLosses)
 12:
 13: // Compute Risk Metrics
 14: expected_loss ← Mean(SortedLosses)
 15: var_index ← Floor(alpha * S)
 16: var_alpha ← SortedLosses[var_index]
 17:
 18: // Conditional expectation in the upper (1 - alpha) tail
 19: TailLosses ← SortedLosses[var_index : S]
 20: cvar_alpha ← Mean(TailLosses)
 21: max_loss ← sum(CapitalAtRisk)
 22: loss_std ← StandardDeviation(SortedLosses)
 23:
 24: Return { "expected_loss": Round(expected_loss, 2),
 25:          "var_95": Round(var_alpha, 2),
 26:          "cvar_95": Round(cvar_alpha, 2),
 27:          "max_loss": Round(max_loss, 2),
 28:          "loss_std": Round(loss_std, 2) }
```
* **Complexity**: Time $\mathcal{O}(S \cdot N + S \log S)$, Space $\mathcal{O}(S \cdot N)$.

---

### Algorithm 5: Multi-Objective Constrained Knapsack to QUBO Mapping
* **Module**: [`src/qubo_builder.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/qubo_builder.py)
* **Purpose**: Constructs the upper-triangular QUBO matrix $Q \in \mathbb{R}^{N \times N}$ encoding the multi-objective utility, quadratic budget capacity penalty, and demographic parity balance constraint.
* **Penalty Energy Functional**:
  $$E(x) = -\sum_{i=1}^N C_i x_i + \lambda_B \left( \sum_{i=1}^N w_i x_i - B \right)^2 + \lambda_F \left( \frac{\sum_{i \in G_A} x_i}{|G_A|} - \frac{\sum_{j \in G_B} x_j}{|G_B|} \right)^2$$
  Since $x_i \in \{0, 1\}$, idempotent identity $x_i^2 = x_i$ is used to absorb linear terms into the diagonal $Q_{ii}$.

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
  6: // Define fairness parity sensitivity coefficients sigma_i
  7: sigma ← zeros(N)
  8: For i ← 1 to N do:
  9:    sigma[i] ← (+ 1.0 / N_A) if G[i] == 0 else (- 1.0 / N_B)
 10: End For
 11:
 12: // Diagonal Elements: Linear utility + self-penalty expansions (x_i^2 = x_i)
 13: For i ← 1 to N do:
 14:    term_utility ← - C[i]
 15:    term_budget_linear ← lambda_B * (w[i]^2 - 2.0 * B * w[i])
 16:    term_fairness_linear ← lambda_F * (sigma[i]^2)
 17:    Q[i, i] ← term_utility + term_budget_linear + term_fairness_linear
 18: End For
 19:
 20: // Off-Diagonal Elements: Pairwise quadratic interaction cross-terms (2 * x_i * x_j)
 21: For i ← 1 to N do:
 22:    For j ← (i + 1) to N do:
 23:        cross_budget ← 2.0 * lambda_B * w[i] * w[j]
 24:        cross_fairness ← 2.0 * lambda_F * sigma[i] * sigma[j]
 25:        Q[i, j] ← cross_budget + cross_fairness
 26:    End For
 27: End For
 28:
 29: E_offset ← lambda_B * (B^2)
 30: Return Q, E_offset
```
* **Complexity**: Time $\mathcal{O}(N^2)$, Space $\mathcal{O}(N^2)$.

---

### Algorithm 6: QUBO to Ising Spin Glass Transformation
* **Module**: [`src/qubo_builder.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/qubo_builder.py)
* **Purpose**: Transforms binary variables $x_i \in \{0, 1\}$ into quantum spin operators $Z_i \in \{+1, -1\}$ via affine mapping $x_i = \frac{I - Z_i}{2}$ to synthesize the Problem Hamiltonian:
  $$H_C = \sum_{i=1}^N h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \text{offset} \cdot I$$

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
  5: // Step 1: Compute Two-Qubit Coupling J_ij
  6: For i ← 1 to N do:
  7:    For j ← (i + 1) to N do:
  8:        J[i, j] ← 0.25 * Q[i, j]
  9:    End For
 10: End For
 11:
 12: // Step 2: Compute Single-Qubit Field h_i
 13: For i ← 1 to N do:
 14:    sum_couplings ← sum(Q[i, j] for j > i) + sum(Q[j, i] for j < i)
 15:    h[i] ← - 0.5 * Q[i, i] - 0.25 * sum_couplings
 16: End For
 17:
 18: // Step 3: Compute Global Hamiltonian Energy Shift
 19: sum_diag ← sum(Q[i, i] for i in 1..N)
 20: sum_upper ← sum(Q[i, j] for i in 1..N, j > i)
 21: ising_offset ← ising_offset + 0.5 * sum_diag + 0.25 * sum_upper
 22:
 23: Return h, J, ising_offset
```
* **Complexity**: Time $\mathcal{O}(N^2)$, Space $\mathcal{O}(N^2)$.

---

### Algorithm 7: Classical Optimization Solvers Suite
* **Module**: [`src/classical_solvers.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/classical_solvers.py)
* **Purpose**: Provides four distinct classical optimization benchmarks to evaluate quantum performance against industry-standard heuristics and exact global ground truths.

```text
Algorithm 7: ClassicalSolversSuite

// Subroutine 7A: Greedy Ratio Knapsack Heuristic
Procedure SolveGreedyKnapsack(C, w, B):
  1: ratios ← C / w  // Utility per unit capital
  2: sorted_order ← SortIndicesDescending(ratios)
  3: x_greedy ← zeros(N); current_spent ← 0.0
  4: For each index i in sorted_order do:
  5:     If current_spent + w[i] <= B then:
  6:         x_greedy[i] ← 1
  7:         current_spent ← current_spent + w[i]
  8:     End If
  9: End For
 10: Return x_greedy, current_spent

// Subroutine 7B: Exact Combinatorial Ground-State Solver (Global Ground Truth)
Procedure SolveExactCombinatorial(diag_energies):
  1: best_idx ← ArgMin(diag_energies)  // Across all 2^N state eigenvalues
  2: x_exact ← BinaryVectorFromIndex(best_idx, length=N)
  3: Return x_exact, diag_energies[best_idx]

// Subroutine 7C: Exact Classical Mixed-Integer Linear Programming (MILP)
Procedure SolveClassicalMILP(C, w, B, G, tolerance_epsilon):
  1: Objective: Minimize - C^T x
  2: Subject to:
  3:    0 <= w^T x <= B                                 (Budget Capacity)
  4:    -epsilon <= (1/N_A sum_{i in A} x_i) - (1/N_B sum_{j in B} x_j) <= epsilon  (Fairness Parity)
  5:    x_i in {0, 1} for all i
  6: Execute ScipyBranchAndCutSimplex(c=-C, constraints, integrality=ones(N))
  7: Return x_milp, status

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
* **Complexity**:
  - Greedy: $\mathcal{O}(N \log N)$ Time, $\mathcal{O}(N)$ Space
  - Exact Brute-Force: $\mathcal{O}(2^N)$ Time, $\mathcal{O}(2^N)$ Space
  - MILP: $\mathcal{O}(2^N)$ worst-case, $\mathcal{O}(N)$ practical via branch-and-cut
  - Simulated Annealing: $\mathcal{O}(\text{steps} \cdot N)$ Time, $\mathcal{O}(N)$ Space.

---

### Algorithm 8: Standard Quantum Approximate Optimization Algorithm (QAOA)
* **Module**: [`src/quantum_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/quantum_engine.py)
* **Purpose**: Implements problem-tailored QAOA with $p$ alternating cost and mixer layers, minimizing the standard expectation value $\langle \psi(\vec{\gamma}, \vec{\beta}) | H_C | \psi(\vec{\gamma}, \vec{\beta}) \rangle$ via classical COBYLA parameter updates.
* **Unitary Operators**:
  - Problem Unitary: $U_C(\gamma) = e^{-i \gamma H_C} = \prod_{i=1}^N R_Z(2 \gamma h_i) \prod_{i < j} R_{ZZ}(2 \gamma J_{ij})$
  - Transverse Mixer Unitary: $U_M(\beta) = e^{-i \beta \sum X_i} = \prod_{i=1}^N R_X(2 \beta)$

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
  1: // Initialize uniform quantum superposition state |+>^N
  2: |psi_0> ← (1 / sqrt(2^N)) * sum_{k=0}^{2^N - 1} |k>
  3: 
  4: // Parameter initialization
  5: gamma_init ← UniformRandom(0.1, pi, size=p)
  6: beta_init ← UniformRandom(0.1, pi / 2.0, size=p)
  7: params_init ← Concatenate([gamma_init, beta_init])
  8:
  9: Define ObjectiveFunction(params):
 10:     gamma_vec ← params[0 : p]
 11:     beta_vec ← params[p : 2p]
 12:     
 13:     // Statevector evolution across p layers
 14:     |psi> ← |psi_0>
 15:     For k ← 0 to (p - 1) do:
 16:         |psi> ← ApplyDiagonalCostUnitary(|psi>, gamma_vec[k])
 17:         |psi> ← ApplyTransverseMixerUnitary(|psi>, beta_vec[k])
 18:     End For
 19:     
 20:     // Compute standard expectation energy <psi | H_C | psi>
 21:     probs ← |psi|^2
 22:     expectation_energy ← sum(probs[k] * diag_energies[k] for k in 0..2^N - 1)
 23:     Return expectation_energy
 24:
 25: // Classical variational optimization loop
 26: opt_result ← Minimize(ObjectiveFunction, params_init, method="COBYLA", maxiter=max_iter)
 27:
 28: // Synthesize final statevector with optimal parameters
 29: |psi_opt> ← EvolveState(|psi_0>, opt_result.x[:p], opt_result.x[p:])
 30: probs_opt ← |psi_opt|^2
 31: 
 32: // Measurement sampling: select maximum likelihood basis state
 33: best_state_idx ← ArgMax(probs_opt)
 34: x_optimal ← BitstringFromInteger(best_state_idx, length=N)
 35: Return x_optimal, opt_result.fun, probs_opt
```
* **Complexity**: Time $\mathcal{O}(\text{evals} \cdot p \cdot 2^N)$, Space $\mathcal{O}(2^N)$ statevector amplitudes.

---

### Algorithm 9: Tail-Risk Conditional Value-at-Risk QAOA (CVaR-QAOA)
* **Module**: [`src/quantum_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/quantum_engine.py#L110-L140)
* **Purpose**: Implements risk-aware quantum optimization following [Barkoutsos et al. (2020)](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/algorithmic_study.md#L275). Instead of global expectation, it optimizes over the lowest $\alpha$-quantile of the Hamiltonian spectrum, filtering out high-energy sub-optimal states and concentrating quantum probability on ground-state allocations.
* **Objective Function**:
  $$\text{CVaR}_\alpha(H_C) = \frac{1}{\alpha} \left[ \sum_{j=0}^{K_\alpha - 1} P_{(j)} E_{(j)} + \left(\alpha - \sum_{j=0}^{K_\alpha - 1} P_{(j)}\right) E_{(K_\alpha)} \right]$$

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
 12:     // Step A: Sort Hamiltonian eigenvalues in non-decreasing order
 13:     sort_order ← ArgsortAscending(diag_energies)
 14:     sorted_E ← diag_energies[sort_order]
 15:     sorted_P ← probs[sort_order]
 16:     
 17:     // Step B: Determine quantile cutoff index K_alpha
 18:     cum_P ← CumulativeSum(sorted_P)
 19:     k_alpha ← BinarySearch(cum_P, alpha)
 20:     
 21:     // Step C: Analytical tail-risk expectation calculation
 22:     P_head ← sorted_P[0 : k_alpha]
 23:     E_head ← sorted_E[0 : k_alpha]
 24:     residual_P ← alpha - sum(P_head)
 25:     
 26:     cvar_val ← (sum(P_head * E_head) + max(0.0, residual_P) * sorted_E[k_alpha]) / alpha
 27:     Return cvar_val
 28:
 29: opt_res ← ClassicalOptimizer(CVaR_Objective, params_init, method="COBYLA")
 30: |psi_final> ← EvolveCircuit(|psi_0>, opt_res.x, p)
 31: x_cvar ← ArgMaxBasisState(|psi_final|^2)
 32: Return x_cvar, opt_res.fun, |psi_final|^2
```
* **Complexity**: Time $\mathcal{O}(\text{evals} \cdot (p \cdot 2^N + 2^N \log(2^N)))$, Space $\mathcal{O}(2^N)$.

---

### Algorithm 10: Variational Quantum Eigensolver (VQE) with Hardware-Efficient Ansatz
* **Module**: [`src/quantum_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/quantum_engine.py#L240)
* **Purpose**: Solves the resource allocation problem using an unconstrained hardware-efficient ansatz (TwoLocal: parameterized $R_y(\theta)$ rotation gates interleaved with circular Controlled-NOT entangling gates).

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
* **Complexity**: Time $\mathcal{O}(\text{evals} \cdot L \cdot 2^N)$, Space $\mathcal{O}(2^N)$.

---

### Algorithm 11: Native Qiskit Circuit Synthesis & Gate Decomposition
* **Module**: [`src/quantum_engine.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/quantum_engine.py#L300), [`visuals/qiskit_qaoa_circuit.txt`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/visuals/qiskit_qaoa_circuit.txt)
* **Purpose**: Synthesizes native parameterized Qiskit `QuantumCircuit` objects and exports OpenQASM/ASCII gate diagrams for deployment on IBM Quantum superconducting processors.

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
  1: Initialize QuantumRegister of size N, ClassicalRegister of size N
  2: qc ← QuantumCircuit(qreg, creg)
  3: 
  4: // Step 1: Initial Hadamard state preparation |+>^N
  5: For each qubit q in 0 to (N - 1) do:
  6:     qc.h(q)
  7: End For
  8: qc.barrier()
  9:
 10: // Step 2: Assemble p alternating cost and mixer layers
 11: For layer k ← 1 to p do:
 12:     gamma_k ← Parameter(Format("gamma_{}", k))
 13:     beta_k ← Parameter(Format("beta_{}", k))
 14:     
 15:     // Problem Cost Unitary U_C(gamma_k)
 16:     For i ← 0 to (N - 1) do:
 17:         If abs(h[i]) > 1e-5 then:
 18:             qc.rz(2.0 * h[i] * gamma_k, i)
 19:         End If
 20:     End For
 21:     For i ← 0 to (N - 1) do:
 22:         For j ← (i + 1) to (N - 1) do:
 23:             If abs(J[i, j]) > 1e-5 then:
 24:                 qc.rzz(2.0 * J[i, j] * gamma_k, i, j)
 25:             End If
 26:         End For
 27:     End For
 28:     qc.barrier()
 29:     
 30:     // Transverse Mixer Unitary U_M(beta_k)
 31:     For i ← 0 to (N - 1) do:
 32:         qc.rx(2.0 * beta_k, i)
 33:     End For
 34:     qc.barrier()
 35: End For
 36:
 37: // Step 3: Computational basis measurement
 38: qc.measure(qreg, creg)
 39: ascii_diagram ← qc.draw(output="text")
 40: Return qc, ascii_diagram
```
* **Complexity**: Time $\mathcal{O}(p \cdot N^2)$, Space $\mathcal{O}(p \cdot N^2)$ gate objects.

---

### Algorithm 12: Multi-Objective Fairness & Tail-Risk Evaluation
* **Module**: [`src/metrics.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/src/metrics.py)
* **Purpose**: Conducts comprehensive mathematical auditing of allocation vector $x$ across capital expenditure, demographic fairness parity, disparate impact, portfolio tail loss, and quantum approximation ratio.

```text
Algorithm 12: EvaluateAllocationMetrics
Input:
  - Allocation vector x in {0, 1}^N
  - Applicant Table D_scored (containing scores C, F, N_need, S, amounts w, groups G)
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
  7: // Demographic Parity Auditing
  8: mask_A ← (G == 0); mask_B ← (G == 1)
  9: n_A ← max(1, count(mask_A)); n_B ← max(1, count(mask_B))
 10: rate_A ← sum(x[mask_A]) / n_A
 11: rate_B ← sum(x[mask_B]) / n_B
 12: 
 13: // Demographic Parity Difference (DPD): |rate_A - rate_B|
 14: DPD ← abs(rate_A - rate_B)
 15:
 16: // Disparate Impact Ratio (DIR / 80% Rule): rate_A / rate_B
 17: If rate_B > 0 then:
 18:     DIR ← rate_A / rate_B
 19: Else:
 20:     DIR ← 1.0 if rate_A == 0 else +infinity
 21: End If
 22:
 23: // Tail-Risk Assessment
 24: risk_stats ← RiskEngine.Evaluate(x, alpha=0.95)
 25:
 26: // Quantum Approximation Ratio
 27: approx_ratio ← TotalUtility / C_optimal if C_optimal > 0 else 1.0
 28:
 29: Return { "solver": solver_name,
 30:          "total_approved": TotalApproved,
 31:          "total_cost": TotalCost,
 32:          "budget_utilization_pct": budget_utilization,
 33:          "is_feasible": is_feasible,
 34:          "total_utility": TotalUtility,
 35:          "expected_loss": risk_stats["expected_loss"],
 36:          "cvar_95_loss": risk_stats["cvar_95"],
 37:          "rate_group_A": rate_A,
 38:          "rate_group_B": rate_B,
 39:          "demographic_parity_diff": DPD,
 40:          "disparate_impact_ratio": DIR,
 41:          "approximation_ratio": approx_ratio }
```
* **Complexity**: Time $\mathcal{O}(N + S)$, Space $\mathcal{O}(1)$.

---

### Algorithm 13: End-to-End Benchmark Pipeline Orchestrator
* **Module**: [`midsem_pipeline.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/midsem_pipeline.py)
* **Purpose**: Master execution script that coordinates data generation, AI repayment calibration, QUBO synthesis, 8 classical/quantum solvers, multi-metric auditing, and publication visual analytics generation.

```text
Algorithm 13: RunMidsemBenchmarkPipeline
Procedure:
  1: cohort ← GenerateMicrofinanceCohort(N=8, seed=42)                 // Algorithm 1
  2: p_repay, ml_metrics ← TrainAndCalibrateRepaymentModel(cohort)       // Algorithm 2
  3: scored_cohort, C, G ← ComputeMultiobjectiveTriScoring(cohort, p)  // Algorithm 3
  4: risk_engine ← InitializeRiskEngine(cohort.amounts, 1.0 - p_repay) // Algorithm 4
  5: Q, E_offset ← BuildQUBOMatrix(C, cohort.amounts, G, Budget=110k)   // Algorithm 5
  6: h, J, ising_shift ← TransformQUBOToIsing(Q, E_offset)             // Algorithm 6
  7: 
  8: // Solve across 8 distinct paradigms
  9: res_greedy ← SolveGreedyKnapsack(C, cohort.amounts, Budget=110k)    // Algorithm 7A
 10: res_exact  ← SolveExactCombinatorial(GetDiagonal(Q))              // Algorithm 7B
 11: res_milp   ← SolveClassicalMILP(C, cohort.amounts, Budget=110k, G) // Algorithm 7C
 12: res_sa     ← SolveSimulatedAnnealing(Q, E_offset)                 // Algorithm 7D
 13: res_qaoa_p1 ← StandardQAOA(h, J, ising_shift, p=1)                // Algorithm 8
 14: res_qaoa_p2 ← StandardQAOA(h, J, ising_shift, p=2)                // Algorithm 8
 15: res_cvar_qaoa ← CVaR_QAOA(h, J, ising_shift, p=1, alpha=0.25)     // Algorithm 9
 16: res_vqe    ← HardwareEfficientVQE(h, J, L=1)                      // Algorithm 10
 17:
 18: // Audit and compile 8-solver comparative benchmark table
 19: For each solver in {Greedy, Exact, MILP, SA, QAOA_p1, QAOA_p2, CVaR_QAOA, VQE} do:
 20:     metrics[solver] ← EvaluateAllocationMetrics(solver.allocation, scored_cohort, risk_engine) // Algorithm 12
 21: End For
 22:
 23: // Generate 5 high-resolution figures & export Qiskit circuit
 24: PlotQUBOHeatmap(Q, "visuals/qubo_matrix_heatmap.png")
 25: PlotMultiMetricComparison(metrics, "visuals/quantum_vs_classical_comparison.png")
 26: PlotConvergenceProfile(histories, "visuals/qaoa_convergence_profile.png")
 27: PlotParetoFrontier(varying lambda_F, "visuals/fairness_vs_budget_tradeoff.png")
 28: PlotQuantumProbability(res_qaoa_p2.probs, "visuals/bitstring_probability_distribution.png")
 29: ExportQiskitCircuit(h, J, p=1, "visuals/qiskit_qaoa_circuit.txt") // Algorithm 11
 30:
 31: PrintBenchmarkTable(metrics)
 32: Print("[SUCCESS] 70% Accomplishment Milestone Checkpoint Complete")
```
* **Complexity**: Time $\mathcal{O}(2^N + S \cdot N)$ dominated by quantum statevector evolution and Monte Carlo scenario calculation; executes in under 7 seconds for $N=8$.

---

## 3. Algorithm Complexity & Resource Summary Table

| Algorithm | Method / Paradigm | Primary Function | Time Complexity | Memory / Qubits | Target Deliverable |
|---|---|---|:---:|:---:|---|
| **Alg 1** | Empirical Sampling | Microfinance Cohort Generator | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ | `src/data_generator.py` |
| **Alg 2** | Supervised ML & Calibration | Repayment Prediction ($p_i$) | $\mathcal{O}(K \cdot N \cdot d)$ | $\mathcal{O}(N \cdot d)$ | `src/repayment_model.py` |
| **Alg 3** | Convex Synthesis | Tri-Objective Scoring ($C_i$) | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ | `src/scoring.py` |
| **Alg 4** | Monte Carlo Simulation | Portfolio Tail Risk (VaR/CVaR) | $\mathcal{O}(S \cdot N + S \log S)$ | $\mathcal{O}(S \cdot N)$ | `src/risk_engine.py` |
| **Alg 5** | Penalty Mapping | QUBO Matrix Assembly | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ | `src/qubo_builder.py` |
| **Alg 6** | Spin Operator Affine Map | Ising Hamiltonian ($H_C$) | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ | `src/qubo_builder.py` |
| **Alg 7A** | Greedy Ratio Heuristic | Classical Cost Heuristic | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N)$ | `src/classical_solvers.py` |
| **Alg 7B** | Exhaustive Search | Exact Combinatorial Ground State | $\mathcal{O}(2^N)$ | $\mathcal{O}(2^N)$ | `src/classical_solvers.py` |
| **Alg 7C** | Branch-and-Cut Simplex | Classical Exact MILP | $\mathcal{O}(2^N)$ worst-case | $\mathcal{O}(N)$ | `src/classical_solvers.py` |
| **Alg 7D** | Metropolis-Hastings MCMC | Simulated Annealing Heuristic | $\mathcal{O}(\text{steps} \cdot N)$ | $\mathcal{O}(N)$ | `src/classical_solvers.py` |
| **Alg 8** | Variational Quantum ($p=1, 2$) | Standard Expectation QAOA | $\mathcal{O}(\text{evals} \cdot p \cdot 2^N)$ | $N$ Qubits / $2^N$ Floats | `src/quantum_engine.py` |
| **Alg 9** | Risk-Aware Variational | Tail-Risk CVaR-QAOA ($\alpha=0.25$) | $\mathcal{O}(\text{evals} \cdot 2^N \log(2^N))$ | $N$ Qubits / $2^N$ Floats | `src/quantum_engine.py` |
| **Alg 10** | Hardware-Efficient Ansatz | TwoLocal Parameterized VQE | $\mathcal{O}(\text{evals} \cdot L \cdot 2^N)$ | $N$ Qubits / $2^N$ Floats | `src/quantum_engine.py` |
| **Alg 11** | Gate Decomposition | Native Qiskit Circuit Synthesis | $\mathcal{O}(p \cdot N^2)$ | $\mathcal{O}(p \cdot N^2)$ Gates | `visuals/qiskit_qaoa_circuit.txt` |
| **Alg 12** | Multi-Metric Auditing | Fairness & Approximation Auditing | $\mathcal{O}(N + S)$ | $\mathcal{O}(1)$ | `src/metrics.py` |
| **Alg 13** | Pipeline Orchestration | End-to-End Midsem Execution | $\mathcal{O}(2^N + S \cdot N)$ | $\mathcal{O}(2^N)$ | `midsem_pipeline.py` |
