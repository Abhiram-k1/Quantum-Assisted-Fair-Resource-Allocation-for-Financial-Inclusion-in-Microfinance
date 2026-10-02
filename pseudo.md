# Algorithmic Pseudocode Specification (`pseudo.md`)
## Quantum-Assisted Fair Resource Allocation for Financial Inclusion in Microfinance (Q-FAR)
**Course**: Quantum Computing and Advanced Algorithms (QCAA) | Semester-5 Capstone  
**Format Standard**: High-Performance Computing (HPC), Computer Vision & Quantum Optimization Rigor Standard

---

### Executive Overview & Algorithmic Index

This specification outlines the mathematical, procedural, and computational architecture of the **Q-FAR** pipeline. Each algorithm is specified with explicit dimensional typing, algorithmic invariants, **Asymptotic Time Complexity**, **Space / Memory Footprint**, and **High-Performance Computing (HPC) Parallelization Mapping** (SIMD, OpenMP multithreading, GPU tensor contraction, and quantum statevector gates).

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   Q-FAR ALGORITHMIC TAXONOMY                                     │
├───────────────────────────────┬──────────────────────────────────┬───────────────────────────────┤
│    AI & SCORING PIPELINE      │    CLASSICAL HPC SOLVER SUITE    │   QUANTUM VARIATIONAL ENGINES │
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│ • Alg 1: AI Repayment (CV)    │ • Alg 7: Greedy Ratio Knapsack   │ • Alg 5: QAOA Expectation     │
│ • Alg 2: Tri-Objective Welfare │ • Alg 8: HPC Exact MILP Solver   │ • Alg 6: Tail-Risk CVaR-QAOA  │
│ • Alg 3: QUBO Matrix Assembly │ • Alg 9: Simulated Annealing     │ • Alg 10: VQE RealAmplitudes  │
│ • Alg 4: QUBO-to-Ising Glass  │ • Alg 11: Monte Carlo Tail Risk  │ • Alg 12: Fairness Auditing   │
└───────────────────────────────┴──────────────────────────────────┴───────────────────────────────┘
```

---

### Algorithm 1: Calibrated Supervised AI Repayment Prediction (Cross-Validation / CV)
**Goal**: Estimate posterior repayment probabilities $p_i = P(\text{repayment} = 1 \mid X_i)$ using supervised machine learning with $K$-fold stratified probability calibration, isolating protected demographic attributes to prevent direct algorithmic bias amplification.

```text
Input:
  - Feature matrix X in R^{M x D} where M is historical borrowers, D is credit attributes:
    { monthly_income, dti_ratio, repayment_history, poverty_index, dependents, requested_amount }
  - Ground truth binary labels y in {0, 1}^M (1: Repaid, 0: Default)
  - Target applicant feature matrix X_eval in R^{N x D} for the N active applicants
  - Number of calibration folds K in {2, 3, 5}
  - Calibration method: Platt Sigmoid or Isotonic Regression

Output:
  - Calibrated posterior repayment probabilities p in [0, 1]^N
  - Expected individual default probabilities q = (1 - p) in [0, 1]^N
  - Out-of-fold generalization metrics: ROC-AUC, PR-AUC, Brier Score

Procedure:
  1: Standardize continuous features:
  2:    mu ← (1 / M) * sum_{m=1}^M X[m, :]
  3:    sigma ← sqrt((1 / M) * sum_{m=1}^M (X[m, :] - mu)^2) + 1e-6
  4:    X_norm ← (X - mu) / sigma
  5:
  6: Partition index set {1, ..., M} into K stratified folds {F_1, ..., F_K} preserving class ratio
  7: For fold k from 1 to K do:
  8:    X_train, y_train ← X_norm[not in F_k], y[not in F_k]
  9:    X_val, y_val ← X_norm[in F_k], y[in F_k]
 10:
 11:    // Fit base ensemble (Random Forest with B trees or Regularized Logistic)
 12:    f_base_k ← TrainClassifier(X_train, y_train)
 13:    z_val ← f_base_k(X_val)  // Uncalibrated margins
 14:
 15:    // Platt Calibrator: fit logistic sigmoid P(y=1 | z) = 1 / (1 + exp(A * z + B))
 16:    (A_k, B_k) ← argmin_{A, B} - sum_{j in F_k} [ y_j * ln(sigma(A * z_j + B)) + (1 - y_j) * ln(1 - sigma(A * z_j + B)) ]
 17: End For
 18:
 19: // Aggregate ensemble model and predict on active applicant cohort
 20: X_eval_norm ← (X_eval - mu) / sigma
 21: For applicant i from 1 to N do:
 22:    p[i] ← (1 / K) * sum_{k=1}^K 1 / (1 + exp(A_k * f_base_k(X_eval_norm[i]) + B_k))
 23:    q[i] ← 1.0 - p[i]
 24: End For
 25: Return p, q
```

* **Time Complexity**: $\mathcal{O}(K \cdot (M \cdot D \log M + B \cdot D \cdot M) + N \cdot B \cdot D)$ where $B$ is the number of ensemble decision trees.
* **Space Complexity**: $\mathcal{O}(M \cdot D + N \cdot D)$ storing normalized feature tables and tree structures.
* **HPC Parallel Mapping**: Embarrassingly parallel across decision trees and cross-validation folds using OpenMP / SIMD multi-threading.

---

### Algorithm 2: Multi-Objective Feature Welfare Scoring & Normalization
**Goal**: Synthesize multi-dimensional applicant attributes into a bounded scalar composite welfare score $C_i \in [0, 1]$ balancing financial viability ($F_i$), urgent humanitarian need ($N_i$), and social community impact ($S_i$).

```text
Input:
  - Applicant attributes: income in R^N, dti in R^N, repay_hist in [0, 1]^N, poverty in [0, 1]^N,
    deps in N^N, women_led in {0, 1}^N, group in {"Group_A", "Group_B"}^N, requested_amount w in R^N
  - Policy weights vector: omega = (omega_F, omega_N, omega_S) such that sum(omega) = 1.0

Output:
  - Normalized composite welfare vector C in [0, 1]^N
  - Group indicator vector G in {0, 1}^N (0: Marginalized Group A, 1: General Group B)

Procedure:
  1: median_income ← Median(income)
  2: For i from 1 to N do:
  3:    // 1. Financial Viability Score F_i
  4:    F_i ← 0.40 * (1.0 - clamp(dti[i], 0.0, 1.0)) + 0.40 * repay_hist[i] + 0.20 * clamp(income[i] / (1.5 * median_income), 0.0, 1.0)
  5:
  6:    // 2. Urgent Need Index N_i
  7:    N_i ← 0.50 * poverty[i] + 0.30 * clamp(deps[i] / 6.0, 0.0, 1.0) + 0.20 * clamp(1.0 - income[i] / (2.0 * median_income), 0.0, 1.0)
  8:
  9:    // 3. Social Inclusion & Community Impact S_i
 10:    boost_women ← 1.0 if women_led[i] == True else 0.25
 11:    boost_group ← 1.0 if group[i] == "Group_A" else 0.35
 12:    boost_loan_scale ← clamp(1.0 - w[i] / 60000.0, 0.20, 1.0)
 13:    S_i ← 0.45 * boost_women + 0.35 * boost_group + 0.20 * boost_loan_scale
 14:
 15:    // 4. Convex Composite Utility
 16:    C[i] ← omega_F * clamp(F_i, 0, 1) + omega_N * clamp(N_i, 0, 1) + omega_S * clamp(S_i, 0, 1)
 17:    G[i] ← 0 if group[i] == "Group_A" else 1
 18: End For
 19: Return C, G
```

* **Time Complexity**: $\mathcal{O}(N)$ linear scan over applicant arrays.
* **Space Complexity**: $\mathcal{O}(N)$ in-place contiguous memory allocations.
* **HPC Parallel Mapping**: Vectorized SIMD loop with zero synchronization dependencies across applicants.

---

### Algorithm 3: Multi-Objective Constrained QUBO Penalty Matrix Assembly
**Goal**: Map the constrained multi-objective knapsack problem into an unconstrained Quadratic Unconstrained Binary Optimization (QUBO) matrix $Q \in \mathbb{R}^{N \times N}$ via quadratic penalty expansion.

$$\min_{x \in \{0, 1\}^N} E(x) = -\sum_{i=1}^N C_i x_i + \lambda_B \left( \sum_{i=1}^N w_i x_i - B \right)^2 + \lambda_F \left( \frac{\sum_{i \in G_A} x_i}{|G_A|} - \frac{\sum_{j \in G_B} x_j}{|G_B|} \right)^2$$

```text
Input:
  - Composite utility vector C in R^N, loan amounts w in R^N, group indicators G in {0, 1}^N
  - Capital budget ceiling B in R^+
  - Penalty multipliers: lambda_B (budget enforcement), lambda_F (demographic fairness)

Output:
  - Upper-triangular QUBO matrix Q in R^{N x N}
  - Scalar constant offset E_offset in R

Procedure:
  1: Q ← zeros(N, N)
  2: N_A ← max(1, count(i for which G[i] == 0))
  3: N_B ← max(1, count(i for which G[i] == 1))
  4:
  5: // Compute demographic parity directional weights sigma_i
  6: For i from 1 to N do:
  7:    sigma[i] ← (+ 1.0 / N_A) if G[i] == 0 else (- 1.0 / N_B)
  8: End For
  9:
 10: // Diagonal entries Q_{ii}: Linear utility + squared self-penalties (since x_i^2 = x_i)
 11: For i from 1 to N do:
 12:    term_util ← - C[i]
 13:    term_budget ← lambda_B * (w[i]^2 - 2.0 * B * w[i])
 14:    term_fair ← lambda_F * (sigma[i]^2)
 15:    Q[i, i] ← term_util + term_budget + term_fair
 16: End For
 17:
 18: // Off-diagonal entries Q_{ij} (j > i): Pairwise cross-interaction couplings
 19: For i from 1 to N do:
 20:    For j from (i + 1) to N do:
 21:        cross_budget ← 2.0 * lambda_B * w[i] * w[j]
 22:        cross_fair ← 2.0 * lambda_F * sigma[i] * sigma[j]
 23:        Q[i, j] ← cross_budget + cross_fair
 24:    End For
 25: End For
 26: E_offset ← lambda_B * (B^2)
 27: Return Q, E_offset
```

* **Time Complexity**: $\mathcal{O}(N^2)$ pairwise upper-triangular calculation.
* **Space Complexity**: $\mathcal{O}(N^2)$ dense matrix storage in cache-friendly row-major order.
* **HPC Parallel Mapping**: Outer loop parallelization via `#pragma omp parallel for schedule(dynamic)`.

---

### Algorithm 4: Affine QUBO-to-Ising Spin Glass Hamiltonian Transformation
**Goal**: Transform binary optimization variables $x_i \in \{0, 1\}$ to quantum Pauli-Z spin operators $Z_i \in \{+1, -1\}$ via $x_i = \frac{I - Z_i}{2}$ to yield the cost Hamiltonian $H_C = \sum_{i} h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \text{offset} \cdot I$.

```text
Input:
  - Upper-triangular QUBO matrix Q in R^{N x N}
  - Scalar offset E_offset in R

Output:
  - Longitudinal magnetic field vector h in R^N
  - Exchange coupling interaction matrix J in R^{N x N} (upper triangular)
  - Shifted scalar energy offset ising_offset in R

Procedure:
  1: h ← zeros(N)
  2: J ← zeros(N, N)
  3:
  4: // Pairwise exchange couplings: J_{ij} = (1 / 4) * Q_{ij}
  5: For i from 1 to N do:
  6:    For j from (i + 1) to N do:
  7:        J[i, j] ← 0.25 * Q[i, j]
  8:    End For
  9: End For
 10:
 11: // Longitudinal fields: h_i = - (1 / 2) * Q_{ii} - (1 / 4) * sum_{j != i} Q_{ij}
 12: For i from 1 to N do:
 13:    sum_row_col ← sum(Q[i, j] for j > i) + sum(Q[j, i] for j < i)
 14:    h[i] ← - 0.5 * Q[i, i] - 0.25 * sum_row_col
 15: End For
 16:
 17: // Energy identity shift
 18: sum_diag ← sum(Q[i, i] for i from 1 to N)
 19: sum_couplings ← sum(Q[i, j] for i from 1 to N for j from (i + 1) to N)
 20: ising_offset ← E_offset + 0.5 * sum_diag + 0.25 * sum_couplings
 21: Return h, J, ising_offset
```

* **Time Complexity**: $\mathcal{O}(N^2)$ matrix reduction.
* **Space Complexity**: $\mathcal{O}(N^2)$ memory for exchange interaction matrix.
* **HPC Parallel Mapping**: BLAS-2 matrix-vector kernel execution with SIMD fused multiply-add (FMA).

---

### Algorithm 5: Standard Quantum Approximate Optimization Algorithm (QAOA)
**Goal**: Prepare variational state $|\psi(\vec{\gamma}, \vec{\beta})\rangle = \prod_{l=1}^p \left( e^{-i \beta_l \sum X_i} e^{-i \gamma_l H_C} \right) |+\rangle^{\otimes N}$ and classical optimize $(\vec{\gamma}, \vec{\beta})$ to minimize global expectation $\langle H_C \rangle$.

```text
Input:
  - Cost Hamiltonian H_C = (h, J, ising_offset)
  - Circuit depth p in N^+ (layers)
  - Classical optimizer: COBYLA or SLSQP (max_iter, tolerance)

Output:
  - Optimal parameter vectors (gamma_opt, beta_opt) in R^p x R^p
  - Most probable sampled loan allocation bitstring x_QAOA in {0, 1}^N
  - Ground-state expectation energy E_opt

Procedure:
  1: Initialize uniform superposition: |psi_0⟩ ← H^{⊗N} |0⟩^{⊗N} = (1 / sqrt(2^N)) * sum_{x in {0, 1}^N} |x⟩
  2: Precompute diagonal eigenenergies E_k = ⟨k| H_C |k⟩ for all k in {0, ..., 2^N - 1}
  3:
  4: Define Objective Function Objective_QAOA(params = [gamma, beta]):
  5:    |psi⟩ ← |psi_0⟩
  6:    For layer l from 1 to p do:
  7:        // Apply Problem Unitary U_C(gamma_l) = exp(-i * gamma_l * H_C)
  8:        For basis state k from 0 to (2^N - 1) do:
  9:            |psi[k]⟩ ← exp(-i * gamma[l] * E_k) * |psi[k]⟩
 10:        End For
 11:
 12:        // Apply Mixer Unitary U_M(beta_l) = prod_{j=1}^N Rx(2 * beta_l)
 13:        For qubit j from 1 to N do:
 14:            Apply single-qubit rotation Rx(2 * beta[l]) across all 2^{N-1} entangled pairs
 15:        End For
 16:    End For
 17:
 18:    // Compute Expectation Value
 19:    probs ← |psi|^2
 20:    E_exp ← sum_{k=0}^{2^N - 1} probs[k] * E_k
 21:    Return real(E_exp)
 22:
 23: // Run classical parameter optimization
 24: params_init ← UniformRandom([0, 2*pi]^p x [0, pi]^p)
 25: (params_opt, E_opt) ← ClassicalOptimizer(Objective_QAOA, params_init, method="COBYLA")
 26:
 27: // Evaluate final state and sample maximum likelihood bitstring
 28: |psi_opt⟩ ← EvolveState(params_opt)
 29: probs_opt ← |psi_opt|^2
 30: k_star ← argmax_{k} probs_opt[k]
 31: x_QAOA ← BinaryBitstring(k_star, length=N)
 32: Return x_QAOA, E_opt, probs_opt
```

* **Time Complexity**: $\mathcal{O}(I_{\text{opt}} \cdot p \cdot (2^N + N \cdot 2^{N-1}))$ where $I_{\text{opt}}$ is classical optimizer iterations.
* **Space Complexity**: $\mathcal{O}(2^N)$ complex statevector storage ($\mathbb{C}^{2^N}$).
* **HPC Parallel Mapping**: Statevector unitary gates mapped to cuQuantum GPU kernels or AVX-512 vector chunks ($2^N$ loop split across OpenMP threads).

---

### Algorithm 6: Tail-Risk Conditional Value-at-Risk QAOA (CVaR-QAOA)
**Goal**: Overcome standard expectation limitations by restricting the classical parameter optimization to the best $\alpha$-quantile of the sampled Hamiltonian spectrum (Barkoutsos et al., 2020), actively filtering high-risk tails under budget and demographic constraints.

```text
Input:
  - Cost Hamiltonian H_C, circuit depth p
  - CVaR tail quantile confidence parameter alpha in (0.0, 1.0] (e.g., alpha = 0.25)
  - Classical optimizer: COBYLA (max_iter, tolerance)

Output:
  - Risk-mitigated parameter vectors (gamma_cvar, beta_cvar)
  - Optimal allocation bitstring x_CVaR in {0, 1}^N
  - Tail-expected energy CVaR_alpha

Procedure:
  1: Precompute and sort diagonal energies:
  2:    E_diag ← [ ⟨k| H_C |k⟩ for k in 0 to 2^N - 1 ]
  3:    sort_indices ← ArgsortAscending(E_diag)
  4:    E_sorted ← E_diag[sort_indices]
  5:
  6: Define Objective Function Objective_CVaR(params = [gamma, beta]):
  7:    |psi⟩ ← EvolveStatevector(params, H_C, p)
  8:    probs ← |psi|^2
  9:    probs_sorted ← probs[sort_indices]
 10:
 11:    // Cumulative distribution function over ordered energy spectrum
 12:    cum_probs ← CumulativeSum(probs_sorted)
 13:    k_alpha ← SearchSorted(cum_probs, alpha)  // Find cutoff index where sum(P) >= alpha
 14:
 15:    // Compute conditional tail expectation
 16:    p_head ← probs_sorted[0 : k_alpha]
 17:    e_head ← E_sorted[0 : k_alpha]
 18:    p_sum ← sum(p_head)
 19:    remainder_p ← max(0.0, alpha - p_sum)
 20:
 21:    CVaR_alpha ← (1.0 / alpha) * ( sum(p_head * e_head) + remainder_p * E_sorted[k_alpha] )
 22:    Return real(CVaR_alpha)
 23:
 24: (params_cvar, E_cvar) ← ClassicalOptimizer(Objective_CVaR, params_init, method="COBYLA")
 25: |psi_cvar⟩ ← EvolveStatevector(params_cvar, H_C, p)
 26: k_star ← argmax_{k} |psi_cvar[k]|^2
 27: x_CVaR ← BinaryBitstring(k_star, length=N)
 28: Return x_CVaR, E_cvar
```

* **Theoretical Property**: $\lim_{\alpha \to 1.0} \text{CVaR}_\alpha = \langle H_C \rangle$ (standard QAOA), and $\lim_{\alpha \to 0^+} \text{CVaR}_\alpha = \min E_k$ (exact ground state).
* **Time Complexity**: $\mathcal{O}(2^N \log(2^N))$ pre-sorting + $\mathcal{O}(I_{\text{opt}} \cdot (p \cdot 2^N + \log(2^N)))$.
* **Space Complexity**: $\mathcal{O}(2^N)$ statevector and index arrays.
* **HPC Parallel Mapping**: Parallel prefix sum (`scan`) and vectorized partial dot-product for the $\alpha$-quantile reduction.

---

### Algorithm 7: Greedy Ratio Knapsack Baseline (Fairness-Blind)
**Goal**: Greedily select microfinance applicants in descending order of benefit-to-cost ratio $C_i / w_i$ until the budget is exhausted, exposing the social vulnerability and demographic bias of naive heuristics.

```text
Input:
  - Composite utility vector C in R^N, loan amounts w in R^N, budget ceiling B in R^+

Output:
  - Allocation bitstring x_greedy in {0, 1}^N
  - Consumed budget cost_greedy in R^+

Procedure:
  1: For i from 1 to N do:
  2:    ratio[i] ← C[i] / max(1.0, w[i])
  3: End For
  4:
  5: sorted_indices ← ArgsortDescending(ratio)
  6: x_greedy ← zeros(N)
  7: current_spend ← 0.0
  8:
  9: For each index i in sorted_indices do:
 10:    If (current_spend + w[i] <= B) then:
 11:        x_greedy[i] ← 1
 12:        current_spend ← current_spend + w[i]
 13:    End If
 14: End For
 15: Return x_greedy, current_spend
```

* **Time Complexity**: $\mathcal{O}(N \log N)$ due to sorting.
* **Space Complexity**: $\mathcal{O}(N)$ indices and allocation array.
* **HPC Parallel Mapping**: Parallel radix sort / parallel QuickSort (`std::sort` / TBB).

---

### Algorithm 8: High-Performance Exact Mixed-Integer Linear Programming (MILP)
**Goal**: Solve the constrained microfinance allocation to provable mathematical global optimality using Simplex Branch-and-Cut with exact linear budget and demographic parity bounds.

$$\max_{x \in \{0, 1\}^N} \sum_{i=1}^N C_i x_i \quad \text{s.t.} \quad \sum_{i=1}^N w_i x_i \le B \quad \text{and} \quad -\epsilon \le \frac{1}{|G_A|}\sum_{i \in G_A} x_i - \frac{1}{|G_B|}\sum_{j \in G_B} x_j \le \epsilon$$

```text
Input:
  - Utility vector C in R^N, loan amounts w in R^N, group indicators G in {0, 1}^N
  - Budget ceiling B, fairness tolerance epsilon in [0.0, 1.0]

Output:
  - Globally optimal allocation x_MILP in {0, 1}^N
  - Maximum linear objective utility U_optimal

Procedure:
  1: Formulate cost vector: c ← - C  // Minimize -c^T x
  2: Initialize Constraint Matrix A in R^{2 x N}:
  3:    // Row 1: Budget Constraint
  4:    A[1, :] ← [ w[1], w[2], ..., w[N] ]
  5:    lhs[1] ← 0.0;  rhs[1] ← B
  6:
  7:    // Row 2: Demographic Parity Difference Constraint
  8:    For i from 1 to N do:
  9:        A[2, i] ← (+ 1.0 / N_A) if G[i] == 0 else (- 1.0 / N_B)
 10:    End For
 11:    lhs[2] ← - epsilon;  rhs[2] ← + epsilon
 12:
 13: Set bounds: 0.0 <= x_i <= 1.0 for all i in 1 to N
 14: Set integrality flags: integrality[i] ← 1 (Integer / Binary constraint) for all i
 15:
 16: // HPC Branch-and-Cut Simplex execution
 17: solver_result ← SciPy_MILP(c=c, A=A, lhs=lhs, rhs=rhs, bounds=(0, 1), integrality=integrality)
 18:
 19: If solver_result.success == True then:
 20:    x_MILP ← Round(solver_result.x)
 21:    U_optimal ← sum(C * x_MILP)
 22: Else:
 23:    x_MILP ← FallbackHeuristic()
 24: End If
 25: Return x_MILP, U_optimal
```

* **Time Complexity**: Worst-case $\mathcal{O}(2^N)$ exponential, with average-case $\mathcal{O}(\text{poly}(N))$ pruned via branch-and-cut linear programming relaxations.
* **Space Complexity**: $\mathcal{O}(N \cdot K_{\text{nodes}})$ branch-and-bound search tree stack.
* **HPC Parallel Mapping**: Parallel tree exploration across CPU worker threads with lockless work-stealing queues.

---

### Algorithm 9: Classical Simulated Annealing (MCMC Metropolis-Hastings)
**Goal**: Stochastic exploration of the non-convex QUBO energy landscape via single-spin flips and Boltzmann thermal acceptance probability.

```text
Input:
  - QUBO Matrix Q in R^{N x N}, scalar offset E_offset
  - Thermal schedule: T_initial=10.0, T_min=0.01, cooling_rate=0.985, max_steps=1500

Output:
  - Optimized binary allocation x_SA in {0, 1}^N
  - Minimum encountered energy E_best

Procedure:
  1: x_current ← RandomChoice({0, 1}, size=N)
  2: E_current ← x_current^T * Q * x_current + E_offset
  3: x_best ← x_current;  E_best ← E_current
  4: T ← T_initial
  5:
  6: For step from 1 to max_steps do:
  7:    If T <= T_min then break
  8:
  9:    // Propose bit-flip neighbor at random site k
 10:    k ← UniformRandomInteger(1, N)
 11:    x_candidate ← Copy(x_current)
 12:    x_candidate[k] ← 1 - x_candidate[k]
 13:
 14:    // Fast Delta-E calculation: Delta_E = (1 - 2*x_k) * (Q_{kk} + sum_{j != k} Q_{kj} x_j)
 15:    E_candidate ← x_candidate^T * Q * x_candidate + E_offset
 16:    delta_E ← E_candidate - E_current
 17:
 18:    // Metropolis acceptance criterion
 19:    If (delta_E < 0) or (UniformRandom(0, 1) < exp(-delta_E / T)) then:
 20:        x_current ← x_candidate
 21:        E_current ← E_candidate
 22:        If E_current < E_best then:
 23:            x_best ← x_current
 24:            E_best ← E_current
 25:        End If
 26:    End If
 27:
 28:    T ← T * cooling_rate
 29: End For
 30: Return x_best, E_best
```

* **Time Complexity**: $\mathcal{O}(\text{max\_steps} \cdot N)$ using $\mathcal{O}(N)$ local energy update.
* **Space Complexity**: $\mathcal{O}(N)$ contiguous memory for state vectors.
* **HPC Parallel Mapping**: Multi-walker parallel tempering with periodic replica exchange across MPI nodes.

---

### Algorithm 10: Variational Quantum Eigensolver (VQE) with Hardware-Efficient Ansatz
**Goal**: Minimize ground-state energy using an unconstrained TwoLocal parameterized ansatz ($R_y$ rotations + circular CNOT entanglement) on the Hamiltonian $H_C$.

```text
Input:
  - Cost Hamiltonian H_C in R^{2^N x 2^N}
  - Number of ansatz entanglement layers L in N^+ (e.g., L=1 or 2)
  - Optimizer: COBYLA (max_iter, tolerance)

Output:
  - Optimal parameter vector theta_opt in R^{(L + 1) * N}
  - Ground-state allocation bitstring x_VQE in {0, 1}^N

Procedure:
  1: Define Parameterized Ansatz Circuit U(theta):
  2:    Initialize |psi⟩ ← |0⟩^{⊗N}
  3:    For l from 1 to L do:
  4:        Apply single-qubit rotations Ry(theta_{(l-1)*N + j}) to all qubits j in 1..N
  5:        Apply circular entangling CNOT gates: CNOT(j, (j mod N) + 1) for j in 1..N
  6:    End For
  7:    Apply final layer: Ry(theta_{L*N + j}) to each qubit j in 1..N
  8:    Return |psi⟩
  9:
 10: Define Objective Function Objective_VQE(theta):
 11:    |psi_theta⟩ ← U(theta)
 12:    E_exp ← ⟨psi_theta| H_C |psi_theta⟩
 13:    Return real(E_exp)
 14:
 15: theta_init ← UniformRandom([-0.05, 0.05]^{(L+1)*N})
 16: (theta_opt, E_min) ← ClassicalOptimizer(Objective_VQE, theta_init, method="COBYLA")
 17:
 18: |psi_final⟩ ← U(theta_opt)
 19: k_star ← argmax_{k} |psi_final[k]|^2
 20: x_VQE ← BinaryBitstring(k_star, length=N)
 21: Return x_VQE, E_min
```

* **Time Complexity**: $\mathcal{O}(I_{\text{opt}} \cdot ((L+1) \cdot N \cdot 2^{N-1} + 2^N))$.
* **Space Complexity**: $\mathcal{O}(2^N)$ statevector allocation.
* **HPC Parallel Mapping**: Threaded single-qubit gate application and parallel cyclic bit-shift operations for circular CNOTs.

---

### Algorithm 11: Monte Carlo Stochastic Default Loss Simulation & Tail Risk (VaR / CVaR)
**Goal**: Simulate discrete portfolio default outcomes under repayment uncertainty $q_i = 1 - p_i$ and compute discrete Value-at-Risk ($\text{VaR}_\alpha$) and Conditional Value-at-Risk ($\text{CVaR}_\alpha$) for any allocation $x$.

```text
Input:
  - Allocation vector x in {0, 1}^N
  - Requested loan amounts w in R^N
  - Calibrated default probabilities q in [0, 1]^N (from Algorithm 1)
  - Number of Monte Carlo scenarios S in N^+ (e.g., S = 2,500)
  - Tail risk confidence parameter alpha in (0, 1) (e.g., alpha = 0.95)

Output:
  - Expected Portfolio Loss E_loss in R^+
  - Portfolio Value-at-Risk VaR_alpha in R^+
  - Portfolio Conditional Value-at-Risk CVaR_alpha in R^+

Procedure:
  1: Pre-generate Scenario Default Matrix D in {0, 1}^{S x N}:
  2:    For s from 1 to S do:
  3:        For i from 1 to N do:
  4:            D[s, i] ← 1 if UniformRandom(0, 1) < q[i] else 0
  5:        End For
  6:    End For
  7:
  8: // Compute portfolio loss per scenario: L_s = sum_{i=1}^N w_i * x_i * D_{s, i}
  9: loss_per_applicant ← w * x  // Vector of length N
 10: scenario_losses ← D @ loss_per_applicant  // Matrix-vector product of size S
 11:
 12: // Sort losses in non-decreasing order
 13: sorted_losses ← SortAscending(scenario_losses)
 14:
 15: // 1. Expected Loss (Mean)
 16: E_loss ← (1.0 / S) * sum(sorted_losses)
 17:
 18: // 2. Value-at-Risk (VaR_alpha) at alpha-quantile
 19: idx_var ← min(S - 1, floor(alpha * S))
 20: VaR_alpha ← sorted_losses[idx_var]
 21:
 22: // 3. Conditional Value-at-Risk (CVaR_alpha): Mean of worst (1 - alpha) tail losses
 23: tail_losses ← sorted_losses[idx_var : S]
 24: CVaR_alpha ← (1.0 / length(tail_losses)) * sum(tail_losses)
 25:
 26: Return E_loss, VaR_alpha, CVaR_alpha
```

* **Time Complexity**: $\mathcal{O}(S \cdot N + S \log S)$ for scenario matrix multiplication and sort.
* **Space Complexity**: $\mathcal{O}(S \cdot N)$ binary scenario cache.
* **HPC Parallel Mapping**: BLAS-2 General Matrix-Vector multiplication (`dgemv`) accelerated on multi-core CPU / GPU.

---

### Algorithm 12: Algorithmic Fairness Auditing & Approximation Metric Suite
**Goal**: Compute demographic fairness metrics (Demographic Parity Difference, Disparate Impact Ratio) and approximation ratios with division-by-zero protection.

```text
Input:
  - Evaluated allocation x in {0, 1}^N
  - Applicant attributes: C, w, G, budget B
  - Optimal benchmark utility U_optimal (from Algorithm 8)

Output:
  - Evaluation record M with DPD, DIR, Budget Utilization, and Approximation Ratio

Procedure:
  1: total_cost ← sum(w[i] * x[i] for i from 1 to N)
  2: is_feasible ← (total_cost <= B + 1e-5)
  3: budget_utilization ← (total_cost / B) * 100.0
  4: total_utility ← sum(C[i] * x[i] for i from 1 to N)
  5:
  6: // Group selection rates
  7: mask_A ← (G == 0);  mask_B ← (G == 1)
  8: n_A ← max(1, count(mask_A));  n_B ← max(1, count(mask_B))
  9: approved_A ← sum(x[i] for i in 1..N if mask_A[i])
 10: approved_B ← sum(x[i] for i in 1..N if mask_B[i])
 11: rate_A ← approved_A / n_A
 12: rate_B ← approved_B / n_B
 13:
 14: // Demographic Parity Difference (DPD): |rate_A - rate_B|
 15: DPD ← abs(rate_A - rate_B)
 16:
 17: // Disparate Impact Ratio (DIR / 80% Rule): rate_A / rate_B
 18: If rate_B > 0 then:
 19:    DIR ← rate_A / rate_B
 20: Else:
 21:    DIR ← 1.0 if rate_A == 0 else +infinity
 22: End If
 23:
 24: // Approximation Ratio alpha_approx = Total Utility / Optimal Utility
 25: approx_ratio ← (total_utility / U_optimal) if U_optimal > 0 else 1.0
 26: Return { total_utility, total_cost, budget_utilization, is_feasible, DPD, DIR, approx_ratio }
```

* **Time Complexity**: $\mathcal{O}(N)$ vectorized array reductions.
* **Space Complexity**: $\mathcal{O}(1)$ auxiliary memory.
* **HPC Parallel Mapping**: Vectorized reduction primitives with SIMD lane-level summing.

---

### Computational Complexity & HPC Execution Benchmark Matrix

| Algorithm | Method Category | Asymptotic Time Complexity | Memory / Space Footprint | Concurrency & HPC Parallelism Model |
|---|---|:---:|:---:|---|
| **Alg 1: AI Repayment (CV)** | Supervised ML & Calibration | $\mathcal{O}(K \cdot B \cdot M \log M)$ | $\mathcal{O}(M \cdot D)$ | OpenMP multi-tree bagging, parallel CV folds |
| **Alg 2: Welfare Tri-Scoring** | Multi-Objective Modeling | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ | Vectorized SIMD loop (AVX2/AVX-512) |
| **Alg 3: QUBO Matrix Assembly** | Penalty Quadratic Embedding | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ | Parallel upper-triangular loop distribution |
| **Alg 4: Ising Glass Mapping** | Pauli-Z Spin Decomposition | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ | Row-major memory BLAS-2 linear algebra |
| **Alg 5: Standard QAOA** | Variational Quantum Circuit | $\mathcal{O}(I_{\text{opt}} \cdot p \cdot 2^N)$ | $\mathcal{O}(2^N)$ | Parallel quantum statevector contraction / GPU cuQuantum |
| **Alg 6: CVaR-QAOA** | Tail-Risk Quantum Optimizer | $\mathcal{O}(I_{\text{opt}} \cdot p \cdot 2^N + 2^N \log 2^N)$ | $\mathcal{O}(2^N)$ | Parallel scan / prefix sum reduction on $\alpha$-quantile |
| **Alg 7: Greedy Knapsack** | Classical Greedy Heuristic | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N)$ | Parallel Radix / QuickSort |
| **Alg 8: Exact Classical MILP** | Simplex Branch-and-Cut | Exponential $\mathcal{O}(2^N)$, Avg $\mathcal{O}(\text{poly})$ | $\mathcal{O}(N \cdot K_{\text{nodes}})$ | Multithreaded branch-and-bound search tree |
| **Alg 9: Simulated Annealing** | Classical MCMC Metaheuristic | $\mathcal{O}(\text{steps} \cdot N)$ | $\mathcal{O}(N)$ | Parallel tempering / replica exchange across cores |
| **Alg 10: Quantum VQE** | Hardware-Efficient Ansatz | $\mathcal{O}(I_{\text{opt}} \cdot L \cdot N \cdot 2^N)$ | $\mathcal{O}(2^N)$ | Threaded multi-qubit tensor product rotations |
| **Alg 11: Monte Carlo Tail Risk**| Stochastic Risk Simulation | $\mathcal{O}(S \cdot N + S \log S)$ | $\mathcal{O}(S \cdot N)$ | BLAS `dgemv` matrix-vector scenario multiplication |
| **Alg 12: Fairness Auditing** | Demographic Parity KPIs | $\mathcal{O}(N)$ | $\mathcal{O}(1)$ | SIMD parallel register reduction |
