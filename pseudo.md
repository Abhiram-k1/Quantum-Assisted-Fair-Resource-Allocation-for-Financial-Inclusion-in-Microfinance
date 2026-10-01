# Algorithmic Pseudocode Specification (pseudo.md)
## Quantum-Assisted Fair Resource Allocation for Financial Inclusion in Microfinance (Q-FAR)

This document provides formal, publication-grade pseudocode for all core components of the Q-FAR pipeline, spanning score modeling, QUBO/Ising matrix generation, classical baseline algorithms, quantum variational optimization (QAOA & VQE), and evaluation metrics.

---

### Algorithm 1: Tri-Objective Feature Scoring & Normalization
**Goal**: Compute normalized composite scores combining Financial Viability ($F_i$), Urgent Need ($N_i$), and Social Impact Multiplier ($S_i$) for each applicant $i \in \{1, \dots, N\}$.

```text
Input:
  - Raw applicant dataset D = { (income_i, dti_i, repayment_history_i, poverty_idx_i, dependents_i, women_led_i, group_i, requested_amount_w_i) } for i = 1 to N
  - Weights vector w_obj = (omega_F, omega_N, omega_S) such that omega_F + omega_N + omega_S = 1.0

Output:
  - Composite utility array C of size N, where C_i in [0, 1]
  - Group membership vector G where G_i in {0, 1} (0: Group A / Marginalized, 1: Group B / General)
  - Loan request amounts vector w of size N

Procedure:
  1: For each applicant i from 1 to N do:
  2:    // Financial Viability Score F_i
  3:    F_i ← 0.40 * (1.0 - min(1.0, dti_i)) + 0.35 * repayment_history_i + 0.25 * min(1.0, income_i / income_median)
  4:
  5:    // Urgent Need Index N_i
  6:    N_i ← 0.50 * poverty_idx_i + 0.30 * min(1.0, dependents_i / 6.0) + 0.20 * (1.0 - min(1.0, income_i / income_median))
  7:
  8:    // Social Impact Multiplier S_i
  9:    S_i ← 0.45 * (1.0 if women_led_i == True else 0.2) + 0.35 * (1.0 if group_i == "Marginalized" else 0.3) + 0.20 * min(1.0, requested_amount_w_i / 5000.0)
 10:
 11:    // Clamp individual scores to [0, 1]
 12:    F_i ← clamp(F_i, 0.0, 1.0)
 13:    N_i ← clamp(N_i, 0.0, 1.0)
 14:    S_i ← clamp(S_i, 0.0, 1.0)
 15:
 16:    // Composite Multi-Objective Utility
 17:    C_i ← omega_F * F_i + omega_N * N_i + omega_S * S_i
 18: End For
 19: Return C, G, w
```

---

### Algorithm 2: Multi-Objective Constrained Problem to QUBO Matrix Mapping
**Goal**: Assemble the $N \times N$ upper-triangular QUBO matrix $Q$ encoding the objective function, quadratic budget constraint penalty, and group demographic fairness penalty.

```text
Input:
  - Composite benefit scores C in R^N
  - Loan request amounts w in R^N
  - Group indicators G in {0, 1}^N (Group A: G_i == 0, Group B: G_i == 1)
  - Available total budget B
  - Penalty weights: lambda_B (budget constraint), lambda_F (fairness penalty)

Output:
  - QUBO Matrix Q in R^{N x N}
  - Constant scalar offset E_offset

Procedure:
  1: N ← length(C)
  2: Q ← zeros(N, N)
  3:
  4: N_A ← count(i for which G_i == 0)
  5: N_B ← count(i for which G_i == 1)
  6:
  7: // Calculate demographic fairness weight vectors: sigma_i
  8: For i from 1 to N do:
  9:    If G_i == 0 then:
 10:        sigma_i ← + 1.0 / N_A
 11:    Else:
 12:        sigma_i ← - 1.0 / N_B
 13:    End If
 14: End For
 15:
 16: // Populate Diagonal Elements (Linear terms + Self-interaction)
 17: For i from 1 to N do:
 18:    // Objective: minimize -C_i * x_i
 19:    term_obj ← - C_i
 20:    // Budget penalty linear expansion: lambda_B * (w_i^2 - 2 * B * w_i) * x_i
 21:    term_budget ← lambda_B * (w[i]^2 - 2.0 * B * w[i])
 22:    // Fairness penalty diagonal: lambda_F * sigma_i^2 * x_i
 23:    term_fair ← lambda_F * (sigma_i^2)
 24:
 25:    Q[i, i] ← term_obj + term_budget + term_fair
 26: End For
 27:
 28: // Populate Off-Diagonal Elements (Pairwise Cross-terms)
 29: For i from 1 to N do:
 30:    For j from (i + 1) to N do:
 31:        // Cross-term from budget constraint: 2 * lambda_B * w_i * w_j
 32:        cross_budget ← 2.0 * lambda_B * w[i] * w[j]
 33:        // Cross-term from demographic parity: 2 * lambda_F * sigma_i * sigma_j
 34:        cross_fair ← 2.0 * lambda_F * sigma_i * sigma_j
 35:
 36:        Q[i, j] ← cross_budget + cross_fair
 37:    End For
 38: End For
 39:
 40: E_offset ← lambda_B * (B^2)
 41: Return Q, E_offset
```

---

### Algorithm 3: QUBO to Ising Spin Hamiltonian Transformation
**Goal**: Convert binary variables $x_i \in \{0, 1\}$ into Pauli-Z spin operators $Z_i \in \{+1, -1\}$ via $x_i = \frac{I - Z_i}{2}$ to generate the Ising cost Hamiltonian $H_C = \sum_{i} h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \text{offset} \cdot I$.

```text
Input:
  - QUBO Matrix Q in R^{N x N}
  - Scalar offset E_offset

Output:
  - Longitudinal field vector h in R^N
  - Interaction coupling matrix J in R^{N x N} (upper triangular)
  - Transformed scalar offset ising_offset

Procedure:
  1: N ← dimension(Q, 1)
  2: h ← zeros(N)
  3: J ← zeros(N, N)
  4: ising_offset ← E_offset
  5:
  6: // Two-qubit exchange coupling J_ij
  7: For i from 1 to N do:
  8:    For j from (i + 1) to N do:
  9:        J[i, j] ← 0.25 * Q[i, j]
 10:    End For
 11: End For
 12:
 13: // Single-qubit longitudinal field h_i
 14: For i from 1 to N do:
 15:    sum_row_col ← sum(Q[i, j] for j > i) + sum(Q[j, i] for j < i)
 16:    h[i] ← - 0.5 * Q[i, i] - 0.25 * sum_row_col
 17: End For
 18:
 19: // Global energy scalar shift
 20: sum_diag ← sum(Q[i, i] for i in 1..N)
 21: sum_upper ← sum(Q[i, j] for i in 1..N, j > i)
 22: ising_offset ← ising_offset + 0.5 * sum_diag + 0.25 * sum_upper
 23:
 24: Return h, J, ising_offset
```

---

### Algorithm 4: Quantum Approximate Optimization Algorithm (QAOA)
**Goal**: Find parameters $(\vec{\gamma}^*, \vec{\beta}^*)$ minimizing the expectation $\langle \psi(\vec{\gamma}, \vec{\beta}) | H_C | \psi(\vec{\gamma}, \vec{\beta}) \rangle$ and sample optimal binary loan allocations.

```text
Input:
  - Cost Hamiltonian parameters (h, J, ising_offset)
  - Circuit depth p (number of layers)
  - Optimizer (e.g., COBYLA, max_iterations=150)
  - Number of measurement shots M (e.g., 2048)

Output:
  - Optimal loan allocation bitstring x_QAOA in {0, 1}^N
  - Optimal expectation energy E_min
  - Probability distribution P(x) over basis states

Procedure:
  1: Define Initial State:
  2:    |psi_0⟩ ← (|0⟩ + |1⟩) / sqrt(2) for each of N qubits (uniform superposition)
  3:
  4: Define Cost Unitary U_C(gamma):
  5:    U_C(gamma) = exp(-i * gamma * H_C)
  6:               = prod_{i=1}^N Rz(2 * gamma * h_i) * prod_{i < j} Rzz(2 * gamma * J_{ij})
  7:
  8: Define Mixer Unitary U_M(beta):
  9:    U_M(beta) = exp(-i * beta * sum_{i=1}^N X_i)
 10:              = prod_{i=1}^N Rx(2 * beta)
 11:
 12: Define State Evolution Function Statevector(gamma_vec, beta_vec):
 13:    |psi⟩ ← |psi_0⟩
 14:    For k from 1 to p do:
 15:        |psi⟩ ← U_C(gamma_vec[k]) * |psi⟩
 16:        |psi⟩ ← U_M(beta_vec[k]) * |psi⟩
 17:    End For
 18:    Return |psi⟩
 19:
 20: Define Loss Function Loss(params):
 21:    gamma_vec ← params[1 : p]
 22:    beta_vec ← params[p + 1 : 2p]
 23:    |psi⟩ ← Statevector(gamma_vec, beta_vec)
 24:    energy ← ⟨psi| H_C |psi⟩
 25:    Return real(energy)
 26:
 27: // Initialize variational angles
 28: params_init ← random_uniform([0, 2*pi]^p x [0, pi]^p)
 29:
 30: // Classical optimization loop
 31: (params_opt, E_min) ← ClassicalOptimizer(Loss, params_init, method="COBYLA")
 32:
 33: // Final State & Sampling
 34: |psi_opt⟩ ← Statevector(params_opt[1:p], params_opt[p+1:2p])
 35: For each basis state |x⟩ in {0, 1}^N do:
 36:    P(x) ← |⟨x | psi_opt⟩|^2
 37: End For
 38:
 39: x_QAOA ← argmax_{x} P(x)
 40: Return x_QAOA, E_min, P
```

---

### Algorithm 5: Variational Quantum Eigensolver (VQE) with Hardware-Efficient Ansatz
**Goal**: Optimize an expressive parameterized quantum circuit $U(\vec{\theta})$ to prepare the ground state of the resource allocation Hamiltonian.

```text
Input:
  - Cost Hamiltonian H_C
  - Number of ansatz repetitions L
  - Classical optimizer (e.g., COBYLA, SLSQP)

Output:
  - Optimal loan allocation bitstring x_VQE
  - Ground state energy estimate E_VQE

Procedure:
  1: Define Ansatz Circuit U(theta):
  2:    Start with |0⟩^{\otimes N}
  3:    For layer l from 1 to L do:
  4:        Apply single-qubit rotations Ry(theta_{i, l}) to all qubits i = 1 to N
  5:        Apply entangling circular or linear CNOT / CZ gates across neighboring qubits (i, i+1)
  6:    End For
  7:    Apply final Ry(theta_{i, L+1}) rotation to each qubit
  8:
  9: Define Energy Expectation:
 10:    E(theta) = ⟨0| U†(theta) H_C U(theta) |0⟩
 11:
 12: theta_init ← random_normal(mean=0, std=0.01, size=N*(L+1))
 13: (theta_opt, E_VQE) ← ClassicalOptimizer(E(theta), theta_init, method="COBYLA")
 14:
 15: |psi_VQE⟩ ← U(theta_opt) |0⟩^{\otimes N}
 16: x_VQE ← argmax_{x} |⟨x | psi_VQE⟩|^2
 17: Return x_VQE, E_VQE
```

---

### Algorithm 6: Classical Baseline Solvers
**Goal**: Establish rigorous classical comparison baselines: Greedy Ratio Knapsack, Exact Combinatorial Optimization (Branch & Bound), and Simulated Annealing.

```text
// Part A: Greedy Ratio Knapsack Heuristic
Input: C in R^N, w in R^N, Budget B
Output: x_greedy in {0, 1}^N
Procedure:
  1: Compute ratios: r_i ← C_i / w_i for each applicant i
  2: sorted_indices ← sort indices i in descending order of r_i
  3: x_greedy ← zeros(N)
  4: remaining_budget ← B
  5: For each index i in sorted_indices do:
  6:    If w[i] <= remaining_budget then:
  7:        x_greedy[i] ← 1
  8:        remaining_budget ← remaining_budget - w[i]
  9:    End If
 10: End For
 11: Return x_greedy

// Part B: Exact Classical Combinatorial Solver
Input: Q in R^{N x N}, E_offset
Output: x_exact in {0, 1}^N, E_optimal
Procedure:
  1: E_optimal ← +infinity
  2: x_exact ← None
  3: For each candidate bitstring x in {0, 1}^N do:
  4:    E_x ← x^T * Q * x + E_offset
  5:    If E_x < E_optimal then:
  6:        E_optimal ← E_x
  7:        x_exact ← x
  8:    End If
  9: End For
 10: Return x_exact, E_optimal

// Part C: Classical Simulated Annealing (Metropolis-Hastings)
Input: Q in R^{N x N}, E_offset, T_start=10.0, T_min=0.01, cooling_rate=0.98, max_iter=2000
Output: x_SA in {0, 1}^N, E_SA
Procedure:
  1: x_current ← random_sample({0, 1}^N)
  2: E_current ← x_current^T * Q * x_current + E_offset
  3: x_best ← x_current; E_best ← E_current
  4: T ← T_start
  5: While T > T_min and iter < max_iter do:
  6:    // Propose bit-flip neighbor
  7:    k ← random_integer(1, N)
  8:    x_neighbor ← copy(x_current)
  9:    x_neighbor[k] ← 1 - x_neighbor[k]
 10:    E_neighbor ← x_neighbor^T * Q * x_neighbor + E_offset
 11:    delta_E ← E_neighbor - E_current
 12:
 13:    If delta_E < 0 or random_uniform(0, 1) < exp(-delta_E / T) then:
 14:        x_current ← x_neighbor
 15:        E_current ← E_neighbor
 16:        If E_current < E_best then:
 17:            x_best ← x_current; E_best ← E_current
 18:        End If
 19:    End If
 20:    T ← T * cooling_rate
 21: End While
 22: Return x_best, E_best
```

---

### Algorithm 7: Multi-Metric Benchmark & Disparate Impact Evaluation
**Goal**: Compute key performance indicators (KPIs) to compare quantum and classical allocation solutions.

```text
Input:
  - Allocation vector x in {0, 1}^N
  - Applicant attributes: C, w, G, B
  - Optimal classical energy E_optimal

Output:
  - Metrics record M = { TotalUtility, TotalCost, BudgetUtilization, DPD, DIR, ApproximationRatio, Feasible }

Procedure:
  1: TotalUtility ← sum(C_i * x_i for i in 1..N)
  2: TotalCost ← sum(w_i * x_i for i in 1..N)
  3: BudgetUtilization ← TotalCost / B
  4: Feasible ← (TotalCost <= B)
  5:
  6: // Group Approval Rates
  7: N_A ← count(i for which G_i == 0)
  8: N_B ← count(i for which G_i == 1)
  9: approved_A ← sum(x_i for i in 1..N if G_i == 0)
 10: approved_B ← sum(x_i for i in 1..N if G_i == 1)
 11: rate_A ← approved_A / max(1, N_A)
 12: rate_B ← approved_B / max(1, N_B)
 13:
 14: // Demographic Parity Difference
 15: DPD ← abs(rate_A - rate_B)
 16:
 17: // Disparate Impact Ratio
 18: If rate_B > 0 then:
 19:    DIR ← rate_A / rate_B
 20: Else:
 21:    DIR ← 1.0 if rate_A == 0 else +infinity
 22: End If
 23:
 24: // Energy Evaluation
 25: E_sol ← x^T * Q * x + E_offset
 26: ApproximationRatio ← abs(E_optimal / E_sol) if E_sol != 0 else 0.0
 27:
 28: Return M
```
