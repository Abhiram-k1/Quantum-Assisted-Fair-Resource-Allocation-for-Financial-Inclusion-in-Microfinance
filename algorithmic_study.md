# Algorithmic Study & Literature Review
## Quantum-Assisted Fair Resource Allocation for Financial Inclusion in Microfinance (Q-FAR)

---

### Executive Summary
Microfinance institutions (MFIs) operate at the frontier of financial inclusion, extending micro-credit to underserved, low-income, and marginalized communities who lack collateral and formal credit histories. However, MFIs face persistent capital rationing: loan demand frequently outstrips available lendable capital. Balancing financial sustainability (default risk mitigation, capital recovery) with socio-economic impact (poverty alleviation, demographic equity) under strict budget ceilings forms a high-dimensional, constrained, multi-objective combinatorial optimization problem.

This study establishes the algorithmic foundation of **Q-FAR**:
1. Formulating the Fair Microfinance Resource Allocation Problem as a **Multi-Objective Quadratic Knapsack Problem (MOQKP)**.
2. Embedding multi-group demographic parity constraints as quadratic penalty terms into a **Quadratic Unconstrained Binary Optimization (QUBO)** model.
3. Mapping the QUBO onto an **Ising Spin Hamiltonian** ($H_C$).
4. Applying and benchmarking two leading Noisy Intermediate-Scale Quantum (NISQ) variational algorithms:
   - **Quantum Approximate Optimization Algorithm (QAOA)**
   - **Variational Quantum Eigensolver (VQE)**
5. Benchmarking quantum solutions against classical exact solvers (Integer Linear Programming / Branch & Bound) and heuristics (Greedy Ratio Knapsack, Simulated Annealing).

---

## 1. Problem Definition & Societal Motivation

### 1.1 The Microfinance Capital Allocation Dilemma
In traditional banking, credit allocation relies heavily on centralized credit scores (e.g., FICO, CIBIL), tax filings, and collateral assets. In microfinance settings (e.g., Grameen Bank models, rural joint-liability groups, self-help groups), applicants are unbanked or underbanked micro-entrepreneurs.

When an MFI receives $N$ loan applications totaling an aggregate requested capital $\sum_{i=1}^N w_i$ that exceeds the total lending capital budget $B$ ($\sum w_i > B$), the institution must select a subset of applicants $x \in \{0, 1\}^N$, where:
$$x_i = \begin{cases} 1 & \text{if applicant } i \text{ is approved for loan of amount } w_i \\ 0 & \text{otherwise} \end{cases}$$

### 1.2 The Phenomenon of "Mission Drift"
Commercialized microfinance institutions often suffer from **mission drift**: to maintain solvency and maximize returns, scoring algorithms disproportionately favor lower-risk, wealthier borrowers or urban applicants with semi-formal cash flows. As a consequence:
- Vulnerable rural populations and women-led cooperative enterprises are systematically rejected.
- Algorithmic credit scoring exacerbates existing socio-economic biases.
- Purely profit-driven optimization yields high financial recovery but fails the core mandate of social development and financial inclusion.

### 1.3 Tri-Objective Value Modeling
To mitigate mission drift while ensuring long-term MFI sustainability, each applicant $i$ is characterized by a multi-dimensional feature vector $\mathbf{v}_i$:
- **Financial Viability Score ($F_i \in [0, 1]$)**: Modeled from debt-to-income ratio, business cash-flow margin, and past peer-group repayment adherence.
- **Urgent Need Index ($N_i \in [0, 1]$)**: Quantifying poverty headcount ratio, family dependency ratio, household asset deficit, and vulnerability to economic shocks.
- **Social Impact Multiplier ($S_i \in [0, 1]$)**: Measuring community spillover effects, job generation potential, women empowerment index, and green/sustainable agriculture alignment.

The composite benefit coefficient $C_i$ is expressed as a convex combination:
$$C_i = \omega_F \cdot F_i + \omega_N \cdot N_i + \omega_S \cdot S_i, \quad \text{where } \omega_F + \omega_N + \omega_S = 1, \quad \omega_F, \omega_N, \omega_S \ge 0$$

---

## 2. Computational Complexity & The Need for Quantum Approaches

### 2.1 From 0-1 Knapsack to Quadratic Knapsack (QKP)
The standard 0-1 Knapsack Problem is defined as:
$$\max_{x \in \{0, 1\}^N} \sum_{i=1}^N C_i x_i \quad \text{subject to} \quad \sum_{i=1}^N w_i x_i \le B$$
While 0-1 Knapsack is weakly NP-complete and admits pseudo-polynomial dynamic programming solutions $\mathcal{O}(N \cdot B)$, adding **fairness balance terms** across demographic groups introduces quadratic cross-terms $x_i x_j$.

Let applicants belong to demographic groups $G_A$ (e.g., historically marginalized / rural women) and $G_B$ (e.g., general / urban micro-entrepreneurs). Enforcing demographic parity requires equal selection rates:
$$\frac{1}{|G_A|} \sum_{i \in G_A} x_i \approx \frac{1}{|G_B|} \sum_{j \in G_B} x_j$$
Squaring this difference to penalize disparities produces an objective of the form:
$$\min_{x} \left( \frac{1}{|G_A|} \sum_{i \in G_A} x_i - \frac{1}{|G_B|} \sum_{j \in G_B} x_j \right)^2$$
Expanding this expression yields pairwise interaction terms $x_i x_j$ with both positive and negative coefficients. This transforms the problem into a **Multiobjective Quadratic Knapsack Problem (QKP)**, which is strongly NP-hard.

### 2.2 Classical Algorithmic Limitations
1. **Integer Linear Programming (ILP) with Slack Variables**: Branch-and-bound and cutting-plane methods require linearization of quadratic terms (using Glover or McCormick relaxations), which inflates the variable space from $N$ binary variables to $\mathcal{O}(N^2)$ variables and constraints. For large applicant pools, exact solvers encounter combinatorial bottlenecks.
2. **Greedy Heuristics (Value-to-Cost Ratios)**: Greedy policies select items sorted by $C_i / w_i$. However, greedy heuristics are myopic: they cannot account for pairwise group balancing and frequently incur severe fairness violations or violate budget constraints near the knapsack capacity.
3. **Simulated Annealing (SA)**: While SA can optimize arbitrary cost landscapes, classical thermal fluctuations get trapped in deep local minima separated by tall, narrow energy barriers—a signature characteristic of constrained quadratic penalty landscapes.

### 2.3 Why Quantum? The Quantum Value Proposition
Quantum optimization offers distinctive physical advantages for constrained combinatorial search spaces:
- **Quantum Superposition & Multi-State Exploration**: An $N$-qubit register represents a superposition of all $2^N$ candidate loan allocations simultaneously:
  $$|\psi_0\rangle = \frac{1}{\sqrt{2^N}} \sum_{x \in \{0, 1\}^N} |x\rangle$$
- **Quantum Tunneling through Energy Barriers**: Unlike thermal hopping in classical simulated annealing (which scales as $\exp(-\Delta E / k_B T)$), quantum tunneling can penetrate tall, thin potential energy barriers that separate feasible, fair allocation clusters from sub-optimal configurations.
- **Interference-Guided Constructive Amplification**: Quantum algorithms (such as QAOA) leverage phase cancellation to suppress amplitudes of infeasible or unfair bitstrings while constructively interfering the amplitudes of optimal, fair loan distribution states.

---

## 3. Mathematical Mapping: From Problem to QUBO to Ising

### 3.1 Unconstrained Multi-Objective Penalty Formulation
To formulate the constrained optimization problem on quantum hardware, we construct an unconstrained penalty function to be minimized over binary decision variables $x \in \{0, 1\}^N$:
$$\mathcal{L}(x) = - \sum_{i=1}^N C_i x_i + \lambda_B \cdot \Phi_{\text{budget}}(x) + \lambda_F \cdot \Phi_{\text{fairness}}(x)$$
where $\lambda_B, \lambda_F > 0$ are penalty scaling hyperparameters.

#### A. Budget Constraint Penalty ($\Phi_{\text{budget}}$)
Under capital rationing, we aim to exhaust the lending pool up to budget $B$ without exceeding it:
$$\Phi_{\text{budget}}(x) = \left( \sum_{i=1}^N w_i x_i - B \right)^2 = \sum_{i=1}^N w_i^2 x_i^2 + 2 \sum_{i < j} w_i w_j x_i x_j - 2 B \sum_{i=1}^N w_i x_i + B^2$$
Using the binary idempotency property ($x_i^2 = x_i$ for $x_i \in \{0, 1\}$):
$$\Phi_{\text{budget}}(x) = \sum_{i=1}^N (w_i^2 - 2 B w_i) x_i + 2 \sum_{i < j} w_i w_j x_i x_j + B^2$$

#### B. Demographic Fairness Penalty ($\Phi_{\text{fairness}}$)
Let $\mu_A = \frac{1}{|G_A|}$ and $\mu_B = \frac{1}{|G_B|}$. Define group indicator signs:
$$\sigma_i = \begin{cases} +\mu_A & \text{if } i \in G_A \\ -\mu_B & \text{if } i \in G_B \end{cases}$$
The demographic disparity penalty is:
$$\Phi_{\text{fairness}}(x) = \left( \sum_{i=1}^N \sigma_i x_i \right)^2 = \sum_{i=1}^N \sigma_i^2 x_i + 2 \sum_{i < j} \sigma_i \sigma_j x_i x_j$$

### 3.2 Quadratic Unconstrained Binary Optimization (QUBO) Matrix
Summing all terms yields the canonical QUBO form:
$$\min_{x \in \{0, 1\}^N} x^T Q x + \text{const}$$
The upper-triangular QUBO matrix $Q \in \mathbb{R}^{N \times N}$ has components:
- **Diagonal Elements ($Q_{ii}$)**:
  $$Q_{ii} = - C_i + \lambda_B (w_i^2 - 2 B w_i) + \lambda_F \sigma_i^2$$
- **Off-Diagonal Elements ($Q_{ij}$ for $i < j$)**:
  $$Q_{ij} = 2 \lambda_B w_i w_j + 2 \lambda_F \sigma_i \sigma_j$$

### 3.3 Spin Glass Transformation: QUBO to Ising Spin Hamiltonian
To implement the problem on quantum hardware, binary variables $x_i \in \{0, 1\}$ are mapped to spin-half Pauli-Z operators $Z_i \in \{+1, -1\}$ via the affine transformation:
$$x_i = \frac{I - Z_i}{2}$$
where $I$ is the $2 \times 2$ identity operator, and $Z_i$ is the Pauli-Z matrix acting on the $i$-th qubit:
$$Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$
Notice:
- If $Z_i |0\rangle = +1 |0\rangle$, then $x_i = \frac{1 - (+1)}{2} = 0$ (Applicant rejected).
- If $Z_i |1\rangle = -1 |1\rangle$, then $x_i = \frac{1 - (-1)}{2} = 1$ (Applicant approved).

Substituting $x_i = \frac{I - Z_i}{2}$ into $x^T Q x$:
$$x_i x_j = \left( \frac{I - Z_i}{2} \right) \left( \frac{I - Z_j}{2} \right) = \frac{1}{4} (I - Z_i - Z_j + Z_i Z_j)$$
$$x_i = \frac{I - Z_i}{2}$$

Regrouping terms yields the **Ising Problem Hamiltonian**:
$$H_C = \sum_{i=1}^N h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \text{offset} \cdot I$$
where the single-qubit longitudinal fields $h_i$ and two-qubit exchange couplings $J_{ij}$ are:
$$J_{ij} = \frac{1}{4} Q_{ij}$$
$$h_i = - \frac{1}{2} Q_{ii} - \frac{1}{4} \sum_{j \ne i} Q_{ij}$$
$$\text{offset} = \frac{1}{2} \sum_{i=1}^N Q_{ii} + \frac{1}{4} \sum_{i < j} Q_{ij} + \lambda_B B^2$$

The ground state $|\psi_0\rangle$ of $H_C$ corresponds exactly to the optimal, fair loan allocation bitstring $x^*$.

---

## 4. Quantum Algorithmic Paradigms: QAOA vs. VQE

```
                     ┌───────────────────────────────────┐
                     │   Applicant Data & Constraints    │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │     QUBO Matrix Generation        │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │    Ising Hamiltonian Mapping      │
                     │  H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j │
                     └─────────────────┬─────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
     ┌─────────────────────────────┐       ┌─────────────────────────────┐
     │      QAOA Engine            │       │       VQE Engine            │
     │                             │       │                             │
     │  |ψ(γ,β)⟩ =                 │       │  |ψ(θ)⟩ =                   │
     │  Π (e^{-iβ H_M} e^{-iγ H_C})│       │  U_entang · R_y(θ)|0⟩       │
     │  |+⟩^N                      │       │                             │
     └──────────────┬──────────────┘       └──────────────┬──────────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │    Classical Parameter Optimizer  │
                     │    (COBYLA / Nelder-Mead / SLSQP) │
                     │    min ⟨ψ| H_C |ψ⟩                │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   Quantum State Measurement       │
                     │   Sample Optimal Bitstring x*     │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   Fairness & Impact Benchmarking  │
                     │   vs Classical Exact & Greedy     │
                     └───────────────────────────────────┘
```

### 4.1 Quantum Approximate Optimization Algorithm (QAOA)
QAOA, introduced by Farhi, Goldstone, and Gutmann (2014), is a hybrid quantum-classical variational algorithm inspired by the adiabatic theorem of quantum mechanics.

#### 1. Initial State Preparation
The system is initialized in the uniform superposition of all computational basis states:
$$|\psi_0\rangle = |+\rangle^{\otimes N} = \frac{1}{\sqrt{2^N}} \sum_{x \in \{0, 1\}^N} |x\rangle = \bigotimes_{i=1}^N \left( \frac{|0\rangle + |1\rangle}{\sqrt{2}} \right)$$

#### 2. Alternating Unitaries
For a circuit depth of $p$ layers, parameterized by angle vectors $\vec{\gamma} = (\gamma_1, \dots, \gamma_p)$ and $\vec{\beta} = (\beta_1, \dots, \beta_p)$:
- **Problem Unitary $U(H_C, \gamma_k)$**:
  $$U(H_C, \gamma_k) = \exp(-i \gamma_k H_C) = \prod_{i=1}^N \exp(-i \gamma_k h_i Z_i) \prod_{i < j} \exp(-i \gamma_k J_{ij} Z_i Z_j)$$
  In quantum circuit decomposition:
  - $\exp(-i \gamma_k h_i Z_i)$ is implemented via single-qubit $R_Z(2 \gamma_k h_i)$ rotation gates.
  - $\exp(-i \gamma_k J_{ij} Z_i Z_j)$ is implemented via two-qubit $R_{ZZ}(2 \gamma_k J_{ij})$ gates (or a CNOT-$R_Z$-CNOT sequence).
- **Mixer Unitary $U(H_M, \beta_k)$**:
  The transverse-field mixer Hamiltonian is:
  $$H_M = \sum_{i=1}^N X_i$$
  where $X_i$ is the Pauli-X gate. Its unitary operator is:
  $$U(H_M, \beta_k) = \exp(-i \beta_k H_M) = \prod_{i=1}^N \exp(-i \beta_k X_i) = \prod_{i=1}^N R_X(2 \beta_k)$$

#### 3. State Evolution & Classical Optimization Loop
The trial state is:
$$|\psi(\vec{\gamma}, \vec{\beta})\rangle = \prod_{k=1}^p \left( U(H_M, \beta_k) U(H_C, \gamma_k) \right) |\psi_0\rangle$$
The expected energy is measured on quantum hardware:
$$\langle H_C \rangle_{(\vec{\gamma}, \vec{\beta})} = \langle \psi(\vec{\gamma}, \vec{\beta}) | H_C | \psi(\vec{\gamma}, \vec{\beta})\rangle$$
A classical gradient-free or gradient-based optimizer (e.g., COBYLA, SLSQP, SPSA) iteratively updates $(\vec{\gamma}, \vec{\beta})$ to minimize $\langle H_C \rangle$.

### 4.2 Variational Quantum Eigensolver (VQE)
VQE, introduced by Peruzzo et al. (2014), operates on the Rayleigh-Ritz variational principle:
$$E(\vec{\theta}) = \frac{\langle \psi(\vec{\theta}) | H_C | \psi(\vec{\theta}) \rangle}{\langle \psi(\vec{\theta}) | \psi(\vec{\theta}) \rangle} \ge E_0$$
where $E_0$ is the true ground state energy of $H_C$.

In Q-FAR, VQE employs a **Hardware-Efficient Ansatz (TwoLocal / RealAmplitudes)**:
1. Layers of parameterized single-qubit $R_y(\theta_{i, l})$ rotations allow exploration of real probability amplitudes.
2. Entangling layers of Controlled-Z ($CZ$) or Controlled-NOT ($CNOT$) gates introduce multi-qubit correlations to model pairwise loan constraints.
3. The classical optimizer minimizes $E(\vec{\theta})$, driving the quantum state toward the optimal allocation subspace.

### 4.3 Comparison Matrix: QAOA vs. VQE for Resource Allocation

| Criterion | QAOA | VQE (Hardware-Efficient) |
|---|---|---|
| **Ansatz Structure** | Problem-inspired (dictated by $H_C$ and $H_M$) | Generic parameter-rich circuit (e.g., $R_y$ + CNOT) |
| **Number of Parameters** | $2p$ (Very small: e.g., 2 to 6 parameters for $p=1$ to $3$) | $L \times N$ (Higher: e.g., 16 to 48 parameters) |
| **Circuit Depth** | Proportional to problem graph connectivity | Fixed depth determined by chosen layer count $L$ |
| **Susceptibility to Barren Plateaus**| Low for shallow $p$; geometry structured by $H_C$ | Moderate to High as qubit count and layers scale |
| **Interpretability** | High (corresponds to digitized adiabatic quantum computing) | Lower (acts as a variational black-box trial state) |
| **Optimization Landscape**| Periodic in $\beta \in [0, \pi)$ and $\gamma \in [0, 2\pi)$ | Non-convex with many local minima |

---

## 5. Fairness Metrics in Algorithmic Credit Allocation

To rigorously evaluate the allocation decisions, we implement formal algorithmic fairness criteria:

### 5.1 Demographic Parity Difference (DPD)
Demographic parity requires that the probability of loan approval is independent of protected demographic group membership ($G \in \{A, B\}$):
$$\text{DPD} = \left| \frac{\sum_{i \in G_A} x_i}{|G_A|} - \frac{\sum_{j \in G_B} x_j}{|G_B|} \right|$$
- **Ideal Parity**: $\text{DPD} = 0.0$.
- **Fair Lending Standard**: $\text{DPD} \le 0.10$ (10% allowable selection rate difference).

### 5.2 Disparate Impact Ratio (DIR) / Four-Fifths Rule
The Disparate Impact Ratio evaluates the relative approval rate of the protected group compared to the baseline group:
$$\text{DIR} = \frac{\mathbb{P}(\hat{y}=1 | G=A)}{\mathbb{P}(\hat{y}=1 | G=B)} = \frac{\left( \sum_{i \in G_A} x_i \right) / |G_A|}{\left( \sum_{j \in G_B} x_j \right) / |G_B|}$$
Under the US Equal Employment Opportunity Commission (EEOC) and fair lending guidelines, an allocation suffers from disparate impact if $\text{DIR} < 0.80$ (the 80% or Four-Fifths Rule).

### 5.3 Approximation Ratio ($\alpha$)
The quality of the quantum heuristic solution $x_{\text{quantum}}$ relative to the globally optimal classical exact solution $x_{\text{exact}}$ is defined as:
$$\alpha = \frac{\text{Objective}(x_{\text{quantum}})}{\text{Objective}(x_{\text{exact}})} = \frac{\sum_{i=1}^N C_i x_i^{\text{quantum}}}{\sum_{i=1}^N C_i x_i^{\text{exact}}}$$
Feasible solutions satisfy $\sum w_i x_i \le B$, and the composite utility incorporates fairness penalties.

---

## 6. Comprehensive Literature Review & Prior Art

| Reference | Domain / Contribution | Strengths | Limitations Addressed by Q-FAR |
|---|---|---|---|
| **Lucas (2014)**, *Ising formulations of many NP problems* (Frontiers in Physics) | Groundbreaking paper establishing QUBO/Ising penalty mappings for 21 Karp NP-complete problems, including Knapsack. | Provides theoretical foundations for quadratic penalty expansions. | Only addresses single-objective unconstrained or weakly constrained knapsacks without fairness or social metrics. |
| **Farhi et al. (2014)**, *A Quantum Approximate Optimization Algorithm* (arXiv:1411.4028) | Formulated QAOA with alternating problem and mixer Hamiltonians; proved asymptotic convergence. | Establishes the standard variational quantum combinatorial paradigm. | Assumes unconstrained graph problems (Max-Cut). Constrained knapsacks with balance penalties require careful penalty weight calibration. |
| **Peruzzo et al. (2014)**, *A variational eigenvalue solver on a photonic quantum processor* (Nature Comm.) | Introduced VQE for quantum chemistry and Hamiltonian ground state approximation. | Demonstrated feasibility of hybrid quantum-classical NISQ algorithms. | Primarily demonstrated on molecular electronic structure; adapting to combinatorial fairness landscapes is relatively nascent. |
| **Glover, Kochenberger, Du (2018)**, *A Tutorial on Formulating and Solving Problems with QUBO* | Standardized the engineering of QUBO matrices for operations research, portfolio management, and knapsack. | Deep mathematical clarity on quadratic penalty engineering. | Focused on classical heuristic solvers (Tabu search, SA) without quantum circuit mapping or social fairness objectives. |
| **Hardt, Price, Srebro (2016)**, *Equality of Opportunity in Supervised Learning* (NeurIPS) | Defined mathematical fairness constraints (Equalized Odds, Equal Opportunity) in automated decision-making. | Established rigorous foundations for algorithmic bias mitigation. | Evaluated purely in classical classification contexts; does not address resource-constrained combinatorial allocation. |
| **Karlan & Morduch (2009)**, *Access to Finance* (Handbook of Development Economics) | Analyzed credit market failures, asymmetric information, and mission drift in microcredit institutions. | Grounded empirical validation of microfinance poverty impacts. | Econometric study without algorithmic or quantum optimization formulation. |
| **Barkoutsos et al. (2020)**, *Improving Variational Quantum Optimization using CVaR* (Quantum) | Introduced Conditional Value-at-Risk (CVaR) objectives for QAOA/VQE in financial portfolio optimization. | Enhances tail-risk convergence on NISQ devices. | Addressed purely financial variance-return profiles without social inclusion or demographic equity constraints. |

---

## 7. Theoretical Insights & Novelty of Q-FAR

1. **First Formulation of Fair Microfinance Allocation as QUBO**: While quantum portfolio optimization focuses on Markowitz mean-variance models, Q-FAR pioneers the encoding of **social vulnerability, mission drift mitigation, and multi-group demographic parity** directly into the Hamiltonian.
2. **Dual-Mode Algorithmic Engine**: Provides side-by-side benchmarking of **QAOA vs. VQE** on identical Hamiltonian instances, exposing key structural trade-offs in parameter sensitivity, circuit depth, and ground-state reachability.
3. **Rigorous Tri-Objective Landscape**: Unifies financial viability, immediate humanitarian need, and group equity into a tunable Hamiltonian framework suitable for deployment on near-term NISQ quantum processors.
