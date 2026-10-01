# Q-FAR: Quantum-Assisted Fair Resource Allocation for Financial Inclusion in Microfinance

[![Course](https://img.shields.io/badge/Course-QCAA_Midsem-blue.svg)](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/mid_sem_plan_and_progress.md)
[![Status](https://img.shields.io/badge/Completion-60%25_Milestone-green.svg)](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/midsem_pipeline.py)
[![Algorithms](https://img.shields.io/badge/Algorithms-QAOA_|_VQE_|_QUBO-purple.svg)](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/algorithmic_study.md)
[![Framework](https://img.shields.io/badge/Stack-Qiskit_|_NumPy_|_SciPy-orange.svg)](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/requirements.txt)

> **QCAA Mid-Semester Capstone Project**  
> Formulating, simulating, and evaluating hybrid quantum-classical combinatorial optimization algorithms (QAOA and VQE) for fair, multi-objective capital rationing under strict budget and demographic equity constraints.

---

## 1. System Architecture & End-to-End Pipeline

```
MANY PEOPLE NEED LOANS
          ↓
    LIMITED MONEY (Budget B)
          ↓
  COLLECT APPLICANT DATA
 (income, dti, poverty_index,
  repayment fidelity, group)
          ↓
   CALCULATE SCORES
 (Financial F_i + Need N_i + Social S_i)
          ↓
   SET OUR OBJECTIVES
 Multi-Objective Utility C_i
 + Group Demographic Fairness
          ↓
   ADD BUDGET LIMIT
 Hard/Quadratic Capacity Ceiling
          ↓
 MAKE OPTIMIZATION MODEL
 Constrained Quadratic Knapsack (NP-Hard)
          ↓
        QUBO
 Upper-Triangular Matrix Q ∈ R^{N×N}
          ↓
  ┌───────┴───────┐
  ↓               ↓
 QAOA            VQE
(Quantum)      (Quantum)
  └───────┬───────┘
          ↓
 GET ALLOCATION RESULT (x* ∈ {0, 1}^N)
          ↓
 COMPARE WITH CLASSICAL
 (Greedy Ratio, Classical Exact, Simulated Annealing)
          ↓
     FINAL RESULT & BENCHMARK VISUALS
 (Utility, Fairness DPD, DIR, Budget Utilization)
```

---

## 2. Core Problem Definition & How Quantum Enters the Project

### 2.1 The Microfinance Dilemma: "Mission Drift"
Microfinance institutions (MFIs) seek to provide credit access to underbanked, vulnerable individuals lacking collateral. However, when loan demand exceeds available lendable capital $\sum w_i > B$, standard financial credit scoring favors lower-risk, wealthier borrowers—causing **mission drift** and shutting out marginalized rural women and micro-entrepreneurs.

### 2.2 Mathematical Formulation: Multiobjective Quadratic Knapsack
To balance financial viability ($F_i$), urgent humanitarian need ($N_i$), and social community impact ($S_i$), we construct composite benefit coefficients:
$$C_i = \omega_F F_i + \omega_N N_i + \omega_S S_i, \quad \sum \omega = 1$$

Enforcing group demographic parity (between Group A: Marginalized and Group B: General) requires equalizing approval rates:
$$\left( \frac{1}{|G_A|} \sum_{i \in G_A} x_i - \frac{1}{|G_B|} \sum_{j \in G_B} x_j \right)^2$$

This creates **quadratic cross-terms ($x_i x_j$)**, turning the standard 0-1 Knapsack into a strongly **NP-hard Quadratic Knapsack Problem (QKP)**.

### 2.3 Mapping Binary Variables to Quantum Qubits (Ising Spin Glass)
Binary decisions $x_i \in \{0, 1\}$ (0: rejected, 1: approved) map directly to quantum spin-half Pauli-Z operators $Z_i \in \{+1, -1\}$ via the affine transformation:
$$x_i = \frac{I - Z_i}{2}$$

Substituting this into the Quadratic Unconstrained Binary Optimization (QUBO) penalty function yields the **Ising Problem Hamiltonian**:
$$H_C = \sum_{i=1}^N h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \text{offset} \cdot I$$

The true ground state $|\psi_0\rangle$ of $H_C$ encodes the globally optimal, fair loan distribution.

### 2.4 Quantum Variational Paradigms: QAOA vs. VQE
1. **Quantum Approximate Optimization Algorithm (QAOA)**:
   - Uses alternating problem unitaries $U_C(\gamma) = e^{-i \gamma H_C}$ and transverse mixer unitaries $U_M(\beta) = e^{-i \beta \sum X_i}$.
   - The circuit geometry natively encodes the coupling graph of the microfinance constraint matrix.
2. **Variational Quantum Eigensolver (VQE)**:
   - Employs a hardware-efficient parameterized ansatz ($R_y(\theta)$ single-qubit rotations with circular Controlled-Z entanglement).
   - Minimizes $\langle \psi(\vec{\theta}) | H_C | \psi(\vec{\theta}) \rangle$ via classical COBYLA optimization.

---

## 3. Three-Phase Project Plan & 60% Midsem Checkpoint

| Phase | Scope & Modules | Target Status | Midsem State |
|:---:|---|:---:|:---:|
| **Phase 1** | **Mathematical Modeling & Classical Baselines**<br>• Applicant generator & tri-objective scoring<br>• QUBO quadratic penalty formulation<br>• Classical Greedy, Exact ILP, and Simulated Annealing | 100% | **COMPLETED** |
| **Phase 2** | **Quantum Variational Circuits & Midsem Evaluation**<br>• QUBO-to-Ising Hamiltonian mapping ($h_i, J_{ij}$)<br>• QAOA ($p=1, 2$) and VQE execution engines<br>• Native Qiskit circuit export (`qiskit_qaoa_circuit.txt`)<br>• Multi-metric comparison (DPD, DIR, Approx Ratio)<br>• 5 high-resolution presentation visual plots | 50% | **COMPLETED & VERIFIED** |
| **Phase 3** | **Hardware Scaling & Noise Mitigation (End-Sem Roadmap)**<br>• Real execution on IBM Quantum superconducting QPUs<br>• Zero-Noise Extrapolation (ZNE) error mitigation<br>• Warm-start QAOA & interactive Streamlit web dashboard | 0% | *Planned for End-Sem* |

$$\mathbf{Net\ Midsem\ Progress} = \underbrace{40\%}_{\text{Phase 1}} + \underbrace{20\%}_{\text{Phase 2 (50\% done)}} + \underbrace{0\%}_{\text{Phase 3}} = \mathbf{60.0\%}$$

---

## 4. Repository Structure

```
QCAA/
├── README.md                          # Master documentation and visual roadmap
├── pseudo.md                          # Formal publication-grade pseudocode suite
├── mid_sem_plan_and_progress.md       # 60% progress report, rubrics, & slide guide
├── algorithmic_study.md               # In-depth theory, literature review & complexity
├── requirements.txt                   # Dependency specifications
│
├── src/                               # Modular Python Package
│   ├── __init__.py
│   ├── data_generator.py              # Statistical cohort generator with demographic labels
│   ├── scoring.py                     # Tri-objective normalization (Financial, Need, Impact)
│   ├── qubo_builder.py                # QUBO assembly & Ising Hamiltonian mapping
│   ├── classical_solvers.py           # Greedy Knapsack, Exact Combinatorial, Simulated Annealing
│   ├── quantum_engine.py              # QAOA, VQE, Statevector evolution & Qiskit exporter
│   ├── metrics.py                     # Fairness metrics (DPD, DIR), Approx Ratio, Feasibility
│   └── visualizer.py                  # High-resolution presentation figures generator
│
├── midsem_pipeline.py                 # Designated 60% Midsem Benchmark Pipeline
│
├── visuals/                           # High-Resolution Presentation Visuals
│   ├── qubo_matrix_heatmap.png        # Upper-triangular QUBO interaction heatmap
│   ├── quantum_vs_classical_comparison.png # 4-panel comparative benchmark bar charts
│   ├── qaoa_convergence_profile.png   # Energy minimization trajectory over optimizer steps
│   ├── fairness_vs_budget_tradeoff.png# Pareto frontier (Demographic Equity vs Return)
│   ├── bitstring_probability_distribution.png # Constructive interference measurement peak
│   └── qiskit_qaoa_circuit.txt        # Native Qiskit QAOA circuit ASCII diagram
│
└── data/
    └── sample_applicants.csv          # Benchmark cohort of microfinance borrowers
```

---

## 5. Quickstart & Reproducibility

### Installation
Ensure Python 3.10+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Midsem Pipeline (60% Checkpoint)
Execute the complete midsem pipeline in a single command:
```bash
python midsem_pipeline.py
```
This executes all classical and quantum solvers, prints the comparative benchmark table, updates `data/sample_applicants.csv`, and writes all visual analytics figures into `visuals/`.

---

## 6. Midsem Experimental Results Summary

The midsem pipeline benchmarks 6 solver configurations on an identical applicant cohort under a capital ceiling of INR 110,000:

| Solver | Approved | Total Utility | Budget Util % | Feasible | Demographic Disparity (DPD) | Disparate Impact (DIR) | Approx Ratio |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Classical Greedy** | 4 | 2.01 | 86.4% | Yes | **0.500** *(Severe Bias!)* | 0.33 *(Fails 80% Rule)* | 0.81 |
| **Classical Exact (ILP)** | 4 | 2.48 | 95.5% | Yes | **0.000** *(Perfect Parity)* | 1.00 *(Ideal Equity)* | **1.00** |
| **Simulated Annealing** | 4 | 2.48 | 95.5% | Yes | **0.000** | 1.00 | **1.00** |
| **Quantum QAOA ($p=1$)** | 4 | 2.29 | 90.9% | Yes | **0.000** | 1.00 | **0.92** |
| **Quantum QAOA ($p=2$)** | 4 | 2.48 | 95.5% | Yes | **0.000** | 1.00 | **1.00** |
| **Quantum VQE (Hardware)** | 4 | 2.34 | 93.2% | Yes | **0.250** | 0.67 | **0.94** |

### Key Experimental Insights:
1. **Classical Greedy Fails Social Equity**: Traditional greedy ratio knapsack allocates loans purely based on cost-efficiency, suffering a massive 50% demographic disparity and violating the Four-Fifths fair lending rule ($\text{DIR} = 0.33 < 0.80$).
2. **QAOA ($p=2$) Matches Exact Ground Truth**: Increasing QAOA circuit depth from $p=1$ to $p=2$ achieves an approximation ratio of **1.00**, eliminating group disparity ($\text{DPD} = 0.00$) while maximizing multi-objective utility within budget.
3. **Constructive Quantum Interference**: The sampled measurement probability distribution displays a prominent peak at the optimal ground state allocation, validating the variational phase-interference mechanism.

---

## 7. Presentation Visuals Guide

All figures are automatically generated in `visuals/` for immediate inclusion in presentation slides:
- [qubo_matrix_heatmap.png](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/visuals/qubo_matrix_heatmap.png): Visualizes the quadratic constraint couplings.
- [quantum_vs_classical_comparison.png](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/visuals/quantum_vs_classical_comparison.png): 4-panel comparison of utility, fairness, budget utilization, and approximation ratio.
- [qaoa_convergence_profile.png](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/visuals/qaoa_convergence_profile.png): Energy minimization trajectories over COBYLA iterations.
- [fairness_vs_budget_tradeoff.png](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/visuals/fairness_vs_budget_tradeoff.png): Multi-objective Pareto frontier showing fairness vs return.
- [bitstring_probability_distribution.png](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/visuals/bitstring_probability_distribution.png): Quantum state measurement histogram highlighting ground state peak.
- [qiskit_qaoa_circuit.txt](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/QCAA/visuals/qiskit_qaoa_circuit.txt): ASCII diagram of the parameterized Qiskit quantum circuit.

---

## 8. Midsem Defense FAQ

- **Q: Why use quantum computing for a knapsack problem?**  
  *A*: Standard knapsack is weakly NP-complete, but incorporating **pairwise demographic fairness penalties creates quadratic cross-terms ($x_i x_j$)**, rendering it a **Quadratic Knapsack Problem (QKP)**, which is strongly NP-hard. Classical exact solvers scale as $\mathcal{O}(2^N)$, while quantum algorithms leverage superposition and tunneling across non-convex energy barriers.
- **Q: What is the current progress level?**  
  *A*: Exactly **60% accomplished**. Phase 1 (problem formulation, scoring, QUBO mapping, classical baseline suite) is 100% complete. Phase 2 (QAOA and VQE engines, comparative metrics, presentation visual analytics) is implemented and verified. Phase 3 (IBM Quantum cloud hardware execution, ZNE noise mitigation, web dashboard) represents the remaining 40% reserved for end-sem.
