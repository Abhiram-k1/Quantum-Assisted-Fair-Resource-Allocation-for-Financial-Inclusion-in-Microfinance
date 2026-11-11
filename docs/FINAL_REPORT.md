# Q-FAR Final Report: Architecture Audit, End-to-End Pipeline, Scaling & NISQ Evaluation

> **Truthfulness & scope notice.** All empirical metrics reported below are directly observed from reproducible runs on Q-FAR v2's synthetic 6,000-applicant dataset (`pool_size: 6000`, `60/20/20` train/validation/test split) and legacy cohort `../data/sample_applicants.csv`. They measure algorithmic and circuit-level behaviour on this simulation testbed rather than claims of quantum advantage or production deployment.

---

## 1. Conceptual Audit: Original Plan (`QCAA/` v1) vs. `qfar` v2 & Final Extensions

| Dimension | Original v1 State (`QCAA/` HEAD) | Q-FAR v2 Core Corrections (`FIXES.md` 1–23) | Final End-Sem Completions (This Delivery) |
| :--- | :--- | :--- | :--- |
| **Budget constraint encoding** | Inequality encoded as $(\sum w_i x_i - B)^2$ with $\lambda_B = 0.85$; QUBO ground state spent **118.2%** of ceiling (`130k / 110k` INR), QAOA/CVaR `109.1%`, SA `113.6%`. | Binary slack variables $s_0,\dots,s_{K-1}$ converting $w^\top x \le B$ to equality plus analytical lower bound $\lambda_B = 1.1\,\lambda_{B,\min}$; **100% raw feasibility**. | Verified 100% feasibility on v1 cohort [sample_applicants.csv](../../../data/sample_applicants.csv) (`115,000 / 115,000` INR, `J1 = -2.9887`). |
| **Data hygiene & scoring** | 8-row train+test leakage; cohort-dependent normalization; `demographic_group` leaked into `S_i`. | 6,000-row disjoint split; population-fixed score bounds; group-neutral `F, N, S`; Option A (`inclusion`) + Option B (`priority`) fallback. | Unified `normalize_applicant_df` auto-mapping both v2 cohorts and legacy `Group_A_Marginalized` / `Group_B_General` CSVs. |
| **Risk & objective separation** | Conflated Barkoutsos energy-CVaR (`cvar_qaoa`) with portfolio default CVaR; independent defaults. | Cleanly separates **P1** (QUBO: utility + variance + soft parity), **P2** (+ Rockafellar-Uryasev loss CVaR MILP under hierarchical copula), and energy-CVaR. | End-to-end unified runner comparing P1, P2, hard-fairness MILP, greedy, SA, QAOA, CVaR-QAOA, VQE, and Warm-Start QAOA side by side. |
| **Quantum optimisation & NISQ** | 8-qubit Qiskit circuit with zero slack qubits; no systematic scaling beyond $N=8$; no noise/ZNE benchmark. | Exact noise-free `2^n` statevector QAOA ($p=1..3$), CVaR-QAOA, VQE; honest finding that standard QAOA concentrates on feasible subspace (~88% vs 51%) rather than $x^*$. | **Warm-Start QAOA** (Egger et al. 2021 QP relaxation + aligned mixer), **Qiskit circuit compiler + depolarizing/readout noise + Richardson ZNE** (`c=1,3,5`), **$N \in \{4,6,8,10\}$ scaling**. |
| **Deliverables & UX** | Disconnected prototype scripts; unmaintained metrics. | Reproducible experiment scripts `01`–`04` and 482 unit tests. | **Unified CLI (`run_qfar.py`), `pipeline.py`, 8 PNG figures (`results/figures/`), interactive Dash+Plotly dashboard (`07_dashboard.py`), full test coverage.** |

---

## 2. Explicit Statement of Novelty (Bounded & Verified)

Instead of unverified quantum-speedup claims, the verified engineering and methodological novelty of Q-FAR rests on five concrete, tested pillars:

1. **Slack-exact inequality QUBO with provable penalty sufficiency (`qubo.py`):** Exact binary decomposition (`budget_units` + `slack_coefficients`) combined with `penalty_lower_bound` eliminates the 109%–118% budget-violation failure mode completely (0/30 infeasible raw ground states in `03_allocation`).
2. **Three-way mathematical separation (`risk.py`, `solvers.py`, `quantum.py`):** Distinguishes Markowitz default-covariance risk + soft quadratic parity penalty (**P1**, QUBO-representable), tail-loss Rockafellar-Uryasev CVaR (**P2**, classical MILP with hierarchical global+village latent defaults), and Barkoutsos lower-tail energy CVaR (**CVaR-QAOA**, quantum objective shaping).
3. **Neutrality-by-construction scoring with dual inclusion/priority schema (`scoring.py`, `pipeline.py`):** `F`, `N`, and `S` never ingest protected group labels (`|DPD|` penalty enters exclusively at the portfolio optimization layer), while supporting both multi-dimensional inclusion columns and legacy priority attributes.
4. **QP-relaxed Warm-Start QAOA over slack-augmented microfinance QUBOs (`quantum.py`):** Solving the continuous convex relaxation $\min_{x \in [0,1]^N} J_1(x)$ s.t. $w^\top x \le B$ with fractional slack decomposition initializes $|\phi^*\rangle$ inside the low-energy feasible basin and overcomes standard QAOA's ground-state barren dilution as cohort size grows ($N = 4 \to 10$).
5. **Dual-regime Qiskit NISQ noise + Richardson ZNE backend (`qiskit_backend.py`):** Pairs exact Kraus `DensityMatrix` simulation ($n \le 8$) with unbiased L3-cache-chunked Monte Carlo trajectory simulation ($n > 8$) plus local unitary gate folding ($c \in \{1, 3, 5\}$) without OOM on 8 GB workstations.

---

## 3. Summary of Verified Empirical Results (RQ1–RQ3 & Extensions)

### 3.1 PD Model & Score Independence (`01_train_pd.py`)
- Selected model (`validation_selection.csv`): `logreg` (`calibration: none`, `features: full`, 6 features).
- Held-out test metrics (`n = 1200`, `test_metrics.json`):
  - Overall: ROC-AUC = **0.7417**, PR-AUC (default) = **0.3437**, Brier = **0.1017**, ECE = **0.0104**.
  - Group A (`n = 614`): base repayment rate = `0.7980`, mean predicted = `0.8021`, ECE = `0.0153`.
  - Group B (`n = 586`): base repayment rate = `0.9420`, mean predicted = `0.9429`, ECE = `0.0148`.

### 3.2 Classical & Quantum Allocation across 30 Test Cohorts (`03_allocation_experiment.py`, $N = 8$, $n \approx 13$ qubits)
- **Exact P1 (`exact_enumeration_P1` / `milp_linearised`):** $J_1 = -2.6244$, $J_2 = -1.8476$, $|\text{DPD}| = 0.2078$, 95% CVaR = `46,608` INR, `raw_feasible = 1.0`.
- **Exact P2 (`exact_enumeration_P2` / `milp_linearised_cvar`):** reduces tail CVaR by **8.6%** to `42,600` INR ($J_2 = -1.8830$, $|\text{DPD}| = 0.1928$).
- **Greedy baseline (`greedy_utility_per_cost`):** severely unfair ($|\text{DPD}| = 0.7289$) and high tail risk (`54,747` INR CVaR, `hit_opt_P1 = 0.10`).
- **Simulated Annealing:** near-exact ($J_1 = -2.6126$, `gap_J1 = 0.0043`, `hit_opt_P1 = 0.8667`).
- **Standard QAOA (`qaoa_p1..p3`):** raises feasibility mass from uniform `0.5112` to `0.8401` (`p=1`), `0.8852` (`p=2`), and `0.9115` (`p=3`), with `hit_opt_P1 = 0.4000` at `p=2`, but single-shot $P(x^*)$ stays near uniform (`0.0032`–`0.0039`).
- **Component Ablation (`04_component_ablation.py`):** moving from utility-only `P0` (`|DPD| = 0.6522`, `CVaR = 58,347` INR) to `P1_variance+fairness` cuts `|DPD|` by **78.5%** (`0.1400`) and CVaR by `10,345` INR (`48,002` INR); adding `lambda_R = 0.5` (`P2`) lowers CVaR further to `44,202` INR (`|DPD| = 0.1194`).

### 3.3 Scaling ($N \in \{4, 6, 8, 10\}$): Standard QAOA vs. Warm-Start QAOA (`06_scaling_and_noise.py`, 10 cohorts/size)

| $N$ (Qubits) | `qaoa_p1` $P(x^*)$ | `qaoa_p1` `hit_P1` | `qaoa_p1` `gap_J1` | `warm_start_qaoa_p1` $P(x^*)$ | `warm_start_qaoa_p1` `hit_P1` | `warm_start_qaoa_p1` `gap_J1` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** (`8.0`) | `0.0810` | `1.0` | `0.0000` | **`0.3899`** (`6.2x` unif) | `0.9` | `0.0072` |
| **6** (`10.6`) | `0.0174` | `1.0` | `0.0000` | **`0.2954`** (`18.9x` unif) | `1.0` | `0.0000` |
| **8** (`13.0`) | `0.0040` | `0.4` | `0.0307` | **`0.1379`** (`35.3x` unif) | `0.7` | `0.0100` |
| **10** (`15.0`) | `0.0009` | **`0.0`** | `0.0864` | **`0.1634`** (**`163x` unif**) | **`0.9`** | **`0.0009`** |

At $N=10$ (15 qubits, $2^{10}=1024$ applicant portfolios), standard `qaoa_p1` drops to `0%` hit rate (`8.64%` optimality gap), whereas **`warm_start_qaoa_p1` achieves a 90% optimal hit rate (`0.09%` optimality gap), beating both standard QAOA (`8.64%`) and SA (`1.01%` gap, `70%` hit rate)**.

### 3.4 Qiskit Depolarising + Readout Noise & Zero-Noise Extrapolation (`p1=0.002, p2=0.01, p_readout=0.02`, folding `c=1,3,5`)

| $N$ (Qubits) | Solver | Ideal $\langle H \rangle$ | Noisy $c=1$ $\langle H \rangle$ | Richardson ZNE(0) | $|E_{c=1} - E_{\text{id}}|$ | $|E_{\text{ZNE}} - E_{\text{id}}|$ | Noisy `gap_J1` ($c=1$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** (`8.0`) | `qaoa_p1` | `73.24` | `87.10` | **`78.88`** | `13.86` | **`5.64`** (-59%) | `0.0000` |
| **4** (`8.0`) | `warm_start_qaoa_p1` | `26.10` | `45.97` | **`33.34`** | `19.86` | **`7.24`** (-64%) | `0.0072` |
| **6** (`10.6`) | `qaoa_p1` | `114.97` | `171.61` | **`139.11`** | `56.64` | **`24.14`** (-57%) | `0.0026` |
| **6** (`10.6`) | `warm_start_qaoa_p1` | `44.06` | `113.32` | **`61.82`** | `69.26` | **`19.17`** (-72%) | **`0.0000`** |
| **8** (`13.0`) | `qaoa_p1` | `219.44` | `340.25` | **`245.35`** | `120.81` | **`29.61`** (-75%) | `0.0394` |
| **8** (`13.0`) | `warm_start_qaoa_p1` | `99.24` | `248.71` | **`153.79`** | `149.48` | **`58.21`** (-61%) | **`0.0012`** |

---

## 4. Quick Usage Guide for Working Module & Dashboard

- **Unified CLI (synthetic cohort `0` with Qiskit circuit & ZNE):**
  ```bash
  python scripts/run_qfar.py --cohort 0 --solver warm_start_qaoa_p1 --qiskit --zne
  ```
- **Run on legacy / custom CSV cohort (including `../data/sample_applicants.csv`):**
  ```bash
  python scripts/run_qfar.py --csv ../data/sample_applicants.csv --solver warm_start_qaoa_p1 --qiskit --zne
  ```
- **Launch interactive web dashboard (`http://127.0.0.1:8050`):**
  ```bash
  python scripts/07_dashboard.py --port 8050
  ```
- **Programmatic Python API:**
  ```python
  from qfar import run_allocation
  res = run_allocation(cohort_idx=0, primary_solver="warm_start_qaoa_p1", include_qiskit_circuit=True)
  print(res.selected_ids)
  print(res.comparison)
  ```

