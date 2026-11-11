# Q-FAR: fairness- and risk-aware microfinance loan allocation

Q-FAR chooses which loan applicants to approve under a fixed budget. It balances
three things:

* **utility**: financial viability, applicant need, and financial inclusion;
* **portfolio risk**: variance of losses, plus financial CVaR as an option;
* **fairness**: the gap in approval rates between two groups.

The problem is written once as a QUBO and solved by exact classical methods,
classical heuristics, and simulated variational quantum algorithms (QAOA,
CVaR-QAOA, VQE).

> **Scope.** This is a **synthetic-data simulation study.** Applicants and their
> repayment outcomes come from a known simulator (`docs/DATA_CARD.md`). Nothing
> here establishes real-world repayment-prediction accuracy or real-world
> allocation outcomes. All quantum results come from **ideal (noise-free)
> statevector simulation**, not quantum hardware.

---

## Contents

```
configs/default.yaml          every parameter; copied into each run's manifest.json
src/qfar/
  config.py                   config loading and run manifests (seed, versions, git commit)
  data.py                     synthetic generator, real labelled-CSV loader, splits, cohorts
  pd_model.py                 repayment model: CV on train, selection on validation, test once
  scoring.py                  F, Need, Inclusion scores with FIXED normalisation; utility U
  risk.py                     copula default model, variance matrix, scenarios, financial CVaR
  qubo.py                     QUBO with slack-bit budget, proven penalty bound, Ising map
  solvers.py                  exact enumeration, linearised MILP (P1/P2), hard-fair MILP, greedy, SA
  quantum.py                  statevector QAOA, CVaR-QAOA, VQE; circuit resource counts
  metrics.py                  directional fairness, return/loss/CVaR, realised loss, mean±SD
  experiment.py               cohort -> one shared Instance -> metric rows
scripts/
  01_train_pd.py              stage 1: repayment model, scored test applicants
  02_weight_ablation.py       weight sensitivity over the simplex (30 cohorts x 66 weights)
  03_allocation_experiment.py solver comparison on identical cohorts (30 cohorts)
  04_component_ablation.py    score-component (E1-E7) and penalty ablations (30 cohorts)
tests/                        pytest suite (482 tests)
docs/FORMULATION.md           every equation, its implementation, and its test
docs/DATA_CARD.md             synthetic data, group definitions, column roles
docs/FIXES.md                 item-by-item list of corrections to v1
results/                      CSV/JSON outputs, each folder with a manifest.json
```

## Quick start

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

```bash
make all
```

`make all` runs the tests and then scripts 01→04. On 4 CPU cores it takes about 4
minutes. Each script writes to `results/<stage>/` and saves a `manifest.json`
holding the full config, seeds, package versions and git commit. Run `01` before
the others, because it creates `data/processed/test_scored.csv`.

---

## The model in one page

Full derivations and proofs are in [`docs/FORMULATION.md`](docs/FORMULATION.md).

**Scores.** All bounds are fixed in the config, so an applicant's score never
depends on who else is in the cohort.

| score | definition | note |
|---|---|---|
| $F_i$ | $\mu_i=p_ir-(1-p_i)LGD$; $F_i=(\mu_i+LGD)/(r+LGD)$ | $r=0.2$, $LGD=1$ ⇒ $F_i=(\mu_i+1)/1.2=p_i$. A normalised expected-return score, **not profit** |
| $N_i$ | $0.5P_i+0.3D_i+0.2(1-I_i)$ | $I_i$ is per-capita income on a fixed [1k, 25k] range; $D_i=\min(\text{dep}/6,1)$ |
| $S_i$ | Inclusion Score (Option A): $0.25\,(E_1+E_2+E_3+E_4)$ | first-time formal borrower, no formal account, underserved locality, branch distance. Falls back to the "Social Priority Score" (Option B) if these columns are missing |
| $U_i$ | $\omega_FF_i+\omega_NN_i+\omega_SS_i$ | declared baseline $\omega=(1/3,1/3,1/3)$ |

The three scores are separate decision dimensions, but they are correlated. On
test applicants the Spearman correlations are $\rho(F,N)=-0.68$, $\rho(F,S)=-0.34$
and $\rho(N,S)=0.41$.

**P1: the QUBO problem** (variance-aware, *soft* fairness). Solved by every solver.

$$E(x,s)=-\sum_iU_ix_i+\eta\,x^\top\Gamma x+\lambda_F\,d(x)^2+\lambda_B\Big(\sum_iw_ix_i+\sum_kc_ks_k-B\Big)^2,\qquad \lambda_B>\sum_i\max(U_i,0)$$

* $\Gamma$ is the covariance of portfolio loss under a Gaussian-copula default
  model with a global and a village factor. This term is **variance, not CVaR.**
* $d(x)=AR_A-AR_B$. The fairness penalty is **soft**: it encourages parity but
  does not guarantee it.
* With $\lambda_B>\sum_i\max(U_i,0)$, every ground state is provably
  budget-feasible and optimal for P1. The proof is in FORMULATION §3.2, and the
  tests check it exhaustively and by sampling up to $N=80$.

**P2: the tail-risk problem.** P1 plus $\lambda_R\,CVaR_{0.95}$ of scenario
portfolio losses (2,000 copula scenarios per cohort, Rockafellar–Uryasev form).
P2 is solved **exactly by a MILP**, which the tests check against brute force.
**There is no QUBO encoding of CVaR, so the quantum solvers optimise P1 only.**

**Solvers**

| solver | solves |
|---|---|
| `exact_enumeration_P1/P2` | P1 / P2 exactly (reference optimum) |
| `milp_linearised` | P1 exactly (McCormick-linearised products), the same problem as the QUBO |
| `milp_linearised_cvar` | P2 exactly |
| `milp_hard_fair_utility` | **a separate policy baseline**: max $U^\top x$ s.t. budget and $\lvert d\rvert\le0.25$, no risk term. Its objective value is not comparable with P1 |
| `greedy_utility_per_cost` | Greedy Utility-per-Cost Baseline. Ignores risk and fairness |
| `simulated_annealing` | P1 QUBO, heuristic |
| `qaoa_p{1,2,3}`, `cvar_qaoa_p1_a0.25`, `vqe_L2` | P1 QUBO, statevector simulation, 1024 shots; returns the **lowest-energy sample** |

"CVaR" in CVaR-QAOA means the mean of the lowest 25% of measured **QUBO
energies**. It is unrelated to the **financial** CVaR of portfolio losses.

---

## Results (synthetic data, seed 20261110)

### Repayment model (`results/01_pd_model/`)

The selected model is logistic regression (uncalibrated, full feature set). It
was chosen by 5-fold CV on train (3,600) and then on validation (1,200), and
evaluated once on test (1,200).

| held-out test | ROC-AUC | Brier | ECE | mean p̂ / observed |
|---|---|---|---|---|
| overall | 0.742 | 0.102 | 0.010 | 0.871 / 0.868 |
| Group A | 0.699 | 0.146 | 0.015 | 0.802 / 0.798 |
| Group B | 0.586 | 0.055 | 0.015 | 0.943 / 0.942 |

Calibration is good in both groups. Ranking power is weaker within Group B,
where defaults are rare. These numbers describe how well the model recovers the
simulator; they are not real-world performance.

### Solver comparison (`results/03_allocation/`): 30 cohorts, N = 8, 13 qubits

The table shows means across cohorts; the CSV also has the SD of every metric.
A "hit" means the solver returned the exact optimum of that problem. $\eta=1$,
$\lambda_F=2$, $\lambda_R=0.5$.

| solver | hit P1 | hit P2 | J1 ↓ | approved | exp. loss (₹) | CVaR95 (₹) | \|DPD\| | DIR A/B | runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| exact / MILP (P1) | 1.00 | 0.60 | −2.624 | 4.37 | 10,479 | 46,608 | 0.21 | 1.51 | <0.1 |
| exact / MILP-CVaR (P2) | 0.60 | 1.00 | −2.593 | 4.30 | 9,090 | **42,600** | 0.19 | 1.43 | 0.6 |
| hard-fair utility MILP (separate problem) | 0.37 | 0.20 | −2.299 | 3.70 | 9,701 | 46,368 | **0.07** | 1.13 | <0.1 |
| greedy utility-per-cost | 0.10 | 0.10 | −1.691 | 4.53 | 15,956 | 54,747 | 0.73 | 2.96 | <0.01 |
| simulated annealing | 0.87 | 0.53 | −2.613 | 4.37 | 10,659 | 46,885 | 0.22 | 1.55 | 0.14 |
| QAOA p=1 | 0.27 | 0.13 | −2.445 | 4.10 | 10,668 | 47,597 | 0.17 | 1.44 | 1.4 |
| QAOA p=2 | 0.23 | 0.13 | −2.511 | 4.23 | 10,696 | 47,105 | 0.20 | 1.51 | 2.1 |
| QAOA p=3 | 0.33 | 0.13 | −2.506 | 4.20 | 10,096 | 46,567 | 0.20 | 1.50 | 2.7 |
| CVaR-QAOA p=1 | 0.23 | 0.17 | −2.461 | 4.10 | 10,505 | 45,628 | 0.15 | 1.39 | 1.5 |
| VQE (2 layers) | 0.03 | 0.03 | −1.438 | 2.67 | 7,557 | 39,227 | 0.23 | 1.19 | 4.7 |

Quantum diagnostics (`quantum_summary.csv`). The uniform-random baselines are
$P_{opt}=1/256=0.0039$ and $P_{feasible}=0.51$.

| | depth | 2-qubit gates | evals | $P_{opt}$ (exact) | $P_{feasible}$ (exact) | most-frequent sample feasible |
|---|---|---|---|---|---|---|
| QAOA p=1 | 72 | 156 | 1000 | 0.0039 | 0.84 | 0.90 |
| QAOA p=2 | 113 | 312 | 1000 | 0.0035 | 0.88 | 0.90 |
| QAOA p=3 | 154 | 468 | 1000 | 0.0032 | 0.91 | 0.97 |
| CVaR-QAOA p=1 | 72 | 156 | 1000 | 0.0035 | 0.86 | 0.90 |
| VQE L=2 | 29 | 26 | 1600 | 0.029 (median 0.000) | 1.00 | 1.00 |

What the results show:

1. **The encoding is correct.** In all 30 cohorts the QUBO ground state is
   feasible and equals the P1 optimum (the script asserts this). Enumeration and
   the linearised MILP agree on both P1 and P2.
2. **Financial CVaR in the objective lowers tail risk.** Compared with P1, P2
   reduces CVaR95 by about 9% (₹46.6k → ₹42.6k) and expected loss by 13%, while
   J1 rises by 1.3%.
3. **QAOA and VQE do not beat classical methods here.**
   * QAOA's probability of the optimal allocation is no better than random
     guessing (0.8–1.0× the uniform baseline), and it does not improve with depth.
   * What QAOA learns is **feasibility**: P(feasible) is 0.84–0.91 against 0.51
     for random guessing. This is because the budget penalty dominates the
     energy scale.
   * Its occasional hits (23–33%) come from best-of-1024 sampling over only 256
     possible allocations, not from amplifying the optimum.
   * VQE collapses onto a near-classical, overly cautious allocation.
   * Simulated annealing hits the optimum in 87% of cohorts, at a fraction of
     the cost.
4. **Greedy utility-per-cost** ignores risk and fairness. It gives the largest
   approval-rate gap and the highest losses of all solvers.

### Weight sensitivity (`results/02_weight_sensitivity/`): 30 cohorts × 66 weights

The mean Jaccard similarity to the baseline allocation across the simplex is
0.84. Single-objective corners change the decision a lot:

| weights (F, N, S) | Jaccard vs baseline | Kendall τ vs baseline | exp. loss (₹) | CVaR95 (₹) |
|---|---|---|---|---|
| (1, 0, 0) | 0.68 | −0.25 | 8,498 | 42,847 |
| (0, 1, 0) | 0.70 | 0.49 | 10,682 | 48,053 |
| (0, 0, 1) | 0.76 | 0.70 | 10,220 | 49,722 |

A pure-F weighting gives an applicant ranking that is *anti-correlated* with
the baseline (τ = −0.25), because $F$ and $N$ are negatively correlated. The
choice of weights is therefore a **policy decision** with measurable effects on
risk and on who gets a loan. Equal weights are declared as a neutral reference,
not as the most stable option.

### Component ablation (`results/04_component_ablation/`): 30 cohorts, exact solutions

| setting | approved | AR_A | AR_B | \|DPD\| | exp. loss (₹) | CVaR95 (₹) | realised loss (₹) |
|---|---|---|---|---|---|---|---|
| utility only | 4.60 | 0.87 | 0.22 | 0.65 | 16,301 | 58,347 | 13,833 |
| + variance (η=1) | 4.57 | 0.81 | 0.29 | 0.52 | 13,837 | 53,467 | 13,000 |
| + fairness (λ_F=2) | 4.33 | 0.61 | 0.46 | 0.15 | 12,320 | 51,930 | 13,000 |
| P1: variance + fairness | 4.30 | 0.60 | 0.46 | 0.14 | 10,778 | 48,002 | 11,667 |
| P2: + CVaR, λ_R=0.5 | 4.20 | 0.57 | 0.47 | 0.12 | 9,148 | 44,202 | 10,333 |
| P2: + CVaR, λ_R=1 | 3.90 | 0.52 | 0.46 | 0.11 | 7,201 | 37,427 | 6,500 |
| P2: + CVaR, λ_R=2 | 1.50 | 0.18 | 0.19 | 0.02 | 1,690 | 11,260 | 3,167 |

Every penalty changes the allocation in its intended direction. With
$\lambda_R\ge2$ the CVaR term overrides utility and lending nearly stops, so
$\lambda_R$ needs to be set as an explicit risk-appetite parameter. The score
ablation (E1–E7) is in `score_components_summary.csv`. Each component on its own
selects a noticeably different set (Jaccard 0.65–0.74 vs E7).

---

## Extended Modules: Warm-Start QAOA, Qiskit Noise/ZNE, CLI & Dashboard

See [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) for the complete v1-vs-v2 conceptual audit, novelty alignment, and scaling/noise tables. Additional capabilities implemented on top of core `01`–`04`:

- **Warm-Start QAOA (`quantum.solve_warm_start_qaoa`):** Initialises from the continuous convex QP relaxation of $J_1(x)$ ($x\in[0,1]^N,\ w^\top x \le B$) with greedy fractional slack initialisation and an aligned single-qubit mixer (`R_y(\theta_i)` / `exp(+i\beta(\cos\theta_i Z_i + \sin\theta_i X_i))`). Across $N \in \{4, 6, 8, 10\}$ (10 cohorts per size), Warm-Start QAOA achieves up to **163x ground-state probability amplification** over uniform at $N=10$ (15 qubits, `P_opt_exact = 0.1634` vs `0.0009` for standard `qaoa_p1`) with `hit_opt_P1 = 0.90` vs `0.00` for `qaoa_p1`.
- **Qiskit Circuit Export, Depolarising + Readout Noise & ZNE (`qiskit_backend.py`):** Exact OpenQASM 2.0 / ASCII circuit compilation, local unitary gate folding ($c \in \{1, 3, 5\}$), exact `DensityMatrix` ($n \le 8$) + unbiased L3-chunked Kraus trajectories ($n > 8$), and Richardson ZNE reducing expectation bias by **57%–75%** across $N \in \{4, 6, 8\}$.
- **Unified Working Module & CLI (`pipeline.py`, `scripts/run_qfar.py`):** Accepts Q-FAR cohorts and legacy v1 CSVs (`Group_A_Marginalized` / `Group_B_General` without `village` or `repaid` labels):
  ```bash
  pip install -e ".[dev,full]"
  python scripts/run_qfar.py --cohort 0 --solver warm_start_qaoa_p1 --qiskit --zne
  python scripts/run_qfar.py --csv ../data/sample_applicants.csv --solver warm_start_qaoa_p1 --qiskit --zne
  ```
- **Scaling Benchmark, Figures & Interactive Dashboard:**
  ```bash
  python scripts/06_scaling_and_noise.py   # results/05_scaling_and_noise/
  python scripts/05_generate_visuals.py    # 8 PNGs in results/figures/
  python scripts/07_dashboard.py           # Dash + Plotly UI at http://127.0.0.1:8050
  ```

---

## Limitations

* **Synthetic data.** Effect sizes depend on the simulator. Real labelled data
  can be plugged in through `data.load_labelled_csv` (see the data card).
* **Fixed assumptions.** $r$, LGD, the copula correlations and the score weights
  are assumptions, not estimates. $F_i=p_i$ follows from using a single $r$ and LGD.
* **Fairness is soft in the QUBO.** Only the MILP solvers can enforce
  $\lvert d\rvert\le\epsilon$ as a hard constraint.
* **CVaR only in classical solvers.** CVaR is optimised exactly by MILP, but it
  is not encoded for the quantum solvers.
* **Small instances.** N = 8 (13 qubits) keeps statevector simulation and
  enumeration exact. Results say nothing about quantum advantage at scale, on
  noisy hardware, or with restricted qubit connectivity. Circuit depths assume
  all-to-all connectivity with no routing.
* **Budget penalty and QAOA.** The proven penalty bound makes the budget term
  dominate the energy landscape, which is known to hurt QAOA. This trade-off is
  reported, not tuned away.
* **Static allocation.** Loans are all-or-nothing, decided once per cohort.

## Reproducibility

* One seed (`seed` in the config) drives the data, the split, the cohorts
  (`seed+2/3/4` per script) and the scenarios (`seed+1000·cohort`).
* Split membership is saved to `data/processed/split.csv`, and cohort membership
  to `results/03_allocation/cohorts.csv`.
* Every solver for a cohort receives the same `Instance` object (scores, budget,
  covariance, scenarios).
* `test_reproducibility_with_fixed_seeds` checks that re-running gives identical
  data, cohorts, scenarios and QUBO matrices.

## License

MIT
