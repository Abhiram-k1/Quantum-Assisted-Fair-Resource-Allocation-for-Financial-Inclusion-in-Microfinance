# Mathematical formulation

This document is the reference for every equation in the code. Each section names
the function that implements it, and the test that checks it.

Notation: applicants $i = 1..N$ in a cohort, decision $x_i \in \{0,1\}$ (1 = approve
the requested loan in full).

---

## 1. Repayment target and model (`pd_model.py`)

$$
y_i=\begin{cases}1 & \text{applicant } i \text{ repaid}\\ 0 & \text{applicant } i \text{ defaulted}\end{cases}
\qquad p_i = \widehat{\Pr}(y_i = 1 \mid \text{features}_i)
$$

Features: `requested_amount, monthly_income, dti_ratio, repayment_history,
poverty_index, dependents`. The model never sees `group`, `is_women_led` or the
inclusion indicators (`test_protected_attributes_are_not_model_features`).

Protocol: stratified 60/20/20 train/validation/test split → k-fold CV on train only
→ the top configurations are compared on validation → the selected configuration
is refitted on train+validation → one evaluation on test. Allocation cohorts are
drawn only from **test** applicants.

**The data are synthetic** (see `DATA_CARD.md`), so these metrics describe how
well the model recovers a simulated mechanism. They are not real-world validation.

---

## 2. Scores, all with fixed normalisation (`scoring.py`)

### 2.1 Financial viability $F_i$

$$
\mu_i = p_i r - (1-p_i)\,LGD, \qquad -LGD \le \mu_i \le r,
\qquad F_i = \frac{\mu_i + LGD}{r + LGD} \in [0,1].
$$

With $r = 0.2$ and $LGD = 1$: $\mu_i = 1.2p_i - 1$ and $F_i = (\mu_i+1)/1.2$.

**Consequence.** With one $r$ and one $LGD$ for everybody, $F_i = p_i$ exactly
(`test_score_ranges_and_F_definition`). $F$ is a *normalised expected-return
score under the assumed $r$ and LGD*. It is not realised profit. It differs from
$p_i$ only if $r_i$ or $LGD_i$ vary by applicant, which `financial_score`
supports by normalising with $r_{\max}$ and $LGD_{\max}$.

### 2.2 Need $N_i$

$$
PCI_i = \frac{\text{Monthly income}_i}{\text{Dependents}_i + 1},\quad
I_i = \operatorname{clip}\!\left(\frac{PCI_i - PCI_{\min}}{PCI_{\max}-PCI_{\min}},0,1\right),\quad
D_i = \operatorname{clip}\!\left(\frac{\text{Dependents}_i}{D_{\max}},0,1\right)
$$

$$
\boxed{N_i = 0.5\,P_i + 0.3\,D_i + 0.2\,(1 - I_i)}
$$

$PCI_{\min} = 1{,}000$, $PCI_{\max} = 25{,}000$ INR/person/month and $D_{\max} = 6$
(config `normalisation`). The 0.5/0.3/0.2 weights are provisional policy choices.

### 2.3 Social score $S_i$: Option A (Inclusion Score) is used

$$
S_i = \alpha_1 E_{1i} + \alpha_2 E_{2i} + \alpha_3 E_{3i} + \alpha_4 E_{4i},\qquad \textstyle\sum_j \alpha_j = 1,\ \alpha_j = 0.25
$$

| | indicator | column |
|---|---|---|
| $E_1$ | first-time formal borrower | `first_time_formal_borrower` |
| $E_2$ | no formal bank account (lack of formal financial access) | `no_formal_account` |
| $E_3$ | underserved locality | `underserved_locality` |
| $E_4$ | $\operatorname{clip}(\text{branch distance}/20\text{ km},0,1)$ | `branch_distance_km` |

Option A is used only if all four columns exist. In the synthetic data they are
*simulated* variables, declared as such. They are not invented for real
applicants. If a real dataset lacks them, `social_score` falls back to
**Option B, the Social Priority Score** $S_i = 0.6\,W_i + 0.4\,(1 - L_i/L_{\max})$,
which is a policy priority rule and not a measure of structural exclusion
(`test_social_score_falls_back_to_priority_without_indicators`).

### 2.4 Composite utility

$$
\boxed{U_i = \omega_F F_i + \omega_N N_i + \omega_S S_i},\qquad \omega \ge 0,\ \textstyle\sum \omega = 1
$$

**Declared baseline: $\omega_F = \omega_N = \omega_S = 1/3$**, used in every
experiment unless the experiment itself varies the weights. We chose it as a
neutral starting point, not because it is "stable". Experiment 02 shows how the
allocation changes as the weights move.

The three scores are distinct decision dimensions but **not statistically
independent**. On test applicants (Spearman): $\rho(F,N) \approx -0.68$,
$\rho(F,S) \approx -0.34$, $\rho(N,S) \approx 0.41$. The PD model uses `poverty_index`
and `dependents`, which also enter $N$ (see
`results/01_pd_model/score_feature_spearman.csv`).

**Independence from the cohort.** No score uses cohort statistics. An applicant's
$F, N, S, U$ are identical in every cohort (`test_scores_are_independent_of_cohort_composition`).

---

## 3. Constraints and penalties

### 3.1 Budget (`qubo.py`)

$$\sum_i L_i x_i \le B.$$

Money is converted to integers in units of $u = 5{,}000$ INR. Loans are rounded
**up**, $w_i = \lceil L_i/u \rceil$, and the budget is rounded **down**,
$B_u = \lfloor B/u \rfloor$. Every integer-feasible $x$ is therefore feasible in rupees
(`test_budget_units_rounding_is_conservative`). The cohort budget is
$B_u = \lfloor 0.5 \sum_i w_i \rfloor$.

Inequality → equality with a slack variable:
$$\sum_i w_i x_i + s = B_u,\quad s = \sum_{k=0}^{K-1} c_k s_k,\quad
c = [1,2,\dots,2^{K-2},\ B_u-(2^{K-1}-1)],\ K = \lceil \log_2(B_u+1)\rceil.$$
These coefficients represent every integer $0..B_u$ and nothing larger
(`test_slack_represents_exactly_0_to_B`, $B_u = 1..299$).

$$P_B = \lambda_B\Big(\sum_i w_i x_i + \sum_k c_k s_k - B_u\Big)^2.$$

### 3.2 The penalty coefficient

$$\boxed{\lambda_B > \sum_i \max(U_i, 0)}\qquad(\text{code: } \lambda_B = 1.1\sum_i\max(U_i,0) + 10^{-3}).$$

*Proof.* Let $r = w^\top x + c^\top s - B_u$, which is an integer. Suppose $\Gamma \succeq 0$
and $\eta,\lambda_F \ge 0$. Then the risk and fairness terms are $\ge 0$, so any
assignment with $r \ne 0$ has
$E \ge -\sum_i \max(U_i,0) + \lambda_B \cdot 1 > 0$. The empty allocation with
$s = B_u$ has $E = 0$. So every ground state has $r = 0$, and therefore $x$ is
budget-feasible. Among states with $r = 0$, $E(x,s) = J_1(x)$, so the ground state
minimises $J_1$ over feasible $x$. ∎

Tests check the separation directly on every state
(`test_penalty_bound_separates_infeasible_states`) and by sampling up to $N = 80$
(`test_penalty_bound_large_instances_by_sampling`). They also check 5 budget
fractions (`test_feasibility_across_budgets`) and that a too-small $\lambda_B$
really does produce infeasible ground states (`test_too_small_penalty_can_break_feasibility`).
The constructor rejects a non-PSD $\Gamma$.

The ZIP's bound $1.5\max U + \eta\max_i\sum_j|\Gamma_{ij}| + \lambda_F$ had no such
guarantee and is no longer used.

### 3.3 Fairness: Approach A, a soft penalty

$$d(x) = \frac{\sum_{i\in A}x_i}{n_A} - \frac{\sum_{i\in B}x_i}{n_B} = \sigma^\top x,\qquad
\sigma_i = \begin{cases}1/n_A & i\in A\\ -1/n_B & i \in B\end{cases},\qquad
\boxed{P_F = \lambda_F\, d(x)^2}$$

This is **soft**. Raising $\lambda_F$ makes approval-rate parity more important
relative to utility, but it does **not** guarantee $|d(x)| \le \epsilon$. A hard
constraint $|d(x)| \le \epsilon$ is available only in the MILP solvers
(`solve_milp(..., fair_eps=ε)`), where it is a real linear constraint
(`test_hard_fair_constraint_is_actually_enforced_in_p1_milp`).

Reported fairness metrics:
$AR_A, AR_B$, signed $DPD = AR_A - AR_B$, $|DPD|$, and **directional** ratios
$DIR_{A/B} = AR_A/AR_B$ and $DIR_{B/A} = AR_B/AR_A$ (`metrics.fairness_metrics`).

---

## 4. Risk (`risk.py`)

### 4.1 One model of correlated defaults

Gaussian one-factor copula with a global factor and a village factor:
$$A_i = \sqrt{\rho_g}\,Z + \sqrt{\rho_v}\,Z_{v(i)} + \sqrt{1-\rho_g-\rho_v}\,\varepsilon_i,\qquad
D_i = \mathbb 1[A_i < \Phi^{-1}(1-p_i)]$$
with $\rho_g = 0.05$ and $\rho_v = 0.25$. **Both** risk measures below come from this model.

### 4.2 Variance proxy

$$\Sigma_{ij} = \operatorname{Cov}(D_i,D_j) = \Phi_2(\tau_i,\tau_j; R_{ij}) - q_iq_j,\qquad
a_i = LGD\cdot L_i/L_{\text{ref}},\qquad \Gamma = \operatorname{diag}(a)\,\Sigma\,\operatorname{diag}(a)$$
$$R_{\text{variance}}(x) = x^\top\Gamma x = \operatorname{Var}[\mathcal L(x)]/L_{\text{ref}}^2.$$
$L_{\text{ref}} = 30{,}000$ INR is a fixed scale, not a cohort statistic. Tests:
$\Sigma$ symmetric PSD with the right diagonal, it matches simulated covariances,
and $x^\top\Gamma x$ equals the variance of simulated losses.

**This is variance. It is not CVaR.**

### 4.3 Financial CVaR

$$\mathcal L_s(x) = \sum_i L_i\,LGD\,D_{si}\,x_i,\qquad
\boxed{CVaR_\alpha(\mathcal L) = \min_t\Big[t + \frac{1}{(1-\alpha)S}\sum_{s=1}^S\max(\mathcal L_s - t, 0)\Big]}$$
with $\alpha = 0.95$ and $S = 2000$ scenarios per cohort (fixed seed per cohort).
The minimum is attained at the empirical $\alpha$-quantile, and `risk.cvar` uses
this. It is checked against brute-force minimisation over $t$, simple closed
cases, and sub-additivity.

| concept | meaning | where |
|---|---|---|
| Variance | spread of portfolio loss | $\eta\,x^\top\Gamma x$ in P1 and P2 |
| Financial CVaR | mean loss in the worst $(1-\alpha)$ tail of loss scenarios (**upper** tail of losses) | objective of P2; metric `cvar_inr` for every solver |
| Quantum CVaR | mean of the lowest $\alpha_q$ fraction of measured QUBO **energies** (lower tail) | CVaR-QAOA training objective only |

---

## 5. The two optimisation problems

### P1: QUBO-compatible, variance-aware, soft fairness

$$\min_{x}\ J_1(x) = -\sum_i U_ix_i + \eta\,x^\top\Gamma x + \lambda_F\,d(x)^2\quad\text{s.t.}\ \sum_i w_ix_i \le B_u$$

QUBO used by the exact-QUBO, SA, QAOA, CVaR-QAOA and VQE solvers:

$$\boxed{E(x,s) = -\sum_iU_ix_i + \eta\,x^\top\Gamma x + \lambda_F\,d(x)^2 + \lambda_B\Big(\sum_iw_ix_i + \sum_kc_ks_k - B_u\Big)^2}$$

**Do not call P1 "CVaR optimisation".** It contains no financial CVaR term.

### P2: tail-risk-aware (financial CVaR inside the objective)

$$\min_x\ J_2(x) = J_1(x) + \lambda_R\,CVaR_\alpha(\mathcal L(x)/L_{\text{ref}})\quad\text{s.t.}\ \sum_iw_ix_i\le B_u$$

**Encoding actually implemented and tested** (`solve_milp(q, tail)`), as an exact MILP:

* each product $x_ix_j$ becomes $y_{ij}$ with $y_{ij}\le x_i$, $y_{ij}\le x_j$,
  $y_{ij}\ge x_i+x_j-1$, $y_{ij}\ge0$, which is exact for binary $x$;
* CVaR uses the Rockafellar–Uryasev form: free $t$, $u_s \ge 0$,
  $u_s \ge \mathcal L_s(x) - t$, and objective term
  $\lambda_R\,(t + \sum_s u_s/((1-\alpha)S))$.

The MILP optimum equals the enumeration optimum of $J_2$ (`test_cvar_milp_equals_enumeration_P2`).

**Not implemented:** a QUBO/Ising encoding of the CVaR term. An exact one would
need $S$ continuous auxiliaries $u_s$, each with its own slack register, so
thousands of extra qubits. QAOA/CVaR-QAOA/VQE therefore solve **P1 only**. This
project makes no claim that the quantum solvers optimise financial CVaR.

Conceptual target with a hard fairness constraint (available via
`solve_milp(q, tail, fair_eps=ε)`):
$$\max_x\Big[\sum_iU_ix_i - \eta R_{\text{variance}}(x) - \lambda_RR_{\text{CVaR}}(x)\Big]\ \text{s.t.}\ \sum_iL_ix_i\le B,\ |d(x)|\le\epsilon,\ x\in\{0,1\}^N.$$

---

## 6. Solvers and what each one solves (`solvers.py`, `quantum.py`)

| label | problem | exact? |
|---|---|---|
| `exact_enumeration_P1` / `_P2` | P1 / P2 | yes (brute force, $N \lesssim 20$) |
| `exact_qubo_ground` | penalised QUBO of P1 (checks the encoding) | yes |
| `milp_linearised` | P1, same objective as enumeration | yes |
| `milp_linearised_cvar` | P2 | yes |
| `milp_hard_fair_utility` | **a different problem**: $\max U^\top x$ s.t. budget, $\lvert d\rvert\le\epsilon$, no risk term | yes for its own problem |
| `greedy_utility_per_cost` | **Greedy Utility-per-Cost Baseline**: sorts by $U_i/w_i$, ignores risk and fairness | heuristic |
| `simulated_annealing` | QUBO of P1 | heuristic |
| `qaoa_p{1,2,3}` | QUBO of P1, minimises $\langle H\rangle$ | variational |
| `cvar_qaoa_p1_a0.25` | QUBO of P1, minimises the lower 25% tail of energies | variational |
| `vqe_L2` | QUBO of P1, RY + CZ-ring ansatz | variational |

Every allocation from every solver is *evaluated* with the same $J_1$, $J_2$ and
metrics. The hard-fair MILP's $J_1$ is reported, but it did not optimise $J_1$, so
read its gap as the cost of the different policy, not as solver quality.

### 6.1 QUBO → Ising

$y_i = (1-z_i)/2$, $E(y) = y^\top My + c$ with $M$ symmetric:
$$H(z) = C + \sum_ih_iz_i + \sum_{i<j}J_{ij}z_iz_j,\qquad h = -\tfrac12 M\mathbf 1,\qquad J_{ij} = \tfrac12M_{ij}.$$
`test_ising_energy_equals_qubo_energy` checks equality on 300 random bitstrings
for $N \in \{3,5,8,12,20\}$ (up to 26 qubits).

### 6.2 Quantum simulation details

* **Ideal statevector simulation**: exact, noise-free, no hardware. 1024 shots
  are sampled from the exact output distribution.
* The QAOA phase separator uses $H/\lVert H\rVert_1$ with
  $\lVert H\rVert_1 = \sum|h_i| + \sum|J_{ij}|$.
* Optimiser: COBYLA, 4 restarts, at most 250 (QAOA) or 400 (VQE) iterations per
  restart. Depth $p+1$ starts from the INTERP extrapolation of the depth-$p$ optimum.
* **Selection rule:** the reported allocation is the **lowest-energy sampled
  bitstring**, not the most frequent one. The most frequent one is reported
  separately (`most_frequent_feasible`, `most_frequent_J1`). If a sample breaks
  the budget, it is repaired by dropping the lowest $U_i/w_i$ applicant, and this
  is flagged (`raw_feasible`, `repaired`).
* Reported: qubits, circuit depth (ASAP schedule, all-to-all connectivity,
  RZZ = CNOT–RZ–CNOT, no routing), two-qubit gate count, shots, function
  evaluations, $P_{\text{opt}}$ and $P_{\text{feasible}}$ (exact from the statevector
  and empirical from shots), each with its uniform-random baseline.

$$P_{\text{opt}} = \frac{\#\text{optimal sampled allocations}}{\#\text{samples}},\qquad
P_{\text{feasible}} = \frac{\#\text{budget-feasible sampled allocations}}{\#\text{samples}}.$$
