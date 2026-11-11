# Data card: synthetic microfinance applicants

**Status: SYNTHETIC.** All results in this repository come from a simulation
study. No real borrowers are involved. The results do not establish real-world
repayment-prediction performance or real-world allocation outcomes.

## Generator

`qfar.data.generate_pool(n, seed)` with defaults n = 6000 and seed = 20261110
(`configs/default.yaml`).

The repayment label is drawn from a known latent model:

$$z_i = 2.9 + 4(h_i - .75) - 3(DTI_i - .35) + 0.8\ln(\text{inc}_i/20000) - (P_i - .5) - 0.3\,L_i/\text{inc}_i + v_{\text{village}(i)} + e_i,$$
$$y_i \sim \text{Bernoulli}(\sigma(z_i)),\quad v \sim N(0, .45^2)\ \text{(unobserved village shock)},\quad e_i \sim N(0,.5^2).$$

The unobserved village shock makes defaults correlated within a village. The
allocation stage models that correlation with the copula in `risk.py`.

Realised repayment rate: Group A ≈ 0.80, Group B ≈ 0.94, overall ≈ 0.87.

## Groups

$A$ = Group A applicants, $B$ = Group B applicants (`group` = 0 or 1). These are
**synthetic groups**, not real demographic categories. Their assumed simulated
characteristics:

| | Group A | Group B |
|---|---|---|
| monthly income (INR) | U(7k, 20k) | U(18k, 45k) |
| DTI | U(.20, .65) | U(.15, .50) |
| poverty index | U(.55, .95) | U(.15, .50) |
| dependents | 2–6 | 0–4 |
| requested loan (INR) | 15k–30k | 20k–50k |
| villages | 0–5 | 6–11 |
| women-led share | 75% | 45% |
| first-time formal borrower | 65% | 30% |
| no formal account | 40% (+15% if underserved) | 12% (+15% if underserved) |
| underserved locality | villages 0–3 | villages 6–7 |

For real data, define groups from a justified, documented attribute in that
dataset. Do not assume that "rural" or "marginalised" maps to one group.

## Column roles

| role | columns |
|---|---|
| target | `repaid` |
| fairness evaluation group | `group` |
| repayment-model features | `requested_amount, monthly_income, dti_ratio, repayment_history, poverty_index, dependents` |
| Need score | `poverty_index, dependents, monthly_income` |
| Inclusion score (Option A) | `first_time_formal_borrower, no_formal_account, underserved_locality, branch_distance_km` |
| default dependence | `village` |
| protected, never a model input | `group, is_women_led` (`is_women_led` is used only by the Option-B fallback score) |

Overlaps: `poverty_index` and `dependents` are both PD features and Need inputs,
so $F$ and $N$ are correlated (Spearman ≈ −0.68 on test applicants).

## Using real labelled data

1. Prepare a CSV with one row per historical loan and an observed outcome.
2. Map its columns to the schema with
   `qfar.data.load_labelled_csv(path, column_map={...})`, which maps
   "Fully Paid"/1/"repaid" to `repaid = 1` and everything else to 0.
3. Set `data.source: csv` and `data.csv_path` in the config.
4. Leave the inclusion columns out if the dataset does not have them. The pipeline
   will then use the Social Priority Score (Option B). Never fill them in by hand.
5. Document the group definition and get ethical approval for any use of
   protected attributes.
