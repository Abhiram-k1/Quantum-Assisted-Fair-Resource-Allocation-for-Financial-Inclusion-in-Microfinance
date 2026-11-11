"""Stage 1 -- repayment model on SYNTHETIC labelled data.

train/validation/test split -> CV on TRAIN -> shortlist checked on VALIDATION ->
refit on TRAIN+VALIDATION -> ONE evaluation on TEST.  Test applicants (never seen
by the model) are scored and become the applicant pool for the allocation experiments.

    python scripts/01_train_pd.py [--config configs/default.yaml] [--quick]
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from qfar.config import load_config, write_manifest  # noqa: E402
from qfar.data import GROUP_LABELS, PD_FEATURES, load_pool, split_pool  # noqa: E402
from qfar.pd_model import (FEATURE_SETS, build_model, cv_ablation, evaluate_test,  # noqa: E402
                           reliability_table, select_on_validation)
from qfar.scoring import compute_scores  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--quick", action="store_true", help="logreg + gbm only")
args = ap.parse_args()
cfg = load_config(args.config)
OUT = ROOT / "results" / "01_pd_model"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "processed"; DATA.mkdir(parents=True, exist_ok=True)
seed, pc = cfg["seed"], cfg["pd_model"]

df = load_pool(cfg)
split = split_pool(df, cfg)
pd.concat([d[["applicant_id"]].assign(split=k) for k, d in split.items()]).to_csv(DATA / "split.csv", index=False)
train, val, test = split["train"], split["validation"], split["test"]
print(f"data source = {cfg['data']['source']} | train {len(train)}  validation {len(val)}  test {len(test)}")

summary = df.groupby("group").agg(n=("repaid", "size"), repaid_rate=("repaid", "mean"),
                                   income=("monthly_income", "mean"), dependents=("dependents", "mean"),
                                   poverty=("poverty_index", "mean"), amount=("requested_amount", "mean"),
                                   first_time=("first_time_formal_borrower", "mean"),
                                   underserved=("underserved_locality", "mean"))
summary.index = [GROUP_LABELS[g] for g in summary.index]
summary.to_csv(OUT / "group_summary.csv"); print(summary.round(3).to_string())

kinds = ("logreg", "gbm") if args.quick else tuple(pc["models"])
abl = cv_ablation(train, kinds, pc["calibrations"], seed, pc["cv_folds"])
abl.to_csv(OUT / "cv_ablation_train.csv", index=False)
print("\nCV on TRAIN (top 5 by Brier):\n", abl.head(5).round(4).to_string())

vsel = select_on_validation(abl, train, val, pc["shortlist"], seed)
vsel.to_csv(OUT / "validation_selection.csv", index=False)
print("\nShortlist on VALIDATION:\n", vsel.round(4).to_string())

best = vsel.iloc[0]; cols = FEATURE_SETS[best.features]
trval = pd.concat([train, val])
model = build_model(best.model, best.calibration, seed).fit(trval[cols].values, trval.repaid.values)
res = evaluate_test(model, test, cols)
res["selected"] = dict(model=best.model, calibration=best.calibration, features=best.features, columns=cols)
(OUT / "test_metrics.json").write_text(json.dumps(res, indent=2))
print("\nHELD-OUT TEST:\n", json.dumps({k: res[k] for k in ("overall", "group_A", "group_B")}, indent=1))

test = test.copy(); test["p_repay"] = model.predict_proba(test[cols].values)[:, 1]
reliability_table(test.repaid.values, test.p_repay.values).to_csv(OUT / "test_reliability.csv", index=False)
test.to_csv(DATA / "test_scored.csv", index=False)
joblib.dump(dict(model=model, cols=cols), OUT / "pd_model.joblib")

# Correlation between decision dimensions and model features (fix #5): the three scores
# are distinct dimensions, but they are NOT statistically independent.
sc = compute_scores(test, test.p_repay.values, cfg)
corr = pd.concat([sc[["F", "N", "S"]], test[PD_FEATURES]], axis=1).corr(method="spearman")
corr.to_csv(OUT / "score_feature_spearman.csv")
print("\nSpearman correlation of scores (test applicants):\n", corr.loc[["F", "N", "S"], ["F", "N", "S"]].round(3))
print("score vs features:\n", corr.loc[["F", "N", "S"], PD_FEATURES].round(2).to_string())

write_manifest(OUT, cfg, script="01_train_pd.py", selected=res["selected"],
               split_sizes={k: len(v) for k, v in split.items()}, social_score=sc.attrs["social_kind"])
