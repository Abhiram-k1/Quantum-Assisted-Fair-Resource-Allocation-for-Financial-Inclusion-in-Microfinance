"""Stage 1: supervised, calibrated repayment model  p_i = P(repaid_i = 1 | features_i).

Protocol (fix #1):
  1. model x calibration x feature-set comparison by stratified k-fold CV on TRAIN only;
  2. the top-``shortlist`` configs are fitted on TRAIN and compared on VALIDATION;
  3. the selected config is refitted on TRAIN + VALIDATION;
  4. it is evaluated ONCE on the held-out TEST split (overall and per group).

Protected attributes (``group``, ``is_women_led``) are never model inputs.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import PD_FEATURES, PROTECTED_COLUMNS

FEATURE_SETS = {
    "full": PD_FEATURES,
    "no_history": [f for f in PD_FEATURES if f != "repayment_history"],
    "core_credit": ["requested_amount", "monthly_income", "dti_ratio", "repayment_history"],
}
assert not any(c in PROTECTED_COLUMNS for cols in FEATURE_SETS.values() for c in cols)


def build_model(kind: str, calib: str, seed: int = 0):
    if kind == "logreg":
        base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    elif kind == "rf":
        base = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                                      random_state=seed, n_jobs=-1)
    elif kind == "gbm":
        base = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=.05,
                                              random_state=seed)
    else:
        raise ValueError(kind)
    return base if calib == "none" else CalibratedClassifierCV(base, method=calib, cv=3)


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(n=int(len(y)), base_rate=float(np.mean(y)), roc_auc=float(roc_auc_score(y, p)),
                pr_auc_default=float(average_precision_score(1 - np.asarray(y), 1 - p)),
                brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p)),
                ece=ece(y, p), mean_pred=float(p.mean()))


def cv_ablation(train: pd.DataFrame, kinds, calibs, seed: int, folds: int) -> pd.DataFrame:
    y = train["repaid"].values
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    rows = []
    for kind, calib, (fs, cols) in product(kinds, calibs, FEATURE_SETS.items()):
        p = cross_val_predict(build_model(kind, calib, seed), train[cols].values, y, cv=skf,
                              method="predict_proba")[:, 1]
        rows.append(dict(model=kind, calibration=calib, features=fs, **metrics(y, p)))
    return (pd.DataFrame(rows).sort_values(["brier", "roc_auc"], ascending=[True, False])
            .reset_index(drop=True))


def select_on_validation(abl: pd.DataFrame, train, val, k: int, seed: int) -> pd.DataFrame:
    rows = []
    for _, r in abl.head(k).iterrows():
        cols = FEATURE_SETS[r.features]
        m = build_model(r.model, r.calibration, seed).fit(train[cols].values, train.repaid.values)
        rows.append(dict(model=r.model, calibration=r.calibration, features=r.features,
                         **metrics(val.repaid.values, m.predict_proba(val[cols].values)[:, 1])))
    return pd.DataFrame(rows).sort_values(["brier", "roc_auc"], ascending=[True, False]).reset_index(drop=True)


def evaluate_test(model, test: pd.DataFrame, cols) -> dict:
    p = model.predict_proba(test[cols].values)[:, 1]
    y = test["repaid"].values
    out = {"overall": metrics(y, p)}
    for g, name in ((0, "group_A"), (1, "group_B")):
        m = test.group.values == g
        out[name] = metrics(y[m], p[m])
    return out


def reliability_table(y, p, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return pd.DataFrame([dict(bin_lo=edges[b], bin_hi=edges[b + 1], n=int((idx == b).sum()),
                              mean_pred=float(p[idx == b].mean()), observed=float(y[idx == b].mean()))
                         for b in range(bins) if (idx == b).any()])
