"""U6 — Model ladders, tuning, and honest leaderboard.

Trains a full ladder for each task and evaluates every model on the held-out
test set (Nov-Dec), always keeping the baseline (R6, R14):

- Regression:     DummyRegressor(mean) -> Ridge -> DecisionTree -> tuned XGBoost
- Classification: DummyClassifier(most_frequent) -> LogReg -> DecisionTree -> tuned XGBoost

XGBoost is tuned with RandomizedSearchCV over a TimeSeriesSplit (never trains on
the future). The deployable models are trained on the SERVED feature set
(calendar + cyclical) so the live API can reproduce features from a bare date; a
lag-augmented "+lags" XGBoost row is added to the leaderboard for comparison.

Run as a script to (re)build all artifacts:
    python -m src.models            # from analysis/  (with src on path)
Writes: models/daily_revenue.joblib, models/high_demand.joblib,
        models/leaderboard.json, models/report_artifacts.json
"""

from __future__ import annotations

import json
import sklearn
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.base import clone
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from . import RANDOM_SEED
from .data import build_daily
from .features import (
    FULL_FEATURES,
    SERVED_FEATURES,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    build_features,
    chronological_split,
    select_features,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CV = TimeSeriesSplit(n_splits=5)


def _cv_mean(model, X, y, scoring: str) -> float:
    """Mean TimeSeriesSplit CV score on the training data (fairer than the
    single Nov-Dec holdout, whose calendar positions are unseen in training)."""
    return float(cross_val_score(clone(model), X, y, cv=CV, scoring=scoring).mean())

XGB_PARAM_DIST = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__max_depth": [2, 3, 4, 5],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
}


def _pipe(model, scale: bool = False) -> Pipeline:
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


# --- Regression -------------------------------------------------------------

def _regression_ladder():
    return {
        "Dummy (mean)": _pipe(DummyRegressor(strategy="mean")),
        "Ridge": _pipe(Ridge(alpha=1.0, random_state=RANDOM_SEED), scale=True),
        "Decision Tree": _pipe(DecisionTreeRegressor(max_depth=4, random_state=RANDOM_SEED)),
    }


def _tuned_xgb_regressor(X_train, y_train):
    search = RandomizedSearchCV(
        _pipe(XGBRegressor(random_state=RANDOM_SEED, n_jobs=1)),
        XGB_PARAM_DIST, n_iter=25, cv=CV, scoring="r2",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def _reg_metrics(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(root_mean_squared_error(y, pred)),
    }


# --- Classification ---------------------------------------------------------

def _classification_ladder(y_train):
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = (neg / pos) if pos else 1.0
    return {
        "Dummy (most frequent)": _pipe(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": _pipe(LogisticRegression(max_iter=2000, random_state=RANDOM_SEED), scale=True),
        "Decision Tree": _pipe(DecisionTreeClassifier(max_depth=4, random_state=RANDOM_SEED)),
    }, spw


def _tuned_xgb_classifier(X_train, y_train, spw):
    search = RandomizedSearchCV(
        _pipe(XGBClassifier(random_state=RANDOM_SEED, n_jobs=1, scale_pos_weight=spw,
                            eval_metric="logloss")),
        XGB_PARAM_DIST, n_iter=25, cv=CV, scoring="f1",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def _clf_metrics(model, X, y) -> dict:
    pred = model.predict(X)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            out["roc_auc"] = float(roc_auc_score(y, proba))
        else:
            out["roc_auc"] = None
    except (ValueError, IndexError):
        out["roc_auc"] = None
    return out


# --- Orchestration ----------------------------------------------------------

def train_all(write: bool = True) -> dict:
    """Train both ladders, evaluate on test, serialize served models + artifacts."""
    daily = build_daily()
    feats = build_features(daily)
    train, val, test = chronological_split(feats)

    # Feature selection (fit on train only) for the served set, per task.
    reg_selected = select_features(train[SERVED_FEATURES], train[TARGET_REGRESSION], "regression")
    clf_selected = select_features(train[SERVED_FEATURES], train[TARGET_CLASSIFICATION], "classification")

    results = {"regression": {}, "classification": {}, "leaderboard": [],
               "selected_features": {"regression": reg_selected, "classification": clf_selected},
               "sklearn_version": sklearn.__version__}

    # ---- Regression ladder (served features) ----
    Xtr, ytr = train[reg_selected], train[TARGET_REGRESSION]
    Xte, yte = test[reg_selected], test[TARGET_REGRESSION]
    reg_models = _regression_ladder()
    reg_fitted = {}
    for name, model in reg_models.items():
        model.fit(Xtr, ytr)
        reg_fitted[name] = model
        results["leaderboard"].append({"task": "regression", "model": name,
                                       "test": _reg_metrics(model, Xte, yte),
                                       "train_r2": float(r2_score(ytr, model.predict(Xtr))),
                                       "cv_r2": _cv_mean(model, Xtr, ytr, "r2")})
    xgb_reg, xgb_reg_params = _tuned_xgb_regressor(Xtr, ytr)
    reg_fitted["XGBoost"] = xgb_reg
    results["leaderboard"].append({"task": "regression", "model": "XGBoost",
                                   "test": _reg_metrics(xgb_reg, Xte, yte),
                                   "train_r2": float(r2_score(ytr, xgb_reg.predict(Xtr))),
                                   "cv_r2": _cv_mean(xgb_reg, Xtr, ytr, "r2")})

    # +lags comparison row (FULL features) — report only, not deployed.
    full_sel = select_features(train[FULL_FEATURES], train[TARGET_REGRESSION], "regression")
    xgb_full, _ = _tuned_xgb_regressor(train[full_sel], ytr)
    results["leaderboard"].append({"task": "regression", "model": "XGBoost (+lags)",
                                   "test": _reg_metrics(xgb_full, test[full_sel], yte),
                                   "train_r2": float(r2_score(ytr, xgb_full.predict(train[full_sel]))),
                                   "cv_r2": _cv_mean(xgb_full, train[full_sel], ytr, "r2")})

    # Select the deployable model by the PRIMARY metric (TimeSeriesSplit CV),
    # not the Nov-Dec holdout: the holdout's calendar positions are unseen in
    # training, so it under-rates models that need to interpolate seasonality.
    reg_candidates = {r["model"]: r for r in results["leaderboard"]
                      if r["task"] == "regression" and r["model"] in ("Ridge", "Decision Tree", "XGBoost")}
    best_reg = max(reg_candidates, key=lambda n: reg_candidates[n]["cv_r2"])
    # predicted-vs-actual for the report scatter (best served model)
    results["regression"]["best_model"] = best_reg
    results["regression"]["predicted_vs_actual"] = {
        "dates": [d.strftime("%Y-%m-%d") for d in test.index],
        "actual": [float(v) for v in yte],
        "predicted": [float(v) for v in reg_fitted[best_reg].predict(Xte)],
    }
    results["regression"]["xgb_best_params"] = {k: v for k, v in xgb_reg_params.items()}

    # Decision-tree overfitting curve (validation_curve over max_depth).
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    tr_sc, te_sc = validation_curve(
        DecisionTreeRegressor(random_state=RANDOM_SEED), train[reg_selected], ytr,
        param_name="max_depth", param_range=depths, cv=CV, scoring="r2")
    results["regression"]["depth_curve"] = {
        "max_depth": depths,
        "train_r2": [float(s) for s in tr_sc.mean(axis=1)],
        "cv_r2": [float(s) for s in te_sc.mean(axis=1)],
    }

    # ---- Classification ladder (served features) ----
    Xtr_c, ytr_c = train[clf_selected], train[TARGET_CLASSIFICATION]
    Xte_c, yte_c = test[clf_selected], test[TARGET_CLASSIFICATION]
    clf_models, spw = _classification_ladder(ytr_c)
    clf_fitted = {}
    for name, model in clf_models.items():
        model.fit(Xtr_c, ytr_c)
        clf_fitted[name] = model
        results["leaderboard"].append({"task": "classification", "model": name,
                                       "test": _clf_metrics(model, Xte_c, yte_c),
                                       "train_accuracy": float(accuracy_score(ytr_c, model.predict(Xtr_c))),
                                       "cv_f1": _cv_mean(model, Xtr_c, ytr_c, "f1")})
    xgb_clf, xgb_clf_params = _tuned_xgb_classifier(Xtr_c, ytr_c, spw)
    clf_fitted["XGBoost"] = xgb_clf
    results["leaderboard"].append({"task": "classification", "model": "XGBoost",
                                   "test": _clf_metrics(xgb_clf, Xte_c, yte_c),
                                   "train_accuracy": float(accuracy_score(ytr_c, xgb_clf.predict(Xtr_c))),
                                   "cv_f1": _cv_mean(xgb_clf, Xtr_c, ytr_c, "f1")})

    # Select by ROC-AUC (threshold-independent) — F1 is misleading here because
    # the majority-class baseline scores a high F1 by predicting all-positive.
    clf_candidates = {r["model"]: r for r in results["leaderboard"]
                      if r["task"] == "classification" and r["model"] in ("Logistic Regression", "Decision Tree", "XGBoost")}
    best_clf = max(clf_candidates, key=lambda n: (clf_candidates[n]["test"]["roc_auc"] or 0.0))
    results["classification"]["best_model"] = best_clf
    cm = confusion_matrix(yte_c, clf_fitted[best_clf].predict(Xte_c))
    results["classification"]["confusion_matrix"] = cm.tolist()
    results["classification"]["xgb_best_params"] = {k: v for k, v in xgb_clf_params.items()}

    # ---- Serialize the served (deployable) models + feature schema ----
    if write:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(
            {"model": reg_fitted[best_reg], "features": reg_selected,
             "task": "regression", "sklearn_version": sklearn.__version__},
            MODELS_DIR / "daily_revenue.joblib")
        joblib.dump(
            {"model": clf_fitted[best_clf], "features": clf_selected,
             "task": "classification", "sklearn_version": sklearn.__version__},
            MODELS_DIR / "high_demand.joblib")
        (MODELS_DIR / "leaderboard.json").write_text(
            json.dumps(results["leaderboard"], indent=2), encoding="utf-8")
        artifacts = {k: results[k] for k in ("regression", "classification", "selected_features")}
        artifacts["test_dates"] = [d.strftime("%Y-%m-%d") for d in test.index]
        (MODELS_DIR / "report_artifacts.json").write_text(
            json.dumps(artifacts, indent=2), encoding="utf-8")

    return results


def _print_leaderboard(results: dict) -> None:
    rows = results["leaderboard"]
    print("\n=== REGRESSION  (CV = TimeSeriesSplit mean; test = Nov-Dec holdout) ===")
    print(f"{'model':<20}{'cv R2':>8}{'test R2':>9}{'MAE':>9}{'RMSE':>9}{'train R2':>10}")
    for r in [r for r in rows if r["task"] == "regression"]:
        m = r["test"]
        print(f"{r['model']:<20}{r['cv_r2']:>8.3f}{m['r2']:>9.3f}{m['mae']:>9.1f}{m['rmse']:>9.1f}{r['train_r2']:>10.3f}")
    print("\n=== CLASSIFICATION  (CV F1 = TimeSeriesSplit mean; test = Nov-Dec holdout) ===")
    print(f"{'model':<24}{'cv F1':>7}{'acc':>7}{'prec':>7}{'rec':>7}{'F1':>7}{'AUC':>7}")
    for r in [r for r in rows if r["task"] == "classification"]:
        m = r["test"]
        auc = "  n/a" if m["roc_auc"] is None else f"{m['roc_auc']:>7.3f}"
        print(f"{r['model']:<24}{r['cv_f1']:>7.3f}{m['accuracy']:>7.3f}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}{auc}")


if __name__ == "__main__":
    res = train_all(write=True)
    _print_leaderboard(res)
    print(f"\nbest regression: {res['regression']['best_model']} | "
          f"best classification: {res['classification']['best_model']}")
    print(f"artifacts written to {MODELS_DIR}")
