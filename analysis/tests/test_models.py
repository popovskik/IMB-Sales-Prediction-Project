"""Tests for U6 model ladders + leaderboard (analysis/src/models.py).

These assert on the *shipped artifacts* (models/leaderboard.json and the two
.joblib files) produced by `python -m src.models`, so they validate exactly what
the API and dashboard consume — and keep the suite fast (no retraining per run).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import build_daily
from src.features import (
    SERVED_FEATURES,
    TARGET_REGRESSION,
    build_features,
    chronological_split,
    make_served_features,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "leaderboard.json").exists(),
    reason="run `python -m src.models` first to build artifacts",
)


@pytest.fixture(scope="module")
def leaderboard():
    return json.loads((MODELS_DIR / "leaderboard.json").read_text(encoding="utf-8"))


def _rows(leaderboard, task):
    return [r for r in leaderboard if r["task"] == task]


# --- Honest leaderboard structure ------------------------------------------

def test_baseline_present_for_both_tasks(leaderboard):
    reg_models = {r["model"] for r in _rows(leaderboard, "regression")}
    clf_models = {r["model"] for r in _rows(leaderboard, "classification")}
    assert "Dummy (mean)" in reg_models
    assert "Dummy (most frequent)" in clf_models


def test_full_ladder_present(leaderboard):
    reg_models = {r["model"] for r in _rows(leaderboard, "regression")}
    clf_models = {r["model"] for r in _rows(leaderboard, "classification")}
    assert {"Ridge", "Decision Tree", "XGBoost"}.issubset(reg_models)
    assert {"Logistic Regression", "Decision Tree", "XGBoost"}.issubset(clf_models)


def test_regression_rows_report_test_and_cv_metrics(leaderboard):
    for r in _rows(leaderboard, "regression"):
        assert set(r["test"]) == {"r2", "mae", "rmse"}
        assert "cv_r2" in r  # TimeSeriesSplit CV reported alongside the holdout


def test_classification_rows_report_test_and_cv_metrics(leaderboard):
    for r in _rows(leaderboard, "classification"):
        assert {"accuracy", "precision", "recall", "f1"}.issubset(r["test"])
        assert "cv_f1" in r


# --- The best tuned model beats the baseline (R6, reported honestly) -------
# Note: on this small single-year dataset XGBoost does NOT win regression on the
# Nov-Dec holdout (unseen calendar positions); the leaderboard reports that
# honestly. The contract we enforce is that the *ladder produces a model that
# beats the baseline*, not that XGBoost specifically wins.

def test_best_model_beats_baseline_regression(leaderboard):
    rows = {r["model"]: r for r in _rows(leaderboard, "regression")}
    dummy = rows.pop("Dummy (mean)")
    # At least one non-baseline model beats the mean baseline on the CV estimate.
    # SARIMA rows have cv_r2=null (no TimeSeriesSplit CV analogue); filter them out.
    cv_scores = [r["cv_r2"] for r in rows.values() if r["cv_r2"] is not None]
    assert max(cv_scores) > dummy["cv_r2"]
    # And on the held-out test set.
    assert max(r["test"]["r2"] for r in rows.values()) > dummy["test"]["r2"]


def test_best_model_beats_baseline_classification(leaderboard):
    rows = {r["model"]: r for r in _rows(leaderboard, "classification")}
    dummy = rows.pop("Dummy (most frequent)")
    assert max(r["test"]["f1"] for r in rows.values()) > dummy["test"]["f1"]
    # ROC-AUC of the best model clears random (0.5).
    aucs = [r["test"]["roc_auc"] for r in rows.values() if r["test"]["roc_auc"] is not None]
    assert max(aucs) > 0.5


# --- Deployed artifacts load and predict -----------------------------------

def test_served_regression_model_loads_and_predicts():
    bundle = joblib.load(MODELS_DIR / "daily_revenue.joblib")
    assert set(bundle["features"]).issubset(set(SERVED_FEATURES))
    assert "sklearn_version" in bundle
    X = make_served_features(["2015-07-03"])[bundle["features"]]
    pred = bundle["model"].predict(X)
    assert pred.shape == (1,)
    assert np.isfinite(pred[0])


def test_served_classification_model_loads_and_predicts():
    bundle = joblib.load(MODELS_DIR / "high_demand.joblib")
    assert set(bundle["features"]).issubset(set(SERVED_FEATURES))
    X = make_served_features(["2015-12-31"])[bundle["features"]]
    pred = bundle["model"].predict(X)
    assert pred.shape == (1,)
    assert pred[0] in (0, 1)


def test_inference_is_deterministic():
    bundle = joblib.load(MODELS_DIR / "daily_revenue.joblib")
    X = make_served_features(["2015-05-15"])[bundle["features"]]
    assert bundle["model"].predict(X)[0] == bundle["model"].predict(X)[0]


# --- Seeded pipeline is reproducible (cheap, no XGB search) -----------------

def test_ridge_pipeline_reproducible():
    feats = build_features(build_daily())
    train, _, test = chronological_split(feats)
    Xtr, ytr = train[SERVED_FEATURES], train[TARGET_REGRESSION]
    Xte, yte = test[SERVED_FEATURES], test[TARGET_REGRESSION]

    def fit_score():
        m = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0, random_state=42))])
        m.fit(Xtr, ytr)
        return r2_score(yte, m.predict(Xte))

    assert fit_score() == fit_score()
