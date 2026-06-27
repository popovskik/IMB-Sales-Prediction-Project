"""Tests for U8 FastAPI prediction service (api/main.py)."""

from __future__ import annotations

import joblib
import pandas as pd
from fastapi.testclient import TestClient

import features as api_features
from main import MODELS_DIR, app


# --- Health -----------------------------------------------------------------

def test_health_reports_models_loaded():
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True


# --- /predict contract ------------------------------------------------------

def test_predict_valid_date_returns_both_predictions():
    with TestClient(app) as client:
        r = client.post("/predict", json={"date": "2015-07-03"})  # a summer Friday
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["predicted_revenue"], (int, float))
    assert body["high_demand"]["label"] in {"High", "Normal"}
    p = body["high_demand"]["probability"]
    assert p is None or 0.0 <= p <= 1.0
    assert body["out_of_training_range"] is False


def test_predict_rejects_malformed_date():
    with TestClient(app) as client:
        r = client.post("/predict", json={"date": "not-a-date"})
    assert r.status_code == 422  # Pydantic validation, not an unhandled 500


def test_predict_out_of_range_date_flagged():
    with TestClient(app) as client:
        r = client.post("/predict", json={"date": "2099-01-01"})
    assert r.status_code == 200
    assert r.json()["out_of_training_range"] is True


# --- Feature parity (guards offline/online drift) ---------------------------

def test_vendored_features_match_analysis_source():
    from src.features import make_served_features as source_fn  # analysis source of truth

    dates = ["2015-01-01", "2015-07-03", "2015-12-25", "2015-12-31"]
    api_df = api_features.make_served_features(dates)
    src_df = source_fn(dates)
    pd.testing.assert_frame_equal(api_df, src_df)


def test_api_features_cover_model_schema():
    served_cols = set(api_features.SERVED_FEATURES)
    for fname in ("daily_revenue.joblib", "high_demand.joblib"):
        bundle = joblib.load(MODELS_DIR / fname)
        assert set(bundle["features"]).issubset(served_cols)
