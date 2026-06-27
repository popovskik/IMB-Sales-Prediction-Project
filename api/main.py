"""U8 — FastAPI prediction service.

Loads the two served models once at startup and exposes:
- GET  /health   -> liveness + whether both models loaded
- POST /predict  -> { date } -> predicted daily revenue + high-demand class/probability

Features are recomputed from the date via the vendored ``features.py`` (identical
to the training-time logic), and asserted against each model's persisted feature
schema before predicting, so offline/online features can never drift silently.
"""

from __future__ import annotations

import os
from datetime import date as date_type
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import make_served_features

HERE = Path(__file__).resolve().parent
# Railway builds the api/ root only, so models are copied to api/models/ at build
# time; locally they live in ../analysis/models. Prefer the local copy if present.
MODELS_DIR = Path(os.environ.get("MODELS_DIR") or (
    HERE / "models" if (HERE / "models" / "daily_revenue.joblib").exists()
    else HERE.parent / "analysis" / "models"
))

app = FastAPI(title="Restaurant Demand Predictor", version="1.0")

_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["*"],
    allow_headers=["*"],
)

_models: dict = {}


@app.on_event("startup")
def _load_models() -> None:
    """Load both bundles once. On failure, leave _models empty so /health reports it."""
    try:
        _models["regression"] = joblib.load(MODELS_DIR / "daily_revenue.joblib")
        _models["classification"] = joblib.load(MODELS_DIR / "high_demand.joblib")
    except Exception as exc:  # noqa: BLE001 — surface load failure via /health
        _models.clear()
        _models["error"] = str(exc)


# --- Schemas ----------------------------------------------------------------

class PredictRequest(BaseModel):
    date: date_type = Field(..., description="Calendar date, ISO format YYYY-MM-DD")


class HighDemand(BaseModel):
    label: str       # "High" | "Normal"
    probability: float | None


class PredictResponse(BaseModel):
    date: date_type
    predicted_revenue: float
    high_demand: HighDemand
    out_of_training_range: bool


# --- Routes -----------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    loaded = "regression" in _models and "classification" in _models
    return {"status": "ok" if loaded else "degraded", "models_loaded": loaded}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if "regression" not in _models or "classification" not in _models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    feats = make_served_features([req.date])
    reg = _models["regression"]
    clf = _models["classification"]

    # Parity guard: the model's persisted feature schema must be reproducible here.
    for bundle in (reg, clf):
        missing = [c for c in bundle["features"] if c not in feats.columns]
        if missing:
            raise HTTPException(status_code=500, detail=f"feature schema mismatch: missing {missing}")

    revenue = float(reg["model"].predict(feats[reg["features"]])[0])
    Xc = feats[clf["features"]]
    label_int = int(clf["model"].predict(Xc)[0])
    proba = None
    if hasattr(clf["model"], "predict_proba"):
        proba = round(float(clf["model"].predict_proba(Xc)[0][1]), 3)

    return PredictResponse(
        date=req.date,
        predicted_revenue=round(revenue, 2),
        high_demand=HighDemand(label="High" if label_int == 1 else "Normal", probability=proba),
        out_of_training_range=(req.date.year != 2015),
    )
