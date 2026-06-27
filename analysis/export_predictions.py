"""U6 producer — write app/public/predictions.json.

This is the single source the React dashboard (U9) reads for:
- EDA charts (revenue by DOW/month, weekend contrast, histogram, DOW x month heatmap),
- the model scorecard (the leaderboard),
- the predictor's offline fallback (per-date predicted/actual for every 2015 day).

Run after `python -m src.models` has written the served models + leaderboard:
    python export_predictions.py        # from analysis/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # make `src` importable when run as a script

from src.data import build_daily          # noqa: E402
from src.eda import chart_data            # noqa: E402
from src.features import make_served_features  # noqa: E402

MODELS_DIR = HERE / "models"
OUT = HERE.parent / "app" / "public" / "predictions.json"


def build_payload() -> dict:
    daily = build_daily()
    served = make_served_features(daily.index)

    reg = joblib.load(MODELS_DIR / "daily_revenue.joblib")
    clf = joblib.load(MODELS_DIR / "high_demand.joblib")

    pred_rev = reg["model"].predict(served[reg["features"]])
    Xc = served[clf["features"]]
    pred_dem = clf["model"].predict(Xc)
    proba = (clf["model"].predict_proba(Xc)[:, 1]
             if hasattr(clf["model"], "predict_proba") else [None] * len(Xc))

    daily_records = []
    for i, date in enumerate(daily.index):
        daily_records.append({
            "date": date.strftime("%Y-%m-%d"),
            "actual_revenue": round(float(daily["daily_revenue"].iloc[i]), 2),
            "predicted_revenue": round(float(pred_rev[i]), 2),
            "high_demand_actual": bool(daily["high_demand_day"].iloc[i]),
            "high_demand_pred": bool(pred_dem[i]),
            "high_demand_prob": (None if proba[i] is None else round(float(proba[i]), 3)),
        })

    leaderboard = json.loads((MODELS_DIR / "leaderboard.json").read_text(encoding="utf-8"))

    return {
        "year": 2015,
        "models": {"regression": reg["features"], "classification": clf["features"],
                   "sklearn_version": reg.get("sklearn_version")},
        "leaderboard": leaderboard,
        "charts": chart_data(daily),
        "daily": daily_records,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({kb:.0f} KB) — {len(payload['daily'])} daily records, "
          f"{len(payload['leaderboard'])} leaderboard rows")
