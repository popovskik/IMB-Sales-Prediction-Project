# Restaurant Demand Analytics

Predicting a pizza restaurant's **daily revenue** (regression) and flagging **high-demand days** (classification) from a full year of order data — an end-to-end machine learning study with a live, deployed prediction app.

> IMB MADA 2026 capstone. Dataset: [Maven Analytics Pizza Place Sales](https://mavenanalytics.io/data-playground/pizza-place-sales) (public, no personal data).

## Live app

- **Dashboard:** _(Vercel link — added in U9/U11)_
- **Prediction API:** _(Railway link — added in U8)_

## What it does

Two day-level prediction questions, answered honestly:

1. **Regression** — predict `Daily_Revenue` (sum of price × quantity across a day's orders).
2. **Classification** — predict `High_Demand_Day` (a day whose order count exceeds the year's mean daily order count).

Both models use only calendar features derived from the date (no leakage). The pipeline trains a full model ladder (Dummy → linear → Decision Tree → tuned XGBoost) and reports test-set metrics with the baseline kept in the leaderboard.

## Architecture

_(architecture.png embedded here in U11; source in `architecture.md`)_

## Repository layout

```
analysis/   ML pipeline + Quarto report (D1)
api/        FastAPI prediction service (D2 — Railway)
app/        React dashboard (D2 — Vercel)
slides/     reveal.js presentation (D4)
docs/       executive summary (D5), AI-workflow reflection (D3)
```

## Run the analysis locally

```bash
cd analysis
python -m venv .venv
.venv/Scripts/activate        # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
quarto render report.qmd      # produces report.html
```

Everything is seeded (`RANDOM_SEED = 42`) and runs end-to-end from a clean checkout.
