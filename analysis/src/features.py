"""U5 — Leakage-safe feature engineering.

Two feature sets:

- ``SERVED_FEATURES`` (calendar + cyclical) are fully determined by the date, so
  the live predictor (U8) can compute them for any date with no history. This is
  the set the deployed models are trained on, and ``make_served_features`` is the
  exact function the API vendors and reuses.
- ``FULL_FEATURES`` adds past-only lag/rolling features for the report's
  best-model comparison (U10). Lags use ``shift`` on the date-sorted series, and
  the 7-day rolling mean uses ``shift(1)`` first so the current day never leaks in.

The chronological split (train Jan-Sep / val Oct / test Nov-Dec) and the
fit-on-train-only feature selector keep evaluation honest (R3-R5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import RANDOM_SEED

CALENDAR_FEATURES = [
    "day_of_week", "month", "week_of_year", "day_of_year", "quarter",
    "is_weekend", "is_month_start", "is_month_end",
]
CYCLICAL_FEATURES = ["sin_dow", "cos_dow", "sin_month", "cos_month"]
LAG_FEATURES = [
    "revenue_lag_1", "revenue_lag_7", "demand_lag_1", "demand_lag_7", "rolling_revenue_7d",
]

SERVED_FEATURES = CALENDAR_FEATURES + CYCLICAL_FEATURES
FULL_FEATURES = SERVED_FEATURES + LAG_FEATURES

TARGET_REGRESSION = "daily_revenue"
TARGET_CLASSIFICATION = "high_demand_day"


def make_served_features(dates) -> pd.DataFrame:
    """Compute the served feature set (calendar + cyclical) for any date(s).

    Used identically at training time and at serve time (the API vendors this
    function), so offline and online features can never drift. Accepts anything
    ``pd.to_datetime`` understands; returns a DataFrame indexed by date with
    columns in ``SERVED_FEATURES`` order.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    df = pd.DataFrame(index=idx)
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["week_of_year"] = idx.isocalendar().week.to_numpy().astype(int)
    df["day_of_year"] = idx.dayofyear
    df["quarter"] = idx.quarter
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_month_start"] = idx.is_month_start.astype(int)
    df["is_month_end"] = idx.is_month_end.astype(int)
    # Cyclical encodings keep circular continuity (Mon adjacent to Sun, Dec to Jan).
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df[SERVED_FEATURES]


def add_lag_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Append past-only lag/rolling features to a date-sorted daily frame.

    Every lag is strictly historical: ``shift(k)`` for k-day lags, and the 7-day
    rolling mean is ``shift(1).rolling(7)`` so the current day is excluded. The
    leading rows that have no history are filled with 0 (no prior data).
    """
    d = daily.sort_index().copy()
    d["revenue_lag_1"] = d["daily_revenue"].shift(1)
    d["revenue_lag_7"] = d["daily_revenue"].shift(7)
    d["demand_lag_1"] = d["order_count"].shift(1)
    d["demand_lag_7"] = d["order_count"].shift(7)
    d["rolling_revenue_7d"] = d["daily_revenue"].shift(1).rolling(7, min_periods=1).mean()
    d[LAG_FEATURES] = d[LAG_FEATURES].fillna(0)
    return d


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Full modelling frame: served + lag features + both targets, date-indexed."""
    served = make_served_features(daily.index)
    lagged = add_lag_features(daily)
    out = served.join(lagged[LAG_FEATURES])
    out[TARGET_REGRESSION] = daily["daily_revenue"]
    out[TARGET_CLASSIFICATION] = daily["high_demand_day"].astype(int)
    return out


def chronological_split(df: pd.DataFrame):
    """Time-ordered split: train = Jan-Sep, val = Oct, test = Nov-Dec.

    Returns (train, val, test) frames. No shuffling — preserves temporal order so
    lag features and evaluation never see the future.
    """
    df = df.sort_index()
    month = df.index.month
    train = df[month <= 9]
    val = df[month == 10]
    test = df[month >= 11]
    return train, val, test


def select_features(X_train: pd.DataFrame, y_train: pd.Series, task: str) -> list[str]:
    """Fit-on-train-only feature selection (R4).

    1. Correlation filter: drop one of any pair of features with |corr| > 0.95.
    2. Model-based: Lasso (regression) or RFE/LogisticRegression (classification).

    Returns the selected feature names. Falls back to the correlation-filtered set
    if the model-based step selects nothing.
    """
    # 1. Collinearity filter.
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    drop = [c for c in upper.columns if (upper[c] > 0.95).any()]
    kept = [c for c in X_train.columns if c not in drop]
    # Standardize (fit on train only) so Lasso penalties and the logistic solver
    # are scale-fair across features like day_of_year (0-365) vs is_weekend (0-1).
    Xk = pd.DataFrame(StandardScaler().fit_transform(X_train[kept]), columns=kept, index=X_train.index)

    # 2. Model-based selection.
    if task == "regression":
        lasso = Lasso(alpha=1.0, random_state=RANDOM_SEED, max_iter=10000)
        lasso.fit(Xk, y_train)
        selected = [f for f, c in zip(kept, lasso.coef_) if abs(c) > 1e-8]
    elif task == "classification":
        n_select = min(8, len(kept))
        rfe = RFE(LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
                  n_features_to_select=n_select)
        rfe.fit(Xk, y_train)
        selected = [f for f, keep in zip(kept, rfe.support_) if keep]
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")

    return selected or kept
