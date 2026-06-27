"""Tests for U5 feature engineering (analysis/src/features.py).

The leakage guards here are the most important tests in the project — they are
what makes the honest-evaluation story (R14) defensible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import build_daily
from src.features import (
    FULL_FEATURES,
    LAG_FEATURES,
    SERVED_FEATURES,
    add_lag_features,
    build_features,
    chronological_split,
    make_served_features,
    select_features,
)


@pytest.fixture(scope="module")
def daily():
    return build_daily()


@pytest.fixture(scope="module")
def feats(daily):
    return build_features(daily)


# --- Served features (date-only, API-safe) ----------------------------------

def test_served_features_from_bare_date():
    df = make_served_features(["2015-07-03"])  # a Friday
    assert list(df.columns) == SERVED_FEATURES
    assert df.iloc[0]["day_of_week"] == 4  # Fri = 4
    assert df.iloc[0]["is_weekend"] == 0


def test_served_features_column_order_is_stable():
    # The API asserts parity against this exact order — it must not drift.
    df = make_served_features(pd.date_range("2015-01-01", periods=5))
    assert list(df.columns) == SERVED_FEATURES


def test_cyclical_encoding_unit_circle(feats):
    # sin^2 + cos^2 == 1 for both cyclical pairs.
    assert np.allclose(feats["sin_dow"] ** 2 + feats["cos_dow"] ** 2, 1.0)
    assert np.allclose(feats["sin_month"] ** 2 + feats["cos_month"] ** 2, 1.0)


def test_monday_sunday_are_circular_neighbors():
    # On the day-of-week circle, Sun (6) should be as close to Mon (0) as Tue (1) is.
    df = make_served_features(["2015-01-05", "2015-01-06", "2015-01-11"])  # Mon, Tue, Sun
    mon = df.loc["2015-01-05", ["sin_dow", "cos_dow"]].to_numpy()
    tue = df.loc["2015-01-06", ["sin_dow", "cos_dow"]].to_numpy()
    sun = df.loc["2015-01-11", ["sin_dow", "cos_dow"]].to_numpy()
    assert np.linalg.norm(mon - sun) == pytest.approx(np.linalg.norm(mon - tue), rel=1e-6)


# --- Lag features are past-only (leakage guard) -----------------------------

def test_lag_features_use_strictly_past_values(daily):
    d = add_lag_features(daily)
    # Pick a row well past the warm-up window.
    i = 100
    today = d.index[i]
    assert d.loc[today, "revenue_lag_1"] == d["daily_revenue"].iloc[i - 1]
    assert d.loc[today, "revenue_lag_7"] == d["daily_revenue"].iloc[i - 7]
    # rolling_revenue_7d must average the 7 days BEFORE today, never today.
    expected = d["daily_revenue"].iloc[i - 7:i].mean()
    assert d.loc[today, "rolling_revenue_7d"] == pytest.approx(expected)


def test_leading_lag_nans_filled(feats):
    # First row has no history; lags filled with 0, no NaN leaks into the matrix.
    assert feats[LAG_FEATURES].notna().all().all()
    assert feats.iloc[0]["revenue_lag_1"] == 0


# --- Chronological split (no future leakage) --------------------------------

def test_split_is_contiguous_and_non_overlapping(feats):
    train, val, test = chronological_split(feats)
    # Disjoint.
    assert train.index.intersection(val.index).empty
    assert val.index.intersection(test.index).empty
    assert train.index.intersection(test.index).empty
    # Ordered: all train dates precede all val dates precede all test dates.
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    # Cover the whole year.
    assert len(train) + len(val) + len(test) == len(feats)


def test_split_boundaries(feats):
    train, val, test = chronological_split(feats)
    assert train.index.month.max() == 9
    assert set(val.index.month) == {10}
    assert set(test.index.month) == {11, 12}


# --- Feature selection fits on train only -----------------------------------

def test_select_features_regression_returns_subset(feats):
    train, _, _ = chronological_split(feats)
    selected = select_features(train[FULL_FEATURES], train["daily_revenue"], "regression")
    assert len(selected) > 0
    assert set(selected).issubset(set(FULL_FEATURES))


def test_select_features_classification_returns_subset(feats):
    train, _, _ = chronological_split(feats)
    selected = select_features(train[FULL_FEATURES], train["high_demand_day"], "classification")
    assert len(selected) > 0
    assert set(selected).issubset(set(FULL_FEATURES))


def test_select_features_rejects_bad_task(feats):
    train, _, _ = chronological_split(feats)
    with pytest.raises(ValueError, match="task"):
        select_features(train[SERVED_FEATURES], train["daily_revenue"], "clustering")
