"""Tests for U3 data pipeline (analysis/src/data.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import build_daily, build_line_items, load_raw


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def daily():
    return build_daily()


# --- Happy path -------------------------------------------------------------

def test_line_revenue_matches_hand_computation(raw):
    lines = build_line_items(raw)
    # Pick one line item and verify revenue = quantity * price from pizzas.csv.
    sample = lines.iloc[0]
    price = raw["pizzas"].set_index("pizza_id").loc[sample["pizza_id"], "price"]
    assert sample["line_revenue"] == pytest.approx(sample["quantity"] * price)


def test_daily_revenue_matches_sum_over_date(raw, daily):
    lines = build_line_items(raw)
    a_date = lines["date"].iloc[0]
    expected = lines.loc[lines["date"] == a_date, "line_revenue"].sum()
    assert daily.loc[a_date, "daily_revenue"] == pytest.approx(expected)


def test_order_count_matches_distinct_orders(raw, daily):
    lines = build_line_items(raw)
    a_date = lines["date"].iloc[0]
    expected = lines.loc[lines["date"] == a_date, "order_id"].nunique()
    assert daily.loc[a_date, "order_count"] == expected


# --- Target derivation ------------------------------------------------------

def test_high_demand_is_strictly_above_mean(daily):
    mean_orders = daily["order_count"].mean()
    # True exactly when strictly greater than the mean.
    expected = daily["order_count"] > mean_orders
    assert daily["high_demand_day"].equals(expected)
    # A day exactly at the mean (if any) must be False, never True.
    at_mean = daily[daily["order_count"] == mean_orders]
    assert not at_mean["high_demand_day"].any()


# --- Join integrity (nil/empty path) ---------------------------------------

def test_unmatched_pizza_id_raises(raw):
    # Inject an order_details row whose pizza_id has no price; the pipeline must
    # detect it rather than silently dropping the revenue.
    broken = {k: v.copy() for k, v in raw.items()}
    bad_row = broken["order_details"].iloc[[0]].copy()
    bad_row["pizza_id"] = "does_not_exist_xxl"
    broken["order_details"] = pd.concat([broken["order_details"], bad_row], ignore_index=True)
    with pytest.raises(AssertionError, match="no price"):
        build_line_items(broken)


# --- Reindex / contiguity ---------------------------------------------------

def test_full_year_contiguous_index(daily):
    assert len(daily) == 365
    assert daily.index.min() == pd.Timestamp("2015-01-01")
    assert daily.index.max() == pd.Timestamp("2015-12-31")
    # Contiguous daily index, no gaps.
    assert (daily.index == pd.date_range("2015-01-01", "2015-12-31", freq="D")).all()


def test_zero_order_days_are_zero_rows(daily):
    # Any day with no orders must be a zero-row, not NaN.
    assert daily["order_count"].notna().all()
    assert daily["daily_revenue"].notna().all()
    zero_days = daily[daily["order_count"] == 0]
    assert (zero_days["daily_revenue"] == 0).all()


# --- Edge: duplicate rows do not double-count -------------------------------

def test_revenue_is_deterministic(raw):
    # Building twice yields identical totals (no accidental double-counting / state).
    assert build_line_items(raw)["line_revenue"].sum() == pytest.approx(
        build_line_items(raw)["line_revenue"].sum()
    )
