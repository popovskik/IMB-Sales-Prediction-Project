"""Tests for U4 EDA foundations (analysis/src/eda.py)."""

from __future__ import annotations

import json

import pytest

from src.data import build_daily
from src.eda import (
    chart_data,
    dow_month_heatmap,
    revenue_by_dow,
    revenue_by_month,
    summary_stats,
    weekend_contrast,
)


@pytest.fixture(scope="module")
def daily():
    return build_daily()


def test_revenue_by_dow_has_seven_rows(daily):
    df = revenue_by_dow(daily)
    assert len(df) == 7
    assert list(df["label"]) == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    # Every day-of-week group has days behind it (no div-by-zero / empty group).
    assert (df["n_days"] > 0).all()


def test_revenue_by_month_days_sum_to_year(daily):
    df = revenue_by_month(daily)
    assert len(df) == 12
    # n_days across months must sum to the full calendar.
    assert df["n_days"].sum() == len(daily)
    # total revenue across months equals the dataset total.
    assert df["total_revenue"].sum() == pytest.approx(daily["daily_revenue"].sum())


def test_heatmap_shape(daily):
    pivot = dow_month_heatmap(daily)
    assert pivot.shape == (7, 12)


def test_weekend_contrast_two_rows(daily):
    df = weekend_contrast(daily)
    assert set(df["day_type"]) == {"Weekday", "Weekend"}
    assert df["n_days"].sum() == len(daily)


def test_summary_stats_reports_closed_days(daily):
    s = summary_stats(daily)
    assert s["n_days"] == 365
    assert s["closed_days"] >= 1  # 2015 had closed (zero-order) days
    assert 0 < s["high_demand_share"] < 1


def test_chart_data_is_json_serializable(daily):
    cd = chart_data(daily)
    # Must round-trip through JSON (it gets baked into predictions.json).
    dumped = json.dumps(cd)
    assert "heatmap" in cd
    assert len(cd["heatmap"]["rows"]) == 7
    assert len(cd["heatmap"]["cols"]) == 12
    # No NaN leaked in (json.dumps would emit NaN, but we converted to None).
    assert "NaN" not in dumped
