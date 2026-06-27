"""U4 — EDA and visualization foundations.

Every function returns a tidy pandas object (the numbers), so the *same*
aggregations feed both the Quarto report (U10, via matplotlib figures) and the
React dashboard (U9, via ``chart_data`` -> predictions.json in U6). Figures are
built from these frames, never from separate ad-hoc queries, so the report and
the dashboard can never disagree.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ORANGE = "#ff4800"


def _with_calendar(daily: pd.DataFrame) -> pd.DataFrame:
    """Attach day_of_week (0=Mon) and month (1-12) from the DatetimeIndex."""
    out = daily.copy()
    out["day_of_week"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_weekend"] = out["day_of_week"] >= 5
    return out


# --- Tidy aggregations (the authoritative numbers) --------------------------

def revenue_by_dow(daily: pd.DataFrame) -> pd.DataFrame:
    """Mean daily revenue and order count by day of week (Mon..Sun)."""
    d = _with_calendar(daily)
    g = d.groupby("day_of_week").agg(
        mean_revenue=("daily_revenue", "mean"),
        mean_orders=("order_count", "mean"),
        n_days=("daily_revenue", "size"),
    ).reindex(range(7))
    g["label"] = DOW_LABELS
    return g.reset_index()


def revenue_by_month(daily: pd.DataFrame) -> pd.DataFrame:
    """Total and mean daily revenue by month (Jan..Dec)."""
    d = _with_calendar(daily)
    g = d.groupby("month").agg(
        total_revenue=("daily_revenue", "sum"),
        mean_revenue=("daily_revenue", "mean"),
        n_days=("daily_revenue", "size"),
    ).reindex(range(1, 13))
    g["label"] = MONTH_LABELS
    return g.reset_index()


def dow_month_heatmap(daily: pd.DataFrame) -> pd.DataFrame:
    """Mean daily revenue pivoted as day-of-week (rows) x month (cols)."""
    d = _with_calendar(daily)
    pivot = d.pivot_table(index="day_of_week", columns="month",
                          values="daily_revenue", aggfunc="mean")
    pivot = pivot.reindex(index=range(7), columns=range(1, 13))
    pivot.index = DOW_LABELS
    pivot.columns = MONTH_LABELS
    return pivot


def weekend_contrast(daily: pd.DataFrame) -> pd.DataFrame:
    """Mean revenue and orders for weekdays vs weekends."""
    d = _with_calendar(daily)
    g = d.groupby("is_weekend").agg(
        mean_revenue=("daily_revenue", "mean"),
        mean_orders=("order_count", "mean"),
        n_days=("daily_revenue", "size"),
    )
    g.index = ["Weekday", "Weekend"]
    return g.reset_index(names="day_type")


def revenue_histogram(daily: pd.DataFrame, bins: int = 30) -> pd.DataFrame:
    """Histogram of daily revenue as (bin_left, bin_right, count)."""
    counts, edges = np.histogram(daily["daily_revenue"], bins=bins)
    return pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts,
    })


def summary_stats(daily: pd.DataFrame) -> dict:
    """Headline numbers for the report's overview and exec summary."""
    return {
        "n_days": int(len(daily)),
        "closed_days": int((daily["order_count"] == 0).sum()),
        "total_revenue": float(daily["daily_revenue"].sum()),
        "mean_daily_revenue": float(daily["daily_revenue"].mean()),
        "median_daily_revenue": float(daily["daily_revenue"].median()),
        "mean_daily_orders": float(daily["order_count"].mean()),
        "high_demand_share": float(daily["high_demand_day"].mean()),
    }


# --- Chart data for the dashboard (JSON-ready) ------------------------------

def chart_data(daily: pd.DataFrame) -> dict:
    """Bundle all EDA aggregations into JSON-serializable structures.

    Consumed by analysis/export_predictions.py (U6) to write predictions.json,
    which the React dashboard (U9) reads for its charts.
    """
    heat = dow_month_heatmap(daily).round(2)
    return {
        "summary": summary_stats(daily),
        "revenue_by_dow": revenue_by_dow(daily).round(2).to_dict(orient="records"),
        "revenue_by_month": revenue_by_month(daily).round(2).to_dict(orient="records"),
        "weekend_contrast": weekend_contrast(daily).round(2).to_dict(orient="records"),
        "revenue_histogram": revenue_histogram(daily).round(2).to_dict(orient="records"),
        "heatmap": {
            "rows": list(heat.index),
            "cols": list(heat.columns),
            # NaN (no such dow/month combo) -> None for valid JSON
            "values": [[None if pd.isna(v) else float(v) for v in row] for row in heat.values],
        },
    }


# --- Figures for the Quarto report ------------------------------------------

def fig_revenue_by_month(daily: pd.DataFrame):
    df = revenue_by_month(daily)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(df["label"], df["mean_revenue"], marker="o", color=ORANGE)
    ax.set_title("Mean daily revenue by month (2015)")
    ax.set_ylabel("Mean daily revenue ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_revenue_by_dow(daily: pd.DataFrame):
    df = revenue_by_dow(daily)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(df["label"], df["mean_revenue"], color=ORANGE)
    ax.set_title("Mean daily revenue by day of week")
    ax.set_ylabel("Mean daily revenue ($)")
    fig.tight_layout()
    return fig


def fig_heatmap(daily: pd.DataFrame):
    pivot = dow_month_heatmap(daily)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Oranges")
    ax.set_xticks(range(12), pivot.columns)
    ax.set_yticks(range(7), pivot.index)
    ax.set_title("Mean daily revenue — day of week x month")
    fig.colorbar(im, ax=ax, label="$")
    fig.tight_layout()
    return fig


def fig_revenue_distribution(daily: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(daily["daily_revenue"], bins=30, color=ORANGE, alpha=0.85)
    ax.set_title("Distribution of daily revenue")
    ax.set_xlabel("Daily revenue ($)")
    ax.set_ylabel("Days")
    fig.tight_layout()
    return fig
