"""U3 — Data pipeline.

Turns the four relational source tables into one clean day-level table for 2015
with both modelling targets:

- ``daily_revenue``    : sum of (quantity * price) over all of a day's orders  (regression target)
- ``order_count``      : number of distinct orders that day
- ``total_pizzas``     : sum of quantities that day
- ``high_demand_day``  : True when ``order_count`` exceeds the year's mean daily order count  (classification target)

Design choices (see plan U3):
- A *left* join from ``order_details`` to ``pizzas`` plus an assertion that no
  ``pizza_id`` is unmatched, so no revenue is silently dropped or turned to NaN.
- The aggregated frame is reindexed against the full 2015 calendar so days with
  zero orders become explicit zero-rows (the lag/rolling features in U5 assume a
  contiguous daily index).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repo-relative default: analysis/data/pizza_sales/ sits two levels up from this file.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pizza_sales"

_YEAR = 2015


def load_raw(data_dir: Path | str = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the four source CSVs. Returns a dict keyed by table name."""
    data_dir = Path(data_dir)
    return {
        "orders": pd.read_csv(data_dir / "orders.csv"),
        "order_details": pd.read_csv(data_dir / "order_details.csv"),
        "pizzas": pd.read_csv(data_dir / "pizzas.csv"),
        # pizza_types.csv ships as Windows-1252 (curly apostrophes in ingredient
        # names, e.g. 'Nduja), not UTF-8 — read it with the matching encoding.
        "pizza_types": pd.read_csv(data_dir / "pizza_types.csv", encoding="cp1252"),
    }


def build_line_items(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join order_details -> pizzas (price) -> orders (date); compute line revenue.

    Raises AssertionError if any order_details.pizza_id has no matching price,
    which would otherwise silently drop or NaN-poison revenue.
    """
    details = raw["order_details"]
    pizzas = raw["pizzas"][["pizza_id", "price"]]

    lines = details.merge(pizzas, on="pizza_id", how="left", indicator=True)
    unmatched = int((lines["_merge"] != "both").sum())
    assert unmatched == 0, (
        f"{unmatched} order_details rows have a pizza_id with no price in pizzas.csv; "
        "revenue would be dropped. Inspect the unmatched pizza_id values before proceeding."
    )
    lines = lines.drop(columns="_merge")
    lines["line_revenue"] = lines["quantity"] * lines["price"]

    orders = raw["orders"][["order_id", "date"]].copy()
    orders["date"] = pd.to_datetime(orders["date"])
    lines = lines.merge(orders, on="order_id", how="left")
    assert lines["date"].notna().all(), "order_details rows reference an order_id absent from orders.csv"
    return lines


def build_daily(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Build the day-level frame for the full 2015 calendar, with both targets.

    Index: DatetimeIndex, one row per calendar day in 2015 (365 rows), contiguous.
    Columns: daily_revenue, order_count, total_pizzas, high_demand_day.
    """
    raw = load_raw(data_dir)
    lines = build_line_items(raw)

    by_date = lines.groupby("date").agg(
        daily_revenue=("line_revenue", "sum"),
        total_pizzas=("quantity", "sum"),
        order_count=("order_id", "nunique"),
    )

    # Reindex to the full 2015 calendar so zero-order days are explicit zero-rows,
    # not gaps (U5 lag/rolling features require a contiguous daily index).
    full_year = pd.date_range(f"{_YEAR}-01-01", f"{_YEAR}-12-31", freq="D", name="date")
    daily = by_date.reindex(full_year)
    daily[["daily_revenue", "total_pizzas", "order_count"]] = (
        daily[["daily_revenue", "total_pizzas", "order_count"]].fillna(0)
    )
    daily["order_count"] = daily["order_count"].astype(int)
    daily["total_pizzas"] = daily["total_pizzas"].astype(int)

    # Classification target: above the year's mean daily order count.
    mean_orders = daily["order_count"].mean()
    daily["high_demand_day"] = daily["order_count"] > mean_orders

    return daily


if __name__ == "__main__":  # quick manual smoke
    df = build_daily()
    print(df.describe())
    print(f"\nrows: {len(df)}  high-demand share: {df['high_demand_day'].mean():.2%}")
