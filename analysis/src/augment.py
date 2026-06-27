"""U7 — Synthetic augmentation (professor-approved 2026-06-27).

The Maven dataset has no customer / table / server identifiers. We synthesize
them to enrich the EDA and dashboard so the project resembles a real restaurant
analytics system. **This data is for exploration and presentation only — it never
enters model training.** The model targets (daily revenue, high-demand day) are
derived solely from the real order data (see src/data.py); none of the columns
created here appear in SERVED_FEATURES or FULL_FEATURES (src/features.py).

Everything is generated under a fixed seed so the augmented EDA is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import RANDOM_SEED
from .data import DEFAULT_DATA_DIR, load_raw

N_CUSTOMERS = 1000
N_TABLES = 15
LUNCH_SERVERS = [1, 2, 3, 4]      # work the lunch shift (hour < 16)
DINNER_SERVERS = [5, 6, 7, 8]     # work the dinner shift (hour >= 16)

OUT_CSV = DEFAULT_DATA_DIR.parent / "augmented_orders.csv"


def augment_orders(data_dir: Path | str = DEFAULT_DATA_DIR, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Assign a synthetic customer / table / server to each real order.

    - customer_id: drawn from a 1000-customer pool with a Zipf-like weighting so a
      minority of regulars account for many visits (realistic repeat behaviour).
    - table_id: 1..15, roughly uniform.
    - server_id: constrained by shift — lunch servers before 16:00, dinner servers after.
    """
    rng = np.random.default_rng(seed)
    raw = load_raw(data_dir)
    orders = raw["orders"][["order_id", "date", "time"]].copy()
    n = len(orders)

    # Zipf-like customer weights: customer k chosen with weight 1/(k+1), normalized.
    ranks = np.arange(1, N_CUSTOMERS + 1)
    weights = (1.0 / ranks)
    weights /= weights.sum()
    orders["customer_id"] = rng.choice(np.arange(1, N_CUSTOMERS + 1), size=n, p=weights)

    orders["table_id"] = rng.integers(1, N_TABLES + 1, size=n)

    hour = pd.to_datetime(orders["time"], format="%H:%M:%S", errors="coerce").dt.hour.fillna(12).astype(int)
    is_dinner = hour >= 16
    servers = np.where(
        is_dinner,
        rng.choice(DINNER_SERVERS, size=n),
        rng.choice(LUNCH_SERVERS, size=n),
    )
    orders["server_id"] = servers
    return orders


# --- Augmented EDA (reads ONLY the synthetic frame) -------------------------

def new_vs_returning(aug: pd.DataFrame) -> pd.DataFrame:
    """Per-order flag of first visit vs repeat, ordered by date — synthetic."""
    a = aug.sort_values(["date", "time"]).copy()
    a["visit_index"] = a.groupby("customer_id").cumcount()
    a["is_returning"] = a["visit_index"] > 0
    share = a.groupby("date")["is_returning"].mean().rename("returning_share")
    return share.reset_index()


def table_utilization(aug: pd.DataFrame) -> pd.DataFrame:
    g = aug.groupby("table_id").size().rename("orders").reset_index()
    return g.sort_values("table_id")


def server_throughput(aug: pd.DataFrame) -> pd.DataFrame:
    g = aug.groupby("server_id").size().rename("orders").reset_index()
    return g.sort_values("server_id")


def customer_frequency(aug: pd.DataFrame) -> pd.DataFrame:
    """Distribution of visit counts per customer — shows the regulars tail."""
    counts = aug.groupby("customer_id").size()
    return pd.DataFrame({
        "metric": ["unique_customers", "repeat_customers", "max_visits", "mean_visits"],
        "value": [int(counts.size), int((counts > 1).sum()), int(counts.max()), float(counts.mean())],
    })


if __name__ == "__main__":
    df = augment_orders()
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} orders)")
    print(customer_frequency(df).to_string(index=False))
