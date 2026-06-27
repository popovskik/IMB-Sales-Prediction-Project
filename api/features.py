"""Vendored served-feature computation for the API.

This is a self-contained copy of ``make_served_features`` from
``analysis/src/features.py``. The API is deployed from the ``api/`` root only
(Railway builds this directory), so it cannot import the analysis package — the
feature logic is vendored here instead. The feature-parity test
(``api/tests/test_api.py``) compares this against the analysis version and against
each model's persisted feature schema, so drift is caught in CI rather than in
production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CALENDAR_FEATURES = [
    "day_of_week", "month", "week_of_year", "day_of_year", "quarter",
    "is_weekend", "is_month_start", "is_month_end",
]
CYCLICAL_FEATURES = ["sin_dow", "cos_dow", "sin_month", "cos_month"]
SERVED_FEATURES = CALENDAR_FEATURES + CYCLICAL_FEATURES


def make_served_features(dates) -> pd.DataFrame:
    """Compute the served feature set (calendar + cyclical) for any date(s).

    Must stay byte-for-byte equivalent to analysis/src/features.make_served_features.
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
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df[SERVED_FEATURES]
