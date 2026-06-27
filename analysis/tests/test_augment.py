"""Tests for U7 synthetic augmentation (analysis/src/augment.py).

The isolation tests are the important ones: they prove the synthetic data never
reaches model training.
"""

from __future__ import annotations

import pytest

from src.augment import (
    N_CUSTOMERS,
    augment_orders,
    customer_frequency,
    server_throughput,
    table_utilization,
)
from src.features import FULL_FEATURES, SERVED_FEATURES


@pytest.fixture(scope="module")
def aug():
    return augment_orders()


# --- Isolation from model training (the whole point) ------------------------

SYNTHETIC_COLS = {"customer_id", "table_id", "server_id"}


def test_synthetic_columns_are_not_model_features():
    # No synthetic column may appear in either feature set used for training.
    assert SYNTHETIC_COLS.isdisjoint(set(SERVED_FEATURES))
    assert SYNTHETIC_COLS.isdisjoint(set(FULL_FEATURES))


def _imports_augment(module) -> bool:
    """True if the module has an actual import of the augment module (not just the
    substring 'augment', which appears in benign words like 'lag-augmented')."""
    import ast

    with open(module.__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "augment" in node.module:
            return True
        if isinstance(node, ast.Import) and any("augment" in n.name for n in node.names):
            return True
    return False


def test_features_module_does_not_import_augment():
    import src.features as feats
    assert not _imports_augment(feats)


def test_models_module_does_not_import_augment():
    import src.models as models
    assert not _imports_augment(models)


# --- Determinism ------------------------------------------------------------

def test_augmentation_is_deterministic():
    a = augment_orders(seed=42)
    b = augment_orders(seed=42)
    assert a.equals(b)


# --- Distributions are sane -------------------------------------------------

def test_customer_pool_and_repeat_behaviour(aug):
    assert aug["customer_id"].between(1, N_CUSTOMERS).all()
    freq = {row["metric"]: row["value"] for _, row in customer_frequency(aug).iterrows()}
    # The Zipf-like weighting must produce real regulars (repeat customers).
    assert freq["repeat_customers"] > 0
    assert freq["max_visits"] > freq["mean_visits"]


def test_tables_in_range(aug):
    assert aug["table_id"].between(1, 15).all()
    assert table_utilization(aug)["orders"].sum() == len(aug)


def test_servers_respect_shift_constraint(aug):
    import pandas as pd
    hour = pd.to_datetime(aug["time"], format="%H:%M:%S", errors="coerce").dt.hour
    dinner = aug[hour >= 16]
    lunch = aug[hour < 16]
    assert set(dinner["server_id"]).issubset({5, 6, 7, 8})
    assert set(lunch["server_id"]).issubset({1, 2, 3, 4})
    assert server_throughput(aug)["orders"].sum() == len(aug)
