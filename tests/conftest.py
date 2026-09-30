"""Shared test fixtures and helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dqgen import ROW_ID


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """A 1000-row frame exercising every injector's target column type.

    Columns:
        amount        float   -> missing_value, out_of_range, distribution_shift
        code          int     -> invalid_category (valid set {1..6})
        text          object  -> type_or_format_error
        fk            int     -> referential_integrity
        start, end    datetime-> cross_column_logic (temporal)
        total, a, b   float   -> cross_column_logic (sum; total == a + b)
    """
    n = 1000
    rng = np.random.default_rng(0)
    a = rng.uniform(1, 50, n).round(2)
    b = rng.uniform(1, 50, n).round(2)
    start = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 30 * 24, n), unit="h")
    df = pd.DataFrame(
        {
            ROW_ID: range(n),
            "amount": rng.uniform(1, 100, n).round(2),
            "code": rng.integers(1, 7, n),
            "text": [f"row-{i}" for i in range(n)],
            "fk": rng.integers(1, 500, n),
            "start": start,
            "end": start + pd.to_timedelta(rng.integers(1, 120, n), unit="m"),
            "a": a,
            "b": b,
            "total": (a + b).round(2),
        }
    )
    return df


def diff_row_ids(before: pd.DataFrame, after: pd.DataFrame) -> set:
    """Return the set of ROW_IDs present in both frames whose cells differ.

    NaN-aware: a cell that is NaN in both is treated as unchanged.
    """
    common = set(before[ROW_ID]) & set(after[ROW_ID])
    b = before[before[ROW_ID].isin(common)].set_index(ROW_ID).sort_index()
    a = after[after[ROW_ID].isin(common)].set_index(ROW_ID).sort_index()
    a = a[b.columns]
    ne = (a.values != b.values) & ~(pd.isna(a.values) & pd.isna(b.values))
    changed = ne.any(axis=1)
    return set(b.index[changed])
