"""Unit tests for dataset cleaning and row-id assignment (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dqgen import ROW_ID
from dqgen.datasets import add_row_ids, clean_nyc_taxi, clean_tpch


def test_add_row_ids_unique_and_contiguous():
    df = pd.DataFrame({"x": [10, 20, 30]}, index=[5, 6, 7])
    out = add_row_ids(df)
    assert list(out[ROW_ID]) == [0, 1, 2]
    assert out[ROW_ID].is_unique
    assert out.columns[0] == ROW_ID


def _taxi_frame() -> pd.DataFrame:
    """Small taxi-shaped frame: 3 valid rows + one violation per rule."""
    base = {
        "tpep_pickup_datetime": pd.Timestamp("2024-01-01 10:00"),
        "tpep_dropoff_datetime": pd.Timestamp("2024-01-01 10:30"),
        "passenger_count": 2,
        "trip_distance": 3.0,
        "fare_amount": 12.0,
        "total_amount": 15.0,
        "RatecodeID": 1,
        "payment_type": 1,
    }
    rows = [dict(base) for _ in range(3)]  # 3 clean rows
    # Violations, one each:
    rows.append({**base, "tpep_dropoff_datetime": pd.Timestamp("2024-01-01 09:00")})  # dropoff<pickup
    rows.append({**base, "passenger_count": 0})  # bad passenger count
    rows.append({**base, "trip_distance": -1.0})  # negative distance
    rows.append({**base, "fare_amount": -5.0})  # negative fare
    rows.append({**base, "RatecodeID": 99})  # invalid ratecode
    rows.append({**base, "payment_type": 0})  # invalid payment type
    return pd.DataFrame(rows)


def test_clean_nyc_taxi_removes_violations():
    df = _taxi_frame()
    cleaned, report = clean_nyc_taxi(df)
    # Only the 3 clean rows survive.
    assert len(cleaned) == 3
    # Report accounts for all removed rows.
    total_removed = sum(s.rows_removed for s in report.steps)
    assert total_removed == len(df) - len(cleaned)
    assert report.dataset == "nyc_taxi"
    # Surviving rows satisfy every rule.
    assert (cleaned["tpep_dropoff_datetime"] > cleaned["tpep_pickup_datetime"]).all()
    assert cleaned["RatecodeID"].isin([1, 2, 3, 4, 5, 6]).all()


def test_clean_tpch_verifies_integrity():
    customer = pd.DataFrame({"c_custkey": [1, 2, 3]})
    orders = pd.DataFrame(
        {"o_orderkey": [10, 11, 12, 12], "o_custkey": [1, 2, 3, 3]}  # dup pk 12
    )
    lineitem = pd.DataFrame(
        {"l_orderkey": [10, 11, 999], "l_linenumber": [1, 1, 1]}  # 999 is a dangling FK
    )
    cleaned, report = clean_tpch(
        {"orders": orders, "lineitem": lineitem, "customer": customer}
    )
    assert cleaned["orders"]["o_orderkey"].is_unique
    assert cleaned["lineitem"]["l_orderkey"].isin(cleaned["orders"]["o_orderkey"]).all()
    assert report.dataset == "tpch"
    # At least the pk-dup and the dangling FK were removed.
    assert sum(s.rows_removed for s in report.steps) >= 2
