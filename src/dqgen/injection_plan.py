"""Per-table error-injection plan (target columns for each error type).

This is an experimental-design artifact: it fixes which column each injector
targets in each table. Rates and seeds come from the config; this module only
fixes the *what*, not the *how much*. Edit here to change injection targets.

Keys are table identifiers: ``"nyc_taxi"`` for the single-table taxi dataset and
``"tpch_<table>"`` for TPC-H tables. Each entry is an injector spec in the form
consumed by :func:`dqgen.inject.make_mixed` and the injector registry:
``{"injector": <name>, "kwargs": {...}}``.

Notes:
* The taxi dataset is single-table, so ``referential_integrity`` is not
  applicable and is omitted there.
* ``referential_integrity`` uses negative sentinel keys, so it needs no parent
  key set at injection time.
"""

from __future__ import annotations

INJECTION_PLANS: dict[str, list[dict]] = {
    "nyc_taxi": [
        {"injector": "missing_value", "kwargs": {"column": "passenger_count"}},
        {"injector": "out_of_range", "kwargs": {"column": "trip_distance", "mode": "high"}},
        {"injector": "type_or_format_error", "kwargs": {"column": "store_and_fwd_flag"}},
        {"injector": "invalid_category", "kwargs": {"column": "RatecodeID", "invalid_value": 99}},
        {"injector": "duplicate_row", "kwargs": {}},
        {"injector": "cross_column_logic",
         "kwargs": {"kind": "temporal", "start_col": "tpep_pickup_datetime",
                    "end_col": "tpep_dropoff_datetime"}},
        {"injector": "distribution_shift", "kwargs": {"column": "fare_amount", "scale": 3.0}},
    ],
    "tpch_customer": [
        {"injector": "missing_value", "kwargs": {"column": "c_name"}},
        {"injector": "out_of_range", "kwargs": {"column": "c_acctbal", "mode": "high"}},
        {"injector": "invalid_category", "kwargs": {"column": "c_mktsegment", "invalid_value": "ZZZ"}},
        {"injector": "duplicate_row", "kwargs": {}},
        {"injector": "distribution_shift", "kwargs": {"column": "c_acctbal", "scale": 3.0}},
    ],
    "tpch_orders": [
        {"injector": "referential_integrity", "kwargs": {"column": "o_custkey"}},
        {"injector": "missing_value", "kwargs": {"column": "o_orderstatus"}},
        {"injector": "invalid_category", "kwargs": {"column": "o_orderstatus", "invalid_value": "Z"}},
        {"injector": "out_of_range", "kwargs": {"column": "o_totalprice", "mode": "high"}},
        {"injector": "duplicate_row", "kwargs": {}},
        {"injector": "distribution_shift", "kwargs": {"column": "o_totalprice", "scale": 3.0}},
    ],
    "tpch_lineitem": [
        {"injector": "referential_integrity", "kwargs": {"column": "l_orderkey"}},
        {"injector": "out_of_range", "kwargs": {"column": "l_quantity", "mode": "high"}},
        {"injector": "invalid_category", "kwargs": {"column": "l_returnflag", "invalid_value": "Z"}},
        {"injector": "cross_column_logic",
         "kwargs": {"kind": "temporal", "start_col": "l_shipdate", "end_col": "l_receiptdate"}},
        {"injector": "duplicate_row", "kwargs": {}},
        {"injector": "distribution_shift", "kwargs": {"column": "l_extendedprice", "scale": 3.0}},
    ],
}

# Per-injector rate used when building the single "mixed" dataset (disjoint
# subsets), independent of the per-error-type sweep rates.
MIXED_RATE_PER_INJECTOR = 0.02


def error_type_for(injector: str) -> str:
    """The label recorded in injected-error labels for a given injector."""
    return injector  # injector name == error_type in labels


def plan_for(table: str) -> list[dict]:
    if table not in INJECTION_PLANS:
        raise KeyError(f"no injection plan for table {table!r}")
    return INJECTION_PLANS[table]
