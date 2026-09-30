"""Tests for the statistical profiler and human baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dqgen import ROW_ID
from dqgen.baselines import deequ
from dqgen.baselines.human_rules import load_human_suite
from dqgen.baselines.stats_profiler import profile_rules, profile_suite
from dqgen.inject import inject_out_of_range
from dqgen.validate import validate_suite


def _clean_df(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            ROW_ID: range(n),
            "amount": rng.uniform(1, 100, n),
            "code": rng.integers(1, 4, n),          # low cardinality
            "flag": rng.choice(["Y", "N"], n),      # string / category
        }
    )


def test_profiler_excludes_row_id_and_covers_kinds():
    rules = profile_rules(_clean_df())
    types = {r["expectation_type"] for r in rules}
    assert not any(r["kwargs"].get("column") == ROW_ID for r in rules)
    assert "expect_column_values_to_not_be_null" in types
    assert "expect_column_values_to_be_between" in types
    assert "expect_column_values_to_be_in_set" in types


def test_profiler_suite_has_no_false_positives_on_clean():
    clean = _clean_df()
    suite = profile_suite(clean)
    report = validate_suite(suite, clean, dataset="clean")
    assert report.dataset_detected is False
    assert report.flagged_row_ids == set()


def test_profiler_suite_detects_out_of_range():
    clean = _clean_df()
    corrupted, labels = inject_out_of_range(clean, rate=0.05, seed=1, column="amount")
    suite = profile_suite(clean)
    report = validate_suite(suite, corrupted, dataset="corrupt")
    assert report.dataset_detected is True
    assert set(labels[ROW_ID]).issubset(report.flagged_row_ids)


def test_human_suites_load_and_build():
    for name, expected_min in [
        ("nyc_taxi", 10),
        ("tpch_customer", 4),
        ("tpch_orders", 5),
        ("tpch_lineitem", 8),
    ]:
        suite = load_human_suite(name)
        assert len(suite) >= expected_min, name


def test_deequ_disabled_returns_empty():
    assert deequ.suggest_rules(_clean_df(), enabled=False) == []
