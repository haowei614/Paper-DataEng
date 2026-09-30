"""Integration tests for suite validation against clean/corrupted data (real GX)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dqgen import ROW_ID
from dqgen.expectations import build_expectation
from dqgen.inject import inject_out_of_range
from dqgen.validate import validate_suite


def _clean_df(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame({ROW_ID: range(n), "amount": rng.uniform(0, 100, n).round(2)})


def test_flagged_rows_match_injected_labels():
    clean = _clean_df()
    corrupted, labels = inject_out_of_range(clean, rate=0.1, seed=1, column="amount")
    exp = build_expectation(
        "expect_column_values_to_be_between",
        {"column": "amount", "min_value": 0, "max_value": 100},
    )
    report = validate_suite([exp], corrupted, dataset="d")
    assert report.dataset_detected is True
    assert report.flagged_row_ids == set(labels[ROW_ID])
    assert report.n_failed == 1
    assert report.n_runtime_errors == 0


def test_no_false_positives_on_clean():
    clean = _clean_df()
    exp = build_expectation(
        "expect_column_values_to_be_between",
        {"column": "amount", "min_value": 0, "max_value": 100},
    )
    report = validate_suite([exp], clean, dataset="d")
    assert report.dataset_detected is False
    assert report.flagged_row_ids == set()


def test_runtime_error_is_captured_not_raised():
    clean = _clean_df()
    # Expectation referencing a column that does not exist.
    exp = build_expectation("expect_column_values_to_not_be_null", {"column": "does_not_exist"})
    report = validate_suite([exp], clean, dataset="d")
    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    # Either GX raises (runtime_error) or returns a failure; both are handled
    # without crashing. A missing column must not count as a clean pass.
    assert outcome.runtime_error or outcome.success is False


def test_not_null_flags_missing_values():
    from dqgen.inject import inject_missing_value

    clean = _clean_df()
    corrupted, labels = inject_missing_value(clean, rate=0.05, seed=2, column="amount")
    exp = build_expectation("expect_column_values_to_not_be_null", {"column": "amount"})
    report = validate_suite([exp], corrupted, dataset="d")
    assert report.flagged_row_ids == set(labels[ROW_ID])
