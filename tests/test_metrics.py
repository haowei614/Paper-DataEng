"""Tests for metric computations."""

from __future__ import annotations

import math

import pandas as pd

from dqgen import ROW_ID
from dqgen.expectations import build_expectation
from dqgen.metrics import (
    detection_scores,
    executability_rate,
    redundancy,
    row_false_positive_rate,
    rule_false_positive_rate,
)
from dqgen.validate import ExpectationOutcome, ValidationReport


def _report(outcomes, n_rows=100):
    return ValidationReport(dataset="d", n_rows=n_rows, outcomes=outcomes)


def test_executability_rate():
    counts = {"valid": 8, "unknown_expectation": 2, "json_error": 0, "invalid_kwargs": 0}
    assert executability_rate(counts, n_runtime_errors=1) == 0.7  # (8-1)/10


def test_executability_rate_empty():
    assert math.isnan(executability_rate({}, 0))


def test_rule_and_row_fp_rates():
    outcomes = [
        ExpectationOutcome(0, "e", {}, success=True),
        ExpectationOutcome(1, "e", {}, success=False, unexpected_row_ids={3, 4}),
        ExpectationOutcome(2, "e", {}, success=False, unexpected_row_ids={4, 5}),
    ]
    report = _report(outcomes, n_rows=100)
    assert rule_false_positive_rate(report) == 2 / 3
    assert row_false_positive_rate(report) == 3 / 100  # union {3,4,5}


def test_detection_scores_basic():
    labels = pd.DataFrame(
        {ROW_ID: [1, 2, 3], "error_type": ["missing_value", "missing_value", "out_of_range"]}
    )
    scores = detection_scores(flagged_row_ids={0, 1, 2}, labels=labels)
    assert scores.true_positives == 2
    assert scores.precision == 2 / 3
    assert scores.recall == 2 / 3
    assert scores.per_type_recall["missing_value"] == 1.0  # both 1,2 flagged
    assert scores.per_type_recall["out_of_range"] == 0.0  # 3 not flagged


def test_detection_scores_empty_labels_is_nan():
    scores = detection_scores(flagged_row_ids=set(), labels=pd.DataFrame({ROW_ID: [], "error_type": []}))
    assert math.isnan(scores.recall)
    assert math.isnan(scores.precision)
    assert math.isnan(scores.f1)


def test_redundancy_counts_exact_duplicates():
    e1 = build_expectation("expect_column_values_to_not_be_null", {"column": "a"})
    e2 = build_expectation("expect_column_values_to_not_be_null", {"column": "a"})  # dup
    e3 = build_expectation("expect_column_values_to_not_be_null", {"column": "b"})
    assert redundancy([e1, e2, e3]) == 1
