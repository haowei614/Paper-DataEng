"""Unit tests for error injectors.

Each injector is checked for: exact corruption count, label/row alignment,
non-corruption of untouched rows, label schema, and determinism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dqgen import ROW_ID
from dqgen.inject import (
    INJECTORS,
    LABEL_COLUMNS,
    inject_cross_column_logic,
    make_mixed,
)
from conftest import diff_row_ids

# (name, kwargs, appends_rows?)
CASES = [
    ("missing_value", {"column": "amount"}, False),
    ("out_of_range", {"column": "amount"}, False),
    ("type_or_format_error", {"column": "text"}, False),
    ("invalid_category", {"column": "code"}, False),
    ("duplicate_row", {}, True),
    ("referential_integrity", {"column": "fk"}, False),
    ("cross_column_temporal", {"kind": "temporal", "start_col": "start", "end_col": "end"}, False),
    ("cross_column_sum", {"kind": "sum", "total_col": "total", "component_cols": ["a", "b"]}, False),
    ("distribution_shift", {"column": "amount"}, False),
]

RATES = [0.01, 0.05]
SEEDS = [1, 2, 3]


def _injector(name):
    # The two cross_column cases share one injector.
    if name.startswith("cross_column"):
        return inject_cross_column_logic
    return INJECTORS[name]


def _error_type(name):
    return "cross_column_logic" if name.startswith("cross_column") else name


@pytest.mark.parametrize("name,kwargs,appends", CASES)
@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("seed", SEEDS)
def test_count_and_label_alignment(synthetic_df, name, kwargs, appends, rate, seed):
    n = len(synthetic_df)
    k = int(round(n * rate))
    injector = _injector(name)
    corrupted, labels = injector(synthetic_df, rate, seed, dataset="synth", **kwargs)

    # Exactly k errors labelled.
    assert len(labels) == k
    assert list(labels.columns) == LABEL_COLUMNS
    assert (labels["error_type"] == _error_type(name)).all()
    assert (labels["rate"] == rate).all() and (labels["seed"] == seed).all()

    labelled_ids = set(labels[ROW_ID])
    assert len(labelled_ids) == k  # ids unique

    if appends:
        # Original rows untouched; exactly k new rows appended matching labels.
        assert len(corrupted) == n + k
        assert diff_row_ids(synthetic_df, corrupted) == set()
        added = set(corrupted[ROW_ID]) - set(synthetic_df[ROW_ID])
        assert added == labelled_ids
    else:
        # Row count preserved; changed rows are exactly the labelled ones.
        assert len(corrupted) == n
        assert diff_row_ids(synthetic_df, corrupted) == labelled_ids


@pytest.mark.parametrize("name,kwargs,appends", CASES)
def test_determinism(synthetic_df, name, kwargs, appends):
    injector = _injector(name)
    c1, l1 = injector(synthetic_df, 0.05, 7, dataset="synth", **kwargs)
    c2, l2 = injector(synthetic_df, 0.05, 7, dataset="synth", **kwargs)
    pd.testing.assert_frame_equal(c1, c2)
    pd.testing.assert_frame_equal(l1, l2)


def test_different_seeds_differ(synthetic_df):
    _, l1 = INJECTORS["missing_value"](synthetic_df, 0.05, 1, column="amount")
    _, l2 = INJECTORS["missing_value"](synthetic_df, 0.05, 2, column="amount")
    assert set(l1[ROW_ID]) != set(l2[ROW_ID])


def test_out_of_range_values_exceed_max(synthetic_df):
    original_max = synthetic_df["amount"].max()
    corrupted, labels = INJECTORS["out_of_range"](synthetic_df, 0.05, 1, column="amount")
    changed = corrupted[corrupted[ROW_ID].isin(labels[ROW_ID])]["amount"]
    assert (changed > original_max).all()


def test_referential_integrity_breaks_fk(synthetic_df):
    valid = set(synthetic_df["fk"])
    corrupted, labels = INJECTORS["referential_integrity"](
        synthetic_df, 0.05, 1, column="fk", valid_keys=valid
    )
    broken = corrupted[corrupted[ROW_ID].isin(labels[ROW_ID])]["fk"]
    assert broken.isin(valid).sum() == 0
    assert (broken < 0).all()


def test_cross_column_temporal_violates_order(synthetic_df):
    corrupted, labels = inject_cross_column_logic(
        synthetic_df, 0.05, 1, kind="temporal", start_col="start", end_col="end"
    )
    bad = corrupted[corrupted[ROW_ID].isin(labels[ROW_ID])]
    assert (bad["end"] < bad["start"]).all()


def test_cross_column_sum_violates_total(synthetic_df):
    corrupted, labels = inject_cross_column_logic(
        synthetic_df, 0.05, 1, kind="sum", total_col="total", component_cols=["a", "b"]
    )
    bad = corrupted[corrupted[ROW_ID].isin(labels[ROW_ID])]
    assert not np.isclose(bad["total"], bad["a"] + bad["b"]).any()


def test_make_mixed_disjoint_and_labelled(synthetic_df):
    spec = [
        {"injector": "missing_value", "rate": 0.02, "kwargs": {"column": "amount"}},
        {"injector": "invalid_category", "rate": 0.02, "kwargs": {"column": "code"}},
        {"injector": "duplicate_row", "rate": 0.01, "kwargs": {}},
    ]
    corrupted, labels = make_mixed(synthetic_df, spec, seed=5, dataset="synth")
    n = len(synthetic_df)

    # Each base row carries at most one error type (disjoint subsets).
    base_labels = labels[labels[ROW_ID] < n]
    assert base_labels[ROW_ID].is_unique

    # Counts add up.
    assert (labels["error_type"] == "missing_value").sum() == round(n * 0.02)
    assert (labels["error_type"] == "invalid_category").sum() == round(n * 0.02)
    assert (labels["error_type"] == "duplicate_row").sum() == round(n * 0.01)
