"""Statistical profiler baseline.

Derives a data-quality suite directly from *clean* data, with no LLM. By
construction these expectations should not fire on the clean data they were
profiled from (near-zero false-positive rate), giving a strong non-LLM baseline
for RQ2.

Derived expectation kinds:

* nullability — columns with no nulls get ``not_be_null``,
* numeric range — ``values_to_be_between`` at the observed [min, max],
* uniqueness — fully-unique columns get ``values_to_be_unique``,
* categories — low-cardinality non-float columns get ``values_to_be_in_set``,
* string length — object columns get ``value_lengths_to_be_between``.

Rules are emitted in the same ``{expectation_type, kwargs}`` dict format the LLM
uses, so downstream parsing/validation treats them identically.
"""

from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from dqgen import ROW_ID
from dqgen.expectations import InvalidKwargsError, UnknownExpectationError, build_expectation


def profile_rules(
    df: pd.DataFrame, *, max_categories: int = 20, row_id: str = ROW_ID
) -> list[dict]:
    """Derive constraint rule dicts from a (clean) dataframe."""
    rules: list[dict] = []
    for col in df.columns:
        if col == row_id:
            continue
        s = df[col]
        note = "profiled from clean data"

        if s.notna().all():
            rules.append(_rule("expect_column_values_to_not_be_null", {"column": col}, note))

        if pdt.is_numeric_dtype(s) and s.notna().any():
            rules.append(
                _rule(
                    "expect_column_values_to_be_between",
                    {"column": col, "min_value": float(s.min()), "max_value": float(s.max())},
                    note,
                )
            )

        if s.is_unique:
            rules.append(_rule("expect_column_values_to_be_unique", {"column": col}, note))

        nunique = int(s.nunique(dropna=True))
        if not pdt.is_float_dtype(s) and 0 < nunique <= max_categories:
            value_set = sorted(v.item() if hasattr(v, "item") else v for v in s.dropna().unique())
            rules.append(
                _rule("expect_column_values_to_be_in_set", {"column": col, "value_set": value_set}, note)
            )

        if pdt.is_object_dtype(s) or pdt.is_string_dtype(s):
            lengths = s.dropna().astype(str).str.len()
            if len(lengths):
                rules.append(
                    _rule(
                        "expect_column_value_lengths_to_be_between",
                        {"column": col, "min_value": int(lengths.min()), "max_value": int(lengths.max())},
                        note,
                    )
                )

    return rules


def build_suite_from_rules(rules: list[dict]) -> list:
    """Instantiate GX expectations from rule dicts, skipping any that fail.

    The statistical profiler's own rules are always valid; this helper is shared
    with the human baseline loader, which may contain mistakes worth skipping.
    """
    suite = []
    for r in rules:
        try:
            suite.append(build_expectation(r["expectation_type"], r.get("kwargs", {})))
        except (UnknownExpectationError, InvalidKwargsError):
            continue
    return suite


def profile_suite(df: pd.DataFrame, *, max_categories: int = 20) -> list:
    """Convenience: derive rules from ``df`` and build the GX suite."""
    return build_suite_from_rules(profile_rules(df, max_categories=max_categories))


def _rule(expectation_type: str, kwargs: dict, rationale: str) -> dict:
    return {"expectation_type": expectation_type, "kwargs": kwargs, "rationale": rationale}
