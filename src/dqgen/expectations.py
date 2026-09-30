"""Canonical allowlist of Great Expectations 1.x expectation types.

This module is the single source of truth for which expectations the LLM may
emit. It feeds both:

* the **prompt** (via :func:`render_allowlist`), which lists the allowed types
  and their key arguments, and
* **parsing/validation** (via :func:`build_expectation`), which rejects any
  type outside the allowlist.

At import time every listed type is resolved against the GX registry, so a typo
fails fast rather than silently disabling a rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from great_expectations.expectations.registry import get_expectation_impl


class UnknownExpectationError(Exception):
    """Raised when a type string is not in the allowlist."""


class InvalidKwargsError(Exception):
    """Raised when an allowed type cannot be instantiated with given kwargs."""


@dataclass(frozen=True)
class ExpectationSpec:
    """Prompt-facing description of one allowed expectation type."""

    type: str
    kwargs: tuple[str, ...]
    description: str


# Ordered allowlist. ``kwargs`` lists the arguments the model should provide
# (required first); ``description`` is shown in the prompt.
_ALLOWED: tuple[ExpectationSpec, ...] = (
    ExpectationSpec("expect_column_values_to_not_be_null", ("column",),
                    "Column has no null/missing values."),
    ExpectationSpec("expect_column_values_to_be_between", ("column", "min_value", "max_value"),
                    "Numeric column values fall within [min_value, max_value]."),
    ExpectationSpec("expect_column_min_to_be_between", ("column", "min_value", "max_value"),
                    "The column minimum falls within [min_value, max_value]."),
    ExpectationSpec("expect_column_max_to_be_between", ("column", "min_value", "max_value"),
                    "The column maximum falls within [min_value, max_value]."),
    ExpectationSpec("expect_column_mean_to_be_between", ("column", "min_value", "max_value"),
                    "The column mean falls within [min_value, max_value]."),
    ExpectationSpec("expect_column_values_to_be_in_set", ("column", "value_set"),
                    "Every value is one of the allowed values in value_set."),
    ExpectationSpec("expect_column_values_to_not_be_in_set", ("column", "value_set"),
                    "No value is in the forbidden value_set."),
    ExpectationSpec("expect_column_distinct_values_to_be_in_set", ("column", "value_set"),
                    "The set of distinct values is a subset of value_set."),
    ExpectationSpec("expect_column_values_to_be_of_type", ("column", "type_"),
                    "Column values have the given type (e.g. 'float64', 'int64', 'object')."),
    ExpectationSpec("expect_column_values_to_be_in_type_list", ("column", "type_list"),
                    "Column values have one of the types in type_list."),
    ExpectationSpec("expect_column_values_to_match_regex", ("column", "regex"),
                    "String values match the given regular expression."),
    ExpectationSpec("expect_column_values_to_match_like_pattern", ("column", "like_pattern"),
                    "String values match the given SQL LIKE pattern."),
    ExpectationSpec("expect_column_values_to_match_strftime_format", ("column", "strftime_format"),
                    "String values parse under the given strftime format (e.g. '%Y-%m-%d')."),
    ExpectationSpec("expect_column_value_lengths_to_be_between", ("column", "min_value", "max_value"),
                    "String value lengths fall within [min_value, max_value]."),
    ExpectationSpec("expect_column_values_to_be_unique", ("column",),
                    "Column values are unique."),
    ExpectationSpec("expect_compound_columns_to_be_unique", ("column_list",),
                    "The combination of the listed columns is unique per row."),
    ExpectationSpec("expect_select_column_values_to_be_unique_within_record", ("column_list",),
                    "Within each row, the listed columns have distinct values."),
    ExpectationSpec("expect_column_pair_values_a_to_be_greater_than_b",
                    ("column_A", "column_B", "or_equal"),
                    "column_A >= column_B (or > when or_equal is false)."),
    ExpectationSpec("expect_column_pair_values_to_be_equal", ("column_A", "column_B"),
                    "column_A equals column_B row-wise."),
    ExpectationSpec("expect_multicolumn_sum_to_equal", ("column_list", "sum_total"),
                    "The row-wise sum of the listed columns equals sum_total."),
    ExpectationSpec("expect_table_row_count_to_be_between", ("min_value", "max_value"),
                    "The table row count falls within [min_value, max_value]."),
)

ALLOWED_SPECS: dict[str, ExpectationSpec] = {s.type: s for s in _ALLOWED}
ALLOWED_EXPECTATIONS: list[str] = list(ALLOWED_SPECS)


def _validate_registry() -> None:
    """Ensure every allowlisted type resolves to a real GX expectation class."""
    missing = []
    for t in ALLOWED_EXPECTATIONS:
        try:
            get_expectation_impl(t)
        except Exception:  # noqa: BLE001 - any resolution failure is fatal here
            missing.append(t)
    if missing:
        raise RuntimeError(f"allowlisted expectation types not found in GX registry: {missing}")


_validate_registry()


def is_allowed(expectation_type: str) -> bool:
    return expectation_type in ALLOWED_SPECS


def build_expectation(expectation_type: str, kwargs: dict):
    """Instantiate a GX expectation from a type string and kwargs.

    Raises:
        UnknownExpectationError: type not in the allowlist.
        InvalidKwargsError: type is allowed but kwargs are invalid.
    """
    if not is_allowed(expectation_type):
        raise UnknownExpectationError(expectation_type)
    impl = get_expectation_impl(expectation_type)
    try:
        return impl(**kwargs)
    except Exception as exc:  # noqa: BLE001 - surfaced as a classified parse failure
        raise InvalidKwargsError(f"{expectation_type}: {exc}") from exc


def render_allowlist() -> str:
    """Render the allowlist as a bullet list for inclusion in the prompt."""
    lines = []
    for s in _ALLOWED:
        args = ", ".join(s.kwargs)
        lines.append(f"- `{s.type}` (kwargs: {args}) — {s.description}")
    return "\n".join(lines)
