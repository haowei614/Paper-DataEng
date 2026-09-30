"""Tests for the expectation allowlist and builder."""

from __future__ import annotations

import pytest

from dqgen.expectations import (
    ALLOWED_EXPECTATIONS,
    InvalidKwargsError,
    UnknownExpectationError,
    build_expectation,
    is_allowed,
    render_allowlist,
)


def test_allowlist_nonempty_and_all_resolve():
    # Import-time _validate_registry already checked resolution; sanity here.
    assert len(ALLOWED_EXPECTATIONS) == 21
    assert is_allowed("expect_column_values_to_not_be_null")
    assert not is_allowed("expect_something_made_up")


def test_build_valid_expectation():
    exp = build_expectation(
        "expect_column_values_to_be_between",
        {"column": "amount", "min_value": 0, "max_value": 100},
    )
    assert exp.expectation_type == "expect_column_values_to_be_between"


def test_build_unknown_type_raises():
    with pytest.raises(UnknownExpectationError):
        build_expectation("expect_made_up_thing", {"column": "x"})


def test_build_invalid_kwargs_raises():
    # Missing required 'column'.
    with pytest.raises(InvalidKwargsError):
        build_expectation("expect_column_values_to_not_be_null", {"wrong_arg": "x"})


def test_render_allowlist_contains_types():
    text = render_allowlist()
    assert "expect_column_values_to_be_in_set" in text
    assert "kwargs:" in text
