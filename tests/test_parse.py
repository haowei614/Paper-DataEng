"""Tests for parsing/classifying LLM constraint output."""

from __future__ import annotations

import json

from dqgen.parse import (
    INVALID_KWARGS,
    JSON_ERROR,
    UNKNOWN_EXPECTATION,
    VALID,
    parse_response,
)

_GOOD_RULE = {
    "expectation_type": "expect_column_values_to_not_be_null",
    "kwargs": {"column": "amount"},
    "rationale": "required",
}


def test_valid_array_all_valid():
    text = json.dumps([_GOOD_RULE, _GOOD_RULE])
    res = parse_response(text)
    assert res.json_ok
    assert len(res.rules) == 2
    assert all(r.status == VALID for r in res.rules)
    assert len(res.valid_rules) == 2


def test_fenced_json_is_extracted():
    text = "Here you go:\n```json\n" + json.dumps([_GOOD_RULE]) + "\n```\nDone."
    res = parse_response(text)
    assert res.json_ok and res.rules[0].status == VALID


def test_prose_around_array_is_extracted():
    text = "Sure! " + json.dumps([_GOOD_RULE]) + " Hope this helps."
    res = parse_response(text)
    assert res.json_ok and res.rules[0].status == VALID


def test_unknown_expectation_type():
    text = json.dumps([{"expectation_type": "expect_nonsense", "kwargs": {}}])
    res = parse_response(text)
    assert res.rules[0].status == UNKNOWN_EXPECTATION


def test_invalid_kwargs():
    text = json.dumps(
        [{"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"bad": 1}}]
    )
    res = parse_response(text)
    assert res.rules[0].status == INVALID_KWARGS


def test_kwargs_not_object():
    text = json.dumps(
        [{"expectation_type": "expect_column_values_to_not_be_null", "kwargs": "oops"}]
    )
    res = parse_response(text)
    assert res.rules[0].status == INVALID_KWARGS


def test_non_dict_element_is_json_error():
    text = json.dumps(["not a rule", _GOOD_RULE])
    res = parse_response(text)
    assert res.rules[0].status == JSON_ERROR
    assert res.rules[1].status == VALID


def test_missing_expectation_type_is_json_error():
    text = json.dumps([{"kwargs": {"column": "x"}}])
    res = parse_response(text)
    assert res.rules[0].status == JSON_ERROR


def test_completely_unparseable_response():
    res = parse_response("I cannot help with that.")
    assert not res.json_ok
    assert len(res.rules) == 1 and res.rules[0].status == JSON_ERROR


def test_counts_sum_to_total():
    text = json.dumps(
        [
            _GOOD_RULE,
            {"expectation_type": "expect_nonsense", "kwargs": {}},
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"bad": 1}},
        ]
    )
    res = parse_response(text)
    counts = res.counts()
    assert counts[VALID] == 1
    assert counts[UNKNOWN_EXPECTATION] == 1
    assert counts[INVALID_KWARGS] == 1
    assert sum(counts.values()) == len(res.rules)
