"""Parse LLM output into GX expectations, classifying every rule.

Each rule in the model's JSON array is classified into exactly one status so
that no failure is silently dropped (required for RQ1):

* ``valid`` — parsed and instantiated as a GX expectation.
* ``json_error`` — the response was not a JSON array, or an element was not a
  well-formed rule object.
* ``unknown_expectation`` — the ``expectation_type`` is not in the allowlist.
* ``invalid_kwargs`` — the type is allowed but kwargs failed instantiation.
* ``runtime_error`` — reserved: assigned later by :mod:`dqgen.validate` when a
  valid expectation raises while executing against data.

If the whole response fails to parse as a JSON array, :attr:`ParseResult.json_ok`
is ``False`` and a single synthetic ``json_error`` rule is recorded so the
failure is still counted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from dqgen.expectations import (
    InvalidKwargsError,
    UnknownExpectationError,
    build_expectation,
)

VALID = "valid"
JSON_ERROR = "json_error"
UNKNOWN_EXPECTATION = "unknown_expectation"
INVALID_KWARGS = "invalid_kwargs"
RUNTIME_ERROR = "runtime_error"


@dataclass
class ParsedRule:
    """One rule from the model output plus its classification."""

    index: int
    status: str
    expectation_type: str | None = None
    kwargs: dict = field(default_factory=dict)
    rationale: str | None = None
    error: str | None = None
    expectation: object | None = None  # GX expectation when status == valid

    def to_record(self) -> dict:
        return {
            "index": self.index,
            "status": self.status,
            "expectation_type": self.expectation_type,
            "kwargs": self.kwargs,
            "rationale": self.rationale,
            "error": self.error,
        }


@dataclass
class ParseResult:
    """All parsed rules for one response."""

    rules: list[ParsedRule]
    json_ok: bool
    raw: str

    @property
    def valid_rules(self) -> list[ParsedRule]:
        return [r for r in self.rules if r.status == VALID]

    def counts(self) -> dict[str, int]:
        out = {
            VALID: 0,
            JSON_ERROR: 0,
            UNKNOWN_EXPECTATION: 0,
            INVALID_KWARGS: 0,
            RUNTIME_ERROR: 0,
        }
        for r in self.rules:
            out[r.status] = out.get(r.status, 0) + 1
        return out


def _extract_json_array(text: str) -> str | None:
    """Best-effort extraction of the JSON array from a model response.

    Handles responses wrapped in ```json fences or surrounded by prose by
    taking the substring from the first ``[`` to the last ``]``.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("[")
    end = candidate.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    return candidate[start : end + 1]


def parse_response(text: str) -> ParseResult:
    """Parse a raw model response into classified rules."""
    array_text = _extract_json_array(text)
    if array_text is None:
        return ParseResult(
            rules=[ParsedRule(0, JSON_ERROR, error="no JSON array found in response")],
            json_ok=False,
            raw=text,
        )
    try:
        data = json.loads(array_text)
    except json.JSONDecodeError as exc:
        return ParseResult(
            rules=[ParsedRule(0, JSON_ERROR, error=f"JSON decode error: {exc}")],
            json_ok=False,
            raw=text,
        )
    if not isinstance(data, list):
        return ParseResult(
            rules=[ParsedRule(0, JSON_ERROR, error="top-level JSON is not an array")],
            json_ok=False,
            raw=text,
        )

    rules: list[ParsedRule] = []
    for i, item in enumerate(data):
        rules.append(_classify_item(i, item))
    return ParseResult(rules=rules, json_ok=True, raw=text)


def _classify_item(index: int, item) -> ParsedRule:
    if not isinstance(item, dict):
        return ParsedRule(index, JSON_ERROR, error="rule is not a JSON object")

    etype = item.get("expectation_type")
    kwargs = item.get("kwargs", {})
    rationale = item.get("rationale")

    if not isinstance(etype, str) or not etype:
        return ParsedRule(
            index, JSON_ERROR, expectation_type=etype if isinstance(etype, str) else None,
            kwargs=kwargs if isinstance(kwargs, dict) else {}, rationale=rationale,
            error="missing or non-string expectation_type",
        )
    if not isinstance(kwargs, dict):
        return ParsedRule(
            index, INVALID_KWARGS, expectation_type=etype, rationale=rationale,
            error="kwargs is not an object",
        )

    try:
        expectation = build_expectation(etype, kwargs)
    except UnknownExpectationError:
        return ParsedRule(
            index, UNKNOWN_EXPECTATION, expectation_type=etype, kwargs=kwargs,
            rationale=rationale, error="expectation_type not in allowlist",
        )
    except InvalidKwargsError as exc:
        return ParsedRule(
            index, INVALID_KWARGS, expectation_type=etype, kwargs=kwargs,
            rationale=rationale, error=str(exc),
        )

    return ParsedRule(
        index, VALID, expectation_type=etype, kwargs=kwargs, rationale=rationale,
        expectation=expectation,
    )
