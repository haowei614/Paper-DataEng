"""Tests for prompt rendering and the generation pipeline (no network)."""

from __future__ import annotations

import json

import great_expectations as gx
import pandas as pd

from dqgen import ROW_ID
from dqgen.config import ModelSpec
from dqgen.context import build_context
from dqgen.generate import generate_suite, render_prompt
from dqgen.llm import CachingClient, LLMResponse


def _df():
    return pd.DataFrame({ROW_ID: range(3), "amount": [1.0, 2.0, 3.0], "code": [1, 2, 3]})


def test_render_prompt_includes_schema_and_allowlist():
    ctx = build_context(dataset="d", table="d", df=_df(), condition="a")
    prompt = render_prompt(ctx)
    assert "amount" in prompt
    assert "expect_column_values_to_be_in_set" in prompt  # allowlist present
    assert ROW_ID not in prompt


def test_render_prompt_condition_c_has_sample(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text("# D\ndocs here\n")
    ctx = build_context(
        dataset="d", table="d", df=_df(), condition="c", doc_path=doc, n_sample_rows=2
    )
    prompt = render_prompt(ctx)
    assert "Sample rows" in prompt and "docs here" in prompt


class _JsonBackend:
    model_id = "fake"

    def __init__(self, payload: str):
        self.payload = payload

    def complete(self, prompt, *, temperature, max_tokens):
        return LLMResponse(self.payload, self.model_id, 20, 8, 0.02)


def test_generate_suite_builds_expectations(tmp_path):
    payload = json.dumps(
        [
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "amount"}},
            {"expectation_type": "expect_column_values_to_be_between",
             "kwargs": {"column": "amount", "min_value": 0, "max_value": 10}},
            {"expectation_type": "expect_bogus", "kwargs": {}},
        ]
    )
    spec = ModelSpec(name="fake", backend="openai", model="fake", base_url="http://x/v1")
    client = CachingClient(spec, cache_dir=tmp_path, backend=_JsonBackend(payload))
    ctx = build_context(dataset="d", table="d", df=_df(), condition="a")

    result = generate_suite(client, ctx, repetition=0)
    summary = result.summary()
    assert summary["n_valid"] == 2
    assert summary["n_unknown_expectation"] == 1
    assert summary["prompt_tokens"] == 20

    suite = result.to_suite()
    assert isinstance(suite, gx.ExpectationSuite)
    assert len(suite.expectations) == 2
