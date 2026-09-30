"""Tests for prompt-context construction across conditions a/b/c."""

from __future__ import annotations

import pandas as pd
import pytest

from dqgen import ROW_ID
from dqgen.context import build_context, load_docs


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ROW_ID: range(5),
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0],
            "code": [1, 2, 3, 4, 5],
        }
    )


def test_condition_a_schema_only(df):
    ctx = build_context(dataset="d", table="d", df=df, condition="a")
    assert ctx.documentation is None
    assert ctx.sample_rows is None
    assert "amount" in ctx.schema and "code" in ctx.schema
    assert ROW_ID not in ctx.schema  # internal id never exposed
    assert ctx.n_rows == 5


def test_condition_b_includes_docs(df, tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text("# D\nColumn amount is the price.\n")
    ctx = build_context(dataset="d", table="d", df=df, condition="b", doc_path=doc)
    assert ctx.documentation is not None and "price" in ctx.documentation
    assert ctx.sample_rows is None


def test_condition_c_includes_sample(df, tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text("# D\ndocs\n")
    ctx = build_context(
        dataset="d", table="d", df=df, condition="c", doc_path=doc,
        n_sample_rows=3, sample_seed=1,
    )
    assert ctx.sample_rows is not None
    assert "amount" in ctx.sample_rows and ROW_ID not in ctx.sample_rows
    # 3 data rows + 1 header line.
    assert len(ctx.sample_rows.splitlines()) == 4


def test_condition_c_sample_is_deterministic(df, tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text("# D\n")
    kw = dict(dataset="d", table="d", df=df, condition="c", doc_path=doc, n_sample_rows=3)
    a = build_context(sample_seed=7, **kw)
    b = build_context(sample_seed=7, **kw)
    assert a.sample_rows == b.sample_rows


def test_condition_b_requires_docs(df):
    with pytest.raises(ValueError):
        build_context(dataset="d", table="d", df=df, condition="b")


def test_invalid_condition(df):
    with pytest.raises(ValueError):
        build_context(dataset="d", table="d", df=df, condition="z")


def test_load_docs_extracts_table_section(tmp_path):
    doc = tmp_path / "tpch.md"
    doc.write_text(
        "# TPC-H\nintro\n\n## orders\norders columns here\n\n## lineitem\nlineitem columns here\n"
    )
    orders = load_docs(doc, table="orders")
    assert "orders columns here" in orders
    assert "lineitem columns here" not in orders  # only the matching section
