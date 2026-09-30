"""Build the prompt context for the three input conditions (RQ3).

Conditions:

* **a** — schema only (column names + inferred types).
* **b** — schema + documentation (the data dictionary).
* **c** — schema + documentation + a fixed sample of rows.

:func:`build_context` returns a :class:`PromptContext` of rendered text blocks;
the jinja2 prompt template (see ``prompts/``) assembles them. The internal
:data:`dqgen.ROW_ID` column is never exposed to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dqgen import ROW_ID

CONDITIONS = ("a", "b", "c")


@dataclass
class PromptContext:
    """Rendered context blocks for one (dataset, table, condition)."""

    dataset: str
    table: str
    condition: str
    schema: str
    documentation: str | None
    sample_rows: str | None
    n_rows: int


def _visible_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != ROW_ID]


def describe_schema(df: pd.DataFrame) -> str:
    """Render a column-name/type schema block (excluding the row-id column)."""
    lines = ["| column | type |", "| --- | --- |"]
    for col in _visible_columns(df):
        lines.append(f"| {col} | {df[col].dtype} |")
    return "\n".join(lines)


def format_sample_rows(df: pd.DataFrame, n: int, seed: int) -> str:
    """Render a fixed-seed sample of ``n`` rows as CSV (header + rows)."""
    visible = df[_visible_columns(df)]
    sample = visible.sample(n=min(n, len(visible)), random_state=seed)
    return sample.to_csv(index=False).strip()


def load_docs(doc_path: str | Path, table: str | None = None) -> str:
    """Load documentation markdown.

    If ``table`` is given, return only the section whose heading contains the
    table name (case-insensitive); otherwise return the whole file. Falls back
    to the full document if no matching section is found.
    """
    text = Path(doc_path).read_text(encoding="utf-8")
    if table is None:
        return text
    section = _extract_section(text, table)
    return section or text


def _extract_section(markdown: str, table: str) -> str | None:
    """Return the markdown block under the first heading mentioning ``table``."""
    lines = markdown.splitlines()
    start = None
    heading_level = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") and table.lower() in line.lower():
            start = i
            heading_level = len(line) - len(line.lstrip("#"))  # number of leading '#'
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].lstrip()
        if stripped.startswith("#"):
            level = len(lines[j]) - len(lines[j].lstrip("#"))
            if level <= heading_level:
                end = j
                break
    return "\n".join(lines[start:end]).strip()


def build_context(
    *,
    dataset: str,
    table: str,
    df: pd.DataFrame,
    condition: str,
    doc_path: str | Path | None = None,
    n_sample_rows: int = 10,
    sample_seed: int = 0,
) -> PromptContext:
    """Assemble the context blocks appropriate to ``condition``.

    Args:
        dataset: Dataset name (e.g. ``"nyc_taxi"``).
        table: Table name (equals ``dataset`` for single-table datasets).
        df: The clean dataframe (carrying :data:`ROW_ID`).
        condition: One of ``"a"``, ``"b"``, ``"c"``.
        doc_path: Path to the data dictionary (required for b/c).
        n_sample_rows: Rows to include for condition c.
        sample_seed: Seed for the sample-row selection (condition c).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")

    schema = describe_schema(df)
    documentation = None
    sample_rows = None

    if condition in ("b", "c"):
        if doc_path is None:
            raise ValueError(f"condition {condition!r} requires doc_path")
        documentation = load_docs(doc_path, table=table)
    if condition == "c":
        sample_rows = format_sample_rows(df, n_sample_rows, sample_seed)

    return PromptContext(
        dataset=dataset,
        table=table,
        condition=condition,
        schema=schema,
        documentation=documentation,
        sample_rows=sample_rows,
        n_rows=len(df),
    )
