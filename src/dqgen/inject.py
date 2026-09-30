"""Error injection with ground-truth labels.

Each injector has the signature::

    injector(df, rate, seed, *, dataset="", column=..., rows=None, **kw)
        -> (corrupted_df, labels_df)

and corrupts a fraction ``rate`` of the rows of ``df`` (which must carry a
:data:`dqgen.ROW_ID` column). It returns a corrupted copy and a *labels*
dataframe with one row per injected error, using the schema
:data:`LABEL_COLUMNS`.

Guarantees (verified by unit tests):

* The set of corrupted / added ``_row_id`` values equals exactly the set of
  labelled ``_row_id`` values.
* Rows that were not selected are left byte-identical (value-identical for the
  type/format injector, which must change a column's dtype to ``object``).
* Two runs with the same ``seed`` produce identical output.

For controlled combinations, :func:`make_mixed` applies several injectors to
*disjoint* row subsets so every row carries at most one ground-truth error.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from dqgen import ROW_ID

logger = logging.getLogger(__name__)

LABEL_COLUMNS = [
    ROW_ID,
    "column",
    "error_type",
    "rate",
    "seed",
    "dataset",
    "original_value",
    "corrupted_value",
]

Injector = Callable[..., tuple[pd.DataFrame, pd.DataFrame]]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _n_to_corrupt(n: int, rate: float) -> int:
    """Number of rows to corrupt for a given rate (round-half-to-even)."""
    return int(round(n * rate))


def _select_positions(
    df: pd.DataFrame, rate: float, seed: int, rows: np.ndarray | None
) -> np.ndarray:
    """Return sorted integer positions of rows to corrupt.

    If ``rows`` (explicit positions) is given it is used verbatim; otherwise a
    random subset of size ``round(len(df) * rate)`` is drawn with ``seed``.
    """
    if rows is not None:
        return np.sort(np.asarray(rows, dtype=int))
    k = _n_to_corrupt(len(df), rate)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(df), size=k, replace=False))


def _as_label_value(v) -> str | None:
    """Normalize a cell value for the labels table (str or None)."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or (pd.isna(v) is True):
        return None
    return str(v)


def _make_labels(
    row_ids,
    column: str | None,
    error_type: str,
    rate: float,
    seed: int,
    dataset: str,
    originals,
    corrupteds,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ROW_ID: list(row_ids),
            "column": column,
            "error_type": error_type,
            "rate": rate,
            "seed": seed,
            "dataset": dataset,
            "original_value": [_as_label_value(v) for v in originals],
            "corrupted_value": [_as_label_value(v) for v in corrupteds],
        },
        columns=LABEL_COLUMNS,
    )


def _require_row_id(df: pd.DataFrame) -> None:
    if ROW_ID not in df.columns:
        raise ValueError(f"dataframe must contain a '{ROW_ID}' column; call add_row_ids first")


# --------------------------------------------------------------------------- #
# Injectors
# --------------------------------------------------------------------------- #
def inject_missing_value(
    df: pd.DataFrame,
    rate: float,
    seed: int,
    *,
    column: str,
    dataset: str = "",
    rows: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set ``column`` to null in the selected rows."""
    _require_row_id(df)
    out = df.copy()
    # Integer columns cannot hold NaN; widen to float so nulls are representable.
    if pd.api.types.is_integer_dtype(out[column]):
        out[column] = out[column].astype("float64")
    pos = _select_positions(df, rate, seed, rows)
    originals = out[column].iloc[pos].tolist()
    out.iloc[pos, out.columns.get_loc(column)] = np.nan
    labels = _make_labels(
        out[ROW_ID].iloc[pos], column, "missing_value", rate, seed, dataset,
        originals, [None] * len(pos),
    )
    return out, labels


def inject_out_of_range(
    df: pd.DataFrame,
    rate: float,
    seed: int,
    *,
    column: str,
    dataset: str = "",
    mode: str = "high",
    rows: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Push ``column`` in selected rows far outside its observed numeric range.

    ``mode="high"`` sets values above the observed maximum; ``mode="low"`` sets
    them below the observed minimum. The offset is 100x the observed span, so
    the values are unambiguously out of range.
    """
    _require_row_id(df)
    out = df.copy()
    pos = _select_positions(df, rate, seed, rows)
    col = out[column]
    lo, hi = float(col.min()), float(col.max())
    span = (hi - lo) or 1.0
    new_value = hi + 100 * span if mode == "high" else lo - 100 * span
    originals = col.iloc[pos].tolist()
    out.iloc[pos, out.columns.get_loc(column)] = new_value
    labels = _make_labels(
        out[ROW_ID].iloc[pos], column, "out_of_range", rate, seed, dataset,
        originals, [new_value] * len(pos),
    )
    return out, labels


def inject_type_or_format_error(
    df: pd.DataFrame,
    rate: float,
    seed: int,
    *,
    column: str,
    dataset: str = "",
    bad_values: list[str] | None = None,
    rows: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write malformed tokens into ``column`` for selected rows.

    The column is cast to ``object`` so malformed strings coexist with the
    original typed values. Default tokens cover malformed numbers and dates.
    """
    _require_row_id(df)
    bad_values = bad_values or ["not_a_value", "2024-13-45", "??", "1e999x"]
    out = df.copy()
    out[column] = out[column].astype(object)
    pos = _select_positions(df, rate, seed, rows)
    rng = np.random.default_rng(seed + 1)
    picks = rng.integers(0, len(bad_values), size=len(pos))
    originals = out[column].iloc[pos].tolist()
    col_idx = out.columns.get_loc(column)
    corrupted = []
    for p, b in zip(pos, picks):
        out.iloc[p, col_idx] = bad_values[b]
        corrupted.append(bad_values[b])
    labels = _make_labels(
        out[ROW_ID].iloc[pos], column, "type_or_format_error", rate, seed, dataset,
        originals, corrupted,
    )
    return out, labels


def inject_invalid_category(
    df: pd.DataFrame,
    rate: float,
    seed: int,
    *,
    column: str,
    dataset: str = "",
    invalid_value=999,
    rows: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace ``column`` with a value outside its documented category set."""
    _require_row_id(df)
    out = df.copy()
    pos = _select_positions(df, rate, seed, rows)
    originals = out[column].iloc[pos].tolist()
    # Cast to object if the sentinel is incompatible with the column dtype.
    if isinstance(invalid_value, str) and out[column].dtype != object:
        out[column] = out[column].astype(object)
    out.iloc[pos, out.columns.get_loc(column)] = invalid_value
    labels = _make_labels(
        out[ROW_ID].iloc[pos], column, "invalid_category", rate, seed, dataset,
        originals, [invalid_value] * len(pos),
    )
    return out, labels


def inject_duplicate_row(
    df: pd.DataFrame,
    rate: float,
    seed: int,
    *,
    dataset: str = "",
    column: str | None = None,
    rows: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append exact duplicates of selected rows with fresh unique row ids.

    The original rows are untouched. Each appended duplicate copies all
    business columns from its source row but receives a new, unique
    :data:`ROW_ID` (so ``_row_id`` remains a valid GX index). Labels mark the
    appended duplicate rows.
    """
    _require_row_id(df)
    pos = _select_positions(df, rate, seed, rows)
    dupes = df.iloc[pos].copy()
    next_id = int(df[ROW_ID].max()) + 1
    new_ids = list(range(next_id, next_id + len(dupes)))
    dupes[ROW_ID] = new_ids
    out = pd.concat([df, dupes], ignore_index=True)
    labels = _make_labels(
        new_ids, column, "duplicate_row", rate, seed, dataset,
        df[ROW_ID].iloc[pos].tolist(),  # source row id, for traceability
        [None] * len(new_ids),
    )
    return out, labels


def inject_referential_integrity(
    df: pd.DataFrame,
    rate: float,
    seed: int,
    *,
    column: str,
    dataset: str = "",
    valid_keys: set | None = None,
    rows: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Break a foreign key by pointing selected rows at a non-existent parent.

    The FK value in ``column`` is replaced with a negative sentinel guaranteed
    absent from any TPC-H key space. If ``valid_keys`` is supplied it is used to
    assert the sentinel is truly absent.
    """
    _require_row_id(df)
    out = df.copy()
    pos = _select_positions(df, rate, seed, rows)
    originals = out[column].iloc[pos].tolist()
    # Distinct negative sentinels, all guaranteed outside positive TPC-H keys.
    corrupted = [-(i + 1) for i in range(len(pos))]
    if valid_keys is not None:
        assert not (set(corrupted) & set(valid_keys)), "sentinel collided with valid key"
    col_idx = out.columns.get_loc(column)
    for p, c in zip(pos, corrupted):
        out.iloc[p, col_idx] = c
    labels = _make_labels(
        out[ROW_ID].iloc[pos], column, "referential_integrity", rate, seed, dataset,
        originals, corrupted,
    )
    return out, labels


def inject_cross_column_logic(
    df: pd.DataFrame,
    rate: float,
    seed: int,
    *,
    dataset: str = "",
    kind: str = "temporal",
    start_col: str | None = None,
    end_col: str | None = None,
    total_col: str | None = None,
    component_cols: list[str] | None = None,
    rows: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Violate a cross-column invariant.

    * ``kind="temporal"``: set ``end_col`` strictly before ``start_col`` (e.g.
      dropoff before pickup) by subtracting one hour from the start.
    * ``kind="sum"``: set ``total_col`` so it no longer equals the sum of
      ``component_cols`` (inflated by a fixed amount).

    The labelled column is the one whose value was changed (``end_col`` or
    ``total_col``).
    """
    _require_row_id(df)
    out = df.copy()
    pos = _select_positions(df, rate, seed, rows)

    if kind == "temporal":
        if not start_col or not end_col:
            raise ValueError("temporal mode requires start_col and end_col")
        col_idx = out.columns.get_loc(end_col)
        originals = out[end_col].iloc[pos].tolist()
        new_vals = (out[start_col].iloc[pos] - pd.Timedelta(hours=1)).tolist()
        for p, v in zip(pos, new_vals):
            out.iloc[p, col_idx] = v
        labels = _make_labels(
            out[ROW_ID].iloc[pos], end_col, "cross_column_logic", rate, seed, dataset,
            originals, new_vals,
        )
        return out, labels

    if kind == "sum":
        if not total_col or not component_cols:
            raise ValueError("sum mode requires total_col and component_cols")
        col_idx = out.columns.get_loc(total_col)
        correct = out[component_cols].iloc[pos].sum(axis=1)
        originals = out[total_col].iloc[pos].tolist()
        new_vals = (correct + 100.0).tolist()  # deviate from the true sum
        for p, v in zip(pos, new_vals):
            out.iloc[p, col_idx] = v
        labels = _make_labels(
            out[ROW_ID].iloc[pos], total_col, "cross_column_logic", rate, seed, dataset,
            originals, new_vals,
        )
        return out, labels

    raise ValueError(f"unknown cross_column_logic kind: {kind!r}")


def inject_distribution_shift(
    df: pd.DataFrame,
    rate: float,
    seed: int,
    *,
    column: str,
    dataset: str = "",
    scale: float = 3.0,
    shift: float = 0.0,
    rows: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale and/or shift ``column`` for a subset of rows.

    Individual shifted values may remain within the column's global range, so
    this error is primarily detectable via aggregate expectations (mean/min/max)
    rather than per-value checks. Labels still record the affected rows so
    row-level recall can be computed for any rule that does flag them.
    """
    _require_row_id(df)
    out = df.copy()
    pos = _select_positions(df, rate, seed, rows)
    col_idx = out.columns.get_loc(column)
    originals = out[column].iloc[pos].astype(float)
    new_vals = (originals * scale + shift).tolist()
    for p, v in zip(pos, new_vals):
        out.iloc[p, col_idx] = v
    labels = _make_labels(
        out[ROW_ID].iloc[pos], column, "distribution_shift", rate, seed, dataset,
        originals.tolist(), new_vals,
    )
    return out, labels


INJECTORS: dict[str, Injector] = {
    "missing_value": inject_missing_value,
    "out_of_range": inject_out_of_range,
    "type_or_format_error": inject_type_or_format_error,
    "invalid_category": inject_invalid_category,
    "duplicate_row": inject_duplicate_row,
    "referential_integrity": inject_referential_integrity,
    "cross_column_logic": inject_cross_column_logic,
    "distribution_shift": inject_distribution_shift,
}


# --------------------------------------------------------------------------- #
# Mixed dataset
# --------------------------------------------------------------------------- #
def make_mixed(
    df: pd.DataFrame,
    spec: list[dict],
    seed: int,
    *,
    dataset: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply several injectors to *disjoint* row subsets.

    Args:
        df: Clean base dataframe (must carry :data:`ROW_ID`).
        spec: List of injector specs, each a dict::

            {"injector": "missing_value", "rate": 0.02, "kwargs": {"column": "x"}}

            Row-appending injectors (``duplicate_row``) are applied last so they
            do not consume positions from the base.
        seed: Base seed; each spec entry gets a derived seed.

    Returns:
        The corrupted dataframe and the concatenation of all per-injector
        labels. Because subsets are disjoint, every base row carries at most one
        error type.
    """
    _require_row_id(df)
    rng = np.random.default_rng(seed)
    n = len(df)

    # Split entries into in-place (consume base positions) and appending.
    appending = {"duplicate_row"}
    inplace_specs = [s for s in spec if s["injector"] not in appending]
    append_specs = [s for s in spec if s["injector"] in appending]

    # Allocate disjoint position sets for in-place injectors.
    total_needed = sum(_n_to_corrupt(n, s["rate"]) for s in inplace_specs)
    if total_needed > n:
        raise ValueError(f"mixed spec needs {total_needed} rows but base has only {n}")
    pool = rng.permutation(n)
    cursor = 0

    out = df
    all_labels: list[pd.DataFrame] = []
    for i, s in enumerate(inplace_specs):
        k = _n_to_corrupt(n, s["rate"])
        pos = pool[cursor : cursor + k]
        cursor += k
        injector = INJECTORS[s["injector"]]
        out, labels = injector(
            out, s["rate"], seed + i + 1, dataset=dataset, rows=pos, **s.get("kwargs", {})
        )
        all_labels.append(labels)

    for j, s in enumerate(append_specs):
        injector = INJECTORS[s["injector"]]
        out, labels = injector(
            out, s["rate"], seed + 1000 + j, dataset=dataset, **s.get("kwargs", {})
        )
        all_labels.append(labels)

    combined = pd.concat(all_labels, ignore_index=True) if all_labels else pd.DataFrame(
        columns=LABEL_COLUMNS
    )
    return out, combined
