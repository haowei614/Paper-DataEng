"""Run expectation suites against data and collect row-level outcomes.

For each expectation we request ``result_format=COMPLETE`` with
``unexpected_index_column_names=[ROW_ID]`` so Great Expectations returns the
:data:`~dqgen.ROW_ID` of every unexpected (flagged) row. Those ids align
directly with the injected-error labels, enabling row-level precision/recall.

Expectations that raise while executing are captured as ``runtime_error``
outcomes (never crashing the run), which is how the ``runtime_error`` status
from :mod:`dqgen.parse` is finally assigned for RQ1.
"""

from __future__ import annotations

import logging
import os

# Silence GX's per-expectation tqdm "Calculating Metrics" bars.
os.environ.setdefault("TQDM_DISABLE", "1")

from dataclasses import dataclass, field  # noqa: E402

import great_expectations as gx  # noqa: E402
import pandas as pd  # noqa: E402

from dqgen import ROW_ID  # noqa: E402

logger = logging.getLogger(__name__)

_RESULT_FORMAT = {"result_format": "COMPLETE", "unexpected_index_column_names": [ROW_ID]}


@dataclass
class ExpectationOutcome:
    """Result of running one expectation against one dataframe."""

    index: int
    expectation_type: str
    kwargs: dict
    success: bool | None  # None => runtime_error
    unexpected_row_ids: set = field(default_factory=set)
    unexpected_count: int | None = None
    element_count: int | None = None
    is_row_level: bool = False  # whether an unexpected_index_list was returned
    exception: str | None = None

    @property
    def runtime_error(self) -> bool:
        return self.success is None


@dataclass
class ValidationReport:
    """Outcomes for a whole suite run against one dataframe."""

    dataset: str
    n_rows: int
    outcomes: list[ExpectationOutcome]

    @property
    def flagged_row_ids(self) -> set:
        """Union of row ids flagged by any row-level expectation."""
        ids: set = set()
        for o in self.outcomes:
            ids |= o.unexpected_row_ids
        return ids

    @property
    def n_runtime_errors(self) -> int:
        return sum(1 for o in self.outcomes if o.runtime_error)

    @property
    def n_failed(self) -> int:
        """Number of expectations that ran and did not pass."""
        return sum(1 for o in self.outcomes if o.success is False)

    @property
    def dataset_detected(self) -> bool:
        """True if any expectation flagged a problem (failed on the dataset)."""
        return any(o.success is False for o in self.outcomes)


def _make_batch(df: pd.DataFrame):
    """Create a whole-dataframe GX batch in a fresh ephemeral context."""
    ctx = gx.get_context(mode="ephemeral")
    ds = ctx.data_sources.add_pandas("pandas_source")
    asset = ds.add_dataframe_asset("asset")
    batch_def = asset.add_batch_definition_whole_dataframe("batch_def")
    return batch_def.get_batch(batch_parameters={"dataframe": df})


def validate_suite(expectations: list, df: pd.DataFrame, dataset: str = "") -> ValidationReport:
    """Run every expectation in ``expectations`` against ``df``.

    Args:
        expectations: GX expectation objects (e.g. from a
            :class:`~dqgen.generate.GenerationResult` or a baseline suite).
        df: Dataframe carrying :data:`ROW_ID`.
        dataset: Label recorded on the report.

    Returns:
        A :class:`ValidationReport`. Individual expectation failures — including
        ones that raise — are captured per expectation.
    """
    if ROW_ID not in df.columns:
        raise ValueError(f"dataframe must contain '{ROW_ID}' for row-level validation")

    batch = _make_batch(df)
    outcomes: list[ExpectationOutcome] = []
    for i, exp in enumerate(expectations):
        etype = getattr(exp, "expectation_type", type(exp).__name__)
        kwargs = _expectation_kwargs(exp)
        try:
            res = batch.validate(exp, result_format=_RESULT_FORMAT)
        except Exception as exc:  # noqa: BLE001 - captured as runtime_error
            logger.debug("runtime error on expectation %s: %s", etype, exc)
            outcomes.append(
                ExpectationOutcome(i, etype, kwargs, success=None, exception=str(exc))
            )
            continue

        result = res.result or {}
        uil = result.get("unexpected_index_list")
        is_row_level = uil is not None
        row_ids = {row[ROW_ID] for row in uil if isinstance(row, dict) and ROW_ID in row} if uil else set()
        outcomes.append(
            ExpectationOutcome(
                index=i,
                expectation_type=etype,
                kwargs=kwargs,
                success=bool(res.success),
                unexpected_row_ids=row_ids,
                unexpected_count=result.get("unexpected_count"),
                element_count=result.get("element_count"),
                is_row_level=is_row_level,
            )
        )
    return ValidationReport(dataset=dataset, n_rows=len(df), outcomes=outcomes)


def _expectation_kwargs(exp) -> dict:
    """Best-effort extraction of an expectation's kwargs for logging."""
    try:
        return exp.configuration.kwargs  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return {}
