"""Metrics for constraint suites: executability, false positives, detection.

All computations are pure functions over :class:`~dqgen.validate.ValidationReport`
objects and injected-error label frames, plus helpers to emit the tidy
long-format results rows used by the report step.

Row-level detection uses the union of flagged :data:`~dqgen.ROW_ID` values from a
run on *corrupted* data versus the injected labels:

* overall precision / recall / F1 over "any injected error",
* per-error-type recall (a row of the same shape per ``error_type``).

Undefined ratios (empty denominators) are reported as ``NaN`` rather than a
misleading 0 or 1.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import pandas as pd

from dqgen import ROW_ID
from dqgen.validate import ValidationReport

# Tidy long-format schema (one metric value per row).
LONG_COLUMNS = [
    "model", "dataset", "condition", "repetition", "method",
    "error_type", "rate", "seed", "metric", "value",
]


def executability_rate(parse_counts: dict, n_runtime_errors: int) -> float:
    """Fraction of generated rules that are executable (RQ1).

    A rule is executable if it parsed as a valid GX expectation *and* did not
    raise at validation time. ``parse_counts`` is
    :meth:`~dqgen.parse.ParseResult.counts`.
    """
    total = sum(parse_counts.values())
    if total == 0:
        return float("nan")
    executable = parse_counts.get("valid", 0) - n_runtime_errors
    return executable / total


def rule_false_positive_rate(clean_report: ValidationReport) -> float:
    """Fraction of executed rules that fail on clean data."""
    n = len(clean_report.outcomes)
    if n == 0:
        return float("nan")
    return clean_report.n_failed / n


def row_false_positive_rate(clean_report: ValidationReport) -> float:
    """Fraction of clean rows flagged by any rule."""
    if clean_report.n_rows == 0:
        return float("nan")
    return len(clean_report.flagged_row_ids) / clean_report.n_rows


@dataclass
class DetectionScores:
    precision: float
    recall: float
    f1: float
    per_type_recall: dict[str, float]
    n_flagged: int
    n_injected: int
    true_positives: int


def _f1(precision: float, recall: float) -> float:
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        return float("nan") if (math.isnan(precision) or math.isnan(recall)) else 0.0
    return 2 * precision * recall / (precision + recall)


def detection_scores(flagged_row_ids: set, labels: pd.DataFrame) -> DetectionScores:
    """Row-level precision/recall/F1 and per-type recall vs. injected labels."""
    injected = set(labels[ROW_ID]) if len(labels) else set()
    tp = len(flagged_row_ids & injected)

    precision = tp / len(flagged_row_ids) if flagged_row_ids else float("nan")
    recall = tp / len(injected) if injected else float("nan")
    f1 = _f1(precision, recall)

    per_type: dict[str, float] = {}
    if len(labels):
        for etype, grp in labels.groupby("error_type"):
            type_ids = set(grp[ROW_ID])
            per_type[str(etype)] = (
                len(flagged_row_ids & type_ids) / len(type_ids) if type_ids else float("nan")
            )

    return DetectionScores(
        precision=precision,
        recall=recall,
        f1=f1,
        per_type_recall=per_type,
        n_flagged=len(flagged_row_ids),
        n_injected=len(injected),
        true_positives=tp,
    )


def suite_size(report: ValidationReport) -> int:
    """Number of executable expectations in the suite."""
    return len(report.outcomes)


def redundancy(expectations: list) -> int:
    """Count exact-duplicate expectations (same type + kwargs).

    Returns ``total - unique``; subsumption (e.g. one range inside another) is
    not attempted here.
    """
    seen = set()
    dupes = 0
    for exp in expectations:
        etype = getattr(exp, "expectation_type", type(exp).__name__)
        try:
            kwargs = exp.configuration.kwargs
        except Exception:  # noqa: BLE001
            kwargs = {}
        key = (etype, json.dumps(kwargs, sort_keys=True, default=str))
        if key in seen:
            dupes += 1
        else:
            seen.add(key)
    return dupes


def long_row(
    metric: str,
    value,
    *,
    model: str,
    dataset: str,
    condition: str,
    repetition: int,
    method: str,
    error_type: str = "overall",
    rate=None,
    seed=None,
) -> dict:
    """Build one tidy long-format results row."""
    return {
        "model": model,
        "dataset": dataset,
        "condition": condition,
        "repetition": repetition,
        "method": method,
        "error_type": error_type,
        "rate": rate,
        "seed": seed,
        "metric": metric,
        "value": value,
    }
