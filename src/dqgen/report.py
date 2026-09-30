"""Summary tables (CSV + LaTeX) and figures from the tidy results table.

Reads ``results/results_long.parquet`` (produced by ``run validate``) and writes:

* ``results/tables/*.csv`` and ``*.tex`` — one file per summary table.
* ``results/figures/*.png`` — the paper figures.

Everything aggregates the long table by taking the mean of ``value`` over the
nuisance dimensions (seed, repetition, and — where noted — rate), grouped by the
dimensions each table/figure is about. Aggregation ignores NaN (undefined
ratios), matching the metric definitions in :mod:`dqgen.metrics`.

The three research figures:

* RQ2 — recall per error type, by method.
* RQ1/RQ4 — executability by model.
* RQ3 — condition a/b/c comparison (executability and detection recall).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never require a display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dqgen.config import ExperimentConfig  # noqa: E402

logger = logging.getLogger(__name__)


def load_results(cfg: ExperimentConfig) -> pd.DataFrame:
    path = Path(cfg.paths.results_dir) / "results_long.parquet"
    if not path.exists():
        raise FileNotFoundError(f"results table not found: {path}; run `validate` first")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _write_table(df: pd.DataFrame, out_dir: Path, name: str, caption: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv")
    try:
        latex = df.to_latex(float_format="%.3f", na_rep="--", caption=caption, label=f"tab:{name}")
    except Exception:  # noqa: BLE001 - older/newer pandas signature differences
        latex = df.to_latex(float_format="%.3f", na_rep="--")
    (out_dir / f"{name}.tex").write_text(latex, encoding="utf-8")
    logger.info("wrote table %s (%d rows)", name, len(df))


def table_executability(df: pd.DataFrame) -> pd.DataFrame:
    """RQ1/RQ4: mean executability and clean false-positive rates per model."""
    metrics = ["executability", "fp_rate_rules", "fp_rate_rows", "suite_size", "redundancy"]
    sub = df[df["metric"].isin(metrics)]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")
        .reindex(columns=metrics)
    )


def table_detection_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """RQ2: mean recall per error type, by method."""
    sub = df[df["metric"] == "recall_by_type"]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="error_type", columns="method", values="value", aggfunc="mean")


def table_detection_overall(df: pd.DataFrame) -> pd.DataFrame:
    """RQ2: overall precision/recall/F1 by method on single-error sweeps."""
    sub = df[(df["metric"].isin(["precision", "recall", "f1"])) & (df["error_type"] != "mixed")]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="method", columns="metric", values="value", aggfunc="mean").reindex(
        columns=["precision", "recall", "f1"]
    )


def table_condition(df: pd.DataFrame) -> pd.DataFrame:
    """RQ3: effect of input condition (a/b/c) for LLM methods."""
    sub = df[df["method"].str.startswith("llm:")]
    exe = sub[sub["metric"] == "executability"]
    rec = sub[sub["metric"] == "recall_by_type"]
    out = pd.DataFrame(
        {
            "executability": exe.groupby("condition")["value"].mean(),
            "recall": rec.groupby("condition")["value"].mean(),
        }
    )
    return out.reindex(index=["a", "b", "c"]).dropna(how="all")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save_fig(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)
    logger.info("wrote figure %s", name)


def fig_recall_by_type(df: pd.DataFrame, out_dir: Path) -> None:
    tbl = table_detection_by_type(df)
    if tbl.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(tbl)), 4))
    tbl.plot(kind="bar", ax=ax)
    ax.set_ylabel("recall")
    ax.set_xlabel("error type")
    ax.set_title("Detection recall per error type, by method")
    ax.set_ylim(0, 1)
    ax.legend(title="method", fontsize="small")
    _save_fig(fig, out_dir, "recall_by_error_type")


def fig_executability_by_model(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[(df["metric"] == "executability")]
    if sub.empty:
        return
    series = sub.groupby("model")["value"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(5, 1.0 * len(series)), 4))
    series.plot(kind="bar", ax=ax, color="tab:blue")
    ax.set_ylabel("executability rate")
    ax.set_xlabel("model")
    ax.set_title("Executability by model")
    ax.set_ylim(0, 1)
    _save_fig(fig, out_dir, "executability_by_model")


def fig_condition_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    tbl = table_condition(df)
    if tbl.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    tbl.plot(kind="bar", ax=ax)
    ax.set_ylabel("mean value")
    ax.set_xlabel("input condition")
    ax.set_title("Effect of input condition (a=schema, b=+docs, c=+samples)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize="small")
    _save_fig(fig, out_dir, "condition_comparison")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_report(cfg: ExperimentConfig) -> Path:
    df = load_results(cfg)
    results_dir = Path(cfg.paths.results_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"

    _write_table(table_executability(df), tables_dir, "executability_by_model",
                 "Executability and clean false-positive rates by model.")
    _write_table(table_detection_by_type(df), tables_dir, "recall_by_error_type",
                 "Detection recall per error type, by method.")
    _write_table(table_detection_overall(df), tables_dir, "detection_overall",
                 "Overall detection precision/recall/F1 by method.")
    _write_table(table_condition(df), tables_dir, "condition_comparison",
                 "Effect of input condition on LLM-generated suites.")

    fig_recall_by_type(df, figures_dir)
    fig_executability_by_model(df, figures_dir)
    fig_condition_comparison(df, figures_dir)

    logger.info("report step complete -> %s", results_dir)
    return results_dir
