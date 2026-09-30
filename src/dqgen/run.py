"""Experiment grid runner and CLI.

Subcommands mirror the Makefile targets:

* ``data``     — download/generate, clean, sample datasets.
* ``inject``   — produce corrupted datasets + labels (per error type × rate ×
  seed, plus one mixed dataset) and a manifest.
* ``generate`` — LLM → constraint suites (cached), writing generation summaries.
* ``validate`` — run every suite (LLM + baselines) on clean + corrupted data and
  write the tidy long-format results table.
* ``report``   — summary tables (CSV + LaTeX) and figures.

Core logic lives in importable ``run_*`` functions; the typer commands are thin
wrappers so the pipeline can be driven programmatically and tested.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd
import typer

from dqgen import ROW_ID
from dqgen.baselines.human_rules import load_human_suite
from dqgen.baselines.stats_profiler import profile_suite
from dqgen.config import ExperimentConfig, ModelSpec, load_config
from dqgen.context import build_context
from dqgen.generate import GenerationResult, generate_suite
from dqgen.inject import INJECTORS, make_mixed
from dqgen.injection_plan import MIXED_RATE_PER_INJECTOR, error_type_for, plan_for
from dqgen.llm import CachingClient
from dqgen.metrics import (
    LONG_COLUMNS,
    detection_scores,
    executability_rate,
    long_row,
    redundancy,
    row_false_positive_rate,
    rule_false_positive_rate,
    suite_size,
)
from dqgen.utils.logging import configure_logging
from dqgen.validate import ValidationReport, validate_suite

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="dqgen experiment runner")

DEFAULT_CONFIG = "configs/experiment.yaml"

ClientFactory = Callable[[ModelSpec], CachingClient]


@dataclass
class TableRef:
    """A single table within a dataset, with its file/doc references."""

    dataset: str
    table_id: str        # used for files, plans, human suites (e.g. "tpch_orders")
    doc_table: str       # used for docs section + schema (e.g. "orders")
    clean_path: Path
    doc_path: str | None
    sample_seed: int


# --------------------------------------------------------------------------- #
# Table discovery
# --------------------------------------------------------------------------- #
def _table_refs(cfg: ExperimentConfig) -> list[TableRef]:
    clean_dir = Path(cfg.paths.clean_dir)
    refs: list[TableRef] = []
    for ds in cfg.datasets:
        if ds.kind == "nyc_taxi":
            refs.append(
                TableRef(ds.name, "nyc_taxi", "nyc_taxi",
                         clean_dir / "nyc_taxi.parquet", ds.doc_path, ds.sample_seed)
            )
        elif ds.kind == "tpch":
            for table in ds.tables:
                refs.append(
                    TableRef(ds.name, f"tpch_{table}", table,
                             clean_dir / f"tpch_{table}.parquet", ds.doc_path, ds.sample_seed)
                )
    return refs


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def run_data(cfg: ExperimentConfig) -> None:
    from dqgen.datasets import prepare_nyc_taxi, prepare_tpch

    for ds in cfg.datasets:
        if ds.kind == "nyc_taxi":
            prepare_nyc_taxi(
                month=ds.month, n=ds.sample_rows, seed=ds.sample_seed,
                raw_dir=Path(cfg.paths.raw_dir), clean_dir=Path(cfg.paths.clean_dir),
            )
        elif ds.kind == "tpch":
            prepare_tpch(
                sf=ds.scale_factor, raw_dir=Path(cfg.paths.raw_dir) / "tpch",
                clean_dir=Path(cfg.paths.clean_dir),
            )
    logger.info("data step complete")


# --------------------------------------------------------------------------- #
# inject
# --------------------------------------------------------------------------- #
def run_inject(cfg: ExperimentConfig) -> Path:
    """Produce corrupted datasets + labels for every table; write a manifest."""
    corrupted_dir = Path(cfg.paths.corrupted_dir)
    corrupted_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corrupted_dir / "manifest.jsonl"
    entries: list[dict] = []

    for tref in _table_refs(cfg):
        if not tref.clean_path.exists():
            logger.warning("clean data missing for %s (%s); run `data` first", tref.table_id, tref.clean_path)
            continue
        clean = pd.read_parquet(tref.clean_path)
        plan = plan_for(tref.table_id)

        # Single-error datasets: each injector × rate × seed.
        for spec in plan:
            injector = INJECTORS[spec["injector"]]
            etype = error_type_for(spec["injector"])
            for rate in cfg.injection.rates:
                for seed in cfg.injection.seeds:
                    corrupted, labels = injector(
                        clean, rate, seed, dataset=tref.table_id, **spec.get("kwargs", {})
                    )
                    entries.append(
                        _save_corrupted(corrupted_dir, tref.table_id, etype, rate, seed, corrupted, labels)
                    )

        # One mixed dataset per seed (disjoint subsets across all injectors).
        mixed_spec = [{"injector": s["injector"], "rate": MIXED_RATE_PER_INJECTOR,
                       "kwargs": s.get("kwargs", {})} for s in plan]
        for seed in cfg.injection.seeds:
            corrupted, labels = make_mixed(clean, mixed_spec, seed, dataset=tref.table_id)
            entries.append(
                _save_corrupted(corrupted_dir, tref.table_id, "mixed", None, seed, corrupted, labels)
            )

    with open(manifest_path, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    logger.info("inject step complete: %d corrupted datasets -> %s", len(entries), manifest_path)
    return manifest_path


def _save_corrupted(
    out_dir: Path, table_id: str, error_type: str, rate, seed: int,
    corrupted: pd.DataFrame, labels: pd.DataFrame,
) -> dict:
    tag = f"{table_id}__{error_type}" + (f"__r{rate}" if rate is not None else "") + f"__s{seed}"
    data_path = out_dir / f"{tag}.parquet"
    labels_path = out_dir / f"{tag}.labels.parquet"
    corrupted.to_parquet(data_path, index=False)
    labels.to_parquet(labels_path, index=False)
    return {
        "table_id": table_id, "error_type": error_type, "rate": rate, "seed": seed,
        "data_path": str(data_path), "labels_path": str(labels_path),
    }


def _load_manifest(cfg: ExperimentConfig) -> list[dict]:
    path = Path(cfg.paths.corrupted_dir) / "manifest.jsonl"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def _default_client_factory(cfg: ExperimentConfig) -> ClientFactory:
    cache_dir = Path(cfg.paths.cache_dir)
    return lambda spec: CachingClient(spec, cache_dir=cache_dir)


def _iter_llm_generations(
    cfg: ExperimentConfig, tref: TableRef, clean: pd.DataFrame, client_factory: ClientFactory
) -> Iterator[tuple[ModelSpec, str, int, GenerationResult]]:
    for spec in cfg.models:
        client = client_factory(spec)
        for condition in cfg.conditions:
            for rep in range(cfg.repetitions):
                ctx = build_context(
                    dataset=tref.dataset, table=tref.doc_table, df=clean, condition=condition,
                    doc_path=tref.doc_path, n_sample_rows=cfg.sample_rows_in_prompt,
                    sample_seed=tref.sample_seed,
                )
                gen = generate_suite(client, ctx, repetition=rep, dataset=tref.dataset, table=tref.table_id)
                yield spec, condition, rep, gen


def run_generate(cfg: ExperimentConfig, client_factory: ClientFactory | None = None) -> Path:
    """Generate all LLM suites (cached) and write generation summaries."""
    client_factory = client_factory or _default_client_factory(cfg)
    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for tref in _table_refs(cfg):
        if not tref.clean_path.exists():
            continue
        clean = pd.read_parquet(tref.clean_path)
        for spec, condition, rep, gen in _iter_llm_generations(cfg, tref, clean, client_factory):
            rows.append({"table_id": tref.table_id, **gen.summary()})
            logger.info("generated %s / %s / cond=%s / rep=%d: %d valid rules",
                        spec.name, tref.table_id, condition, rep, gen.summary()["n_valid"])
    out = results_dir / "generation.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    logger.info("generate step complete -> %s", out)
    return out


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
# Error types whose detection depends on cross-row context (a duplicated row is
# only detectable if its twin is also present), so they must never be
# subsampled. Everything else is per-row and safe to subsample uniformly.
CROSS_ROW_ERROR_TYPES = frozenset({"duplicate_row", "mixed"})


def _subsample(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Uniform fixed-seed subsample of a validation frame if it exceeds the cap.

    Returns the frame unchanged when subsampling is disabled or the frame is
    already within the cap. Row ids are preserved so injected-error labels stay
    alignable.
    """
    cap = cfg.validation_row_cap
    if cap is None or len(df) <= cap:
        return df
    return df.sample(n=cap, random_state=cfg.validation_sample_seed).reset_index(drop=True)


@dataclass
class SuiteRun:
    method: str
    condition: str
    repetition: int
    expectations: list
    gen: GenerationResult | None


def _iter_suites(
    cfg: ExperimentConfig, tref: TableRef, clean: pd.DataFrame,
    client_factory: ClientFactory, skip_llm: bool,
) -> Iterator[SuiteRun]:
    # Baselines (condition-independent).
    yield SuiteRun("stats", "baseline", 0, profile_suite(clean), None)
    try:
        yield SuiteRun("human", "baseline", 0, load_human_suite(tref.table_id), None)
    except FileNotFoundError:
        logger.info("no human suite for %s", tref.table_id)

    if skip_llm:
        return
    for spec, condition, rep, gen in _iter_llm_generations(cfg, tref, clean, client_factory):
        yield SuiteRun(f"llm:{spec.name}", condition, rep, gen.expectations, gen)


def _dims(sr: SuiteRun, tref: TableRef) -> dict:
    return {
        "model": sr.method, "dataset": tref.table_id,
        "condition": sr.condition, "repetition": sr.repetition, "method": sr.method,
    }


def _clean_metric_rows(sr: SuiteRun, tref: TableRef, clean_report: ValidationReport) -> list[dict]:
    d = _dims(sr, tref)
    rows = [
        long_row("suite_size", suite_size(clean_report), **d),
        long_row("redundancy", redundancy(sr.expectations), **d),
        long_row("fp_rate_rules", rule_false_positive_rate(clean_report), **d),
        long_row("fp_rate_rows", row_false_positive_rate(clean_report), **d),
        long_row("n_runtime_errors", clean_report.n_runtime_errors, **d),
    ]
    if sr.gen is not None:
        counts = sr.gen.parse_result.counts()
        rows.append(long_row("executability", executability_rate(counts, clean_report.n_runtime_errors), **d))
        rows.append(long_row("n_rules_total", len(sr.gen.parse_result.rules), **d))
        rows.append(long_row("n_valid", counts["valid"], **d))
        rows.append(long_row("n_json_error", counts["json_error"], **d))
        rows.append(long_row("n_unknown_expectation", counts["unknown_expectation"], **d))
        rows.append(long_row("n_invalid_kwargs", counts["invalid_kwargs"], **d))
        rows.append(long_row("prompt_tokens", sr.gen.response.prompt_tokens, **d))
        rows.append(long_row("completion_tokens", sr.gen.response.completion_tokens, **d))
        rows.append(long_row("latency_s", sr.gen.response.latency_s, **d))
    return rows


def _detection_rows(sr: SuiteRun, tref: TableRef, entry: dict, cfg: ExperimentConfig) -> list[dict]:
    corrupted = pd.read_parquet(entry["data_path"])
    labels = pd.read_parquet(entry["labels_path"])
    # Per-row error types may be subsampled for speed; cross-row ones must not.
    if entry["error_type"] not in CROSS_ROW_ERROR_TYPES:
        corrupted = _subsample(corrupted, cfg)
        labels = labels[labels[ROW_ID].isin(set(corrupted[ROW_ID]))]
    report = validate_suite(sr.expectations, corrupted, dataset=tref.table_id)
    scores = detection_scores(report.flagged_row_ids, labels)
    d = _dims(sr, tref)
    common = dict(rate=entry["rate"], seed=entry["seed"], **d)
    rows = [
        long_row("precision", scores.precision, error_type=entry["error_type"], **common),
        long_row("recall", scores.recall, error_type=entry["error_type"], **common),
        long_row("f1", scores.f1, error_type=entry["error_type"], **common),
        long_row("dataset_detected", int(report.dataset_detected), error_type=entry["error_type"], **common),
    ]
    # Per-error-type recall (informative for the mixed dataset).
    for etype, rec in scores.per_type_recall.items():
        rows.append(long_row("recall_by_type", rec, error_type=etype, **common))
    return rows


def run_validate(
    cfg: ExperimentConfig, client_factory: ClientFactory | None = None, skip_llm: bool = False
) -> Path:
    """Run all suites on clean + corrupted data; write the long results table."""
    client_factory = client_factory or _default_client_factory(cfg)
    manifest = _load_manifest(cfg)
    by_table: dict[str, list[dict]] = {}
    for e in manifest:
        by_table.setdefault(e["table_id"], []).append(e)

    all_rows: list[dict] = []
    for tref in _table_refs(cfg):
        if not tref.clean_path.exists():
            continue
        clean = pd.read_parquet(tref.clean_path)
        # Suites are built from the full clean table (schema, profiler, prompt
        # samples); clean false-positive checks run on the subsampled frame.
        clean_eval = _subsample(clean, cfg)
        for sr in _iter_suites(cfg, tref, clean, client_factory, skip_llm):
            clean_report = validate_suite(sr.expectations, clean_eval, dataset=tref.table_id)
            all_rows.extend(_clean_metric_rows(sr, tref, clean_report))
            for entry in by_table.get(tref.table_id, []):
                all_rows.extend(_detection_rows(sr, tref, entry, cfg))

    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows, columns=LONG_COLUMNS)
    df.to_parquet(results_dir / "results_long.parquet", index=False)
    df.to_csv(results_dir / "results_long.csv", index=False)
    logger.info("validate step complete: %d metric rows -> %s", len(df), results_dir)
    return results_dir / "results_long.parquet"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@app.command()
def data(config: str = DEFAULT_CONFIG):
    """Download/generate, clean, and sample datasets."""
    configure_logging()
    run_data(load_config(config))


@app.command()
def inject(config: str = DEFAULT_CONFIG):
    """Create corrupted datasets and labels."""
    configure_logging()
    run_inject(load_config(config))


@app.command()
def generate(config: str = DEFAULT_CONFIG):
    """Generate LLM constraint suites (cached)."""
    configure_logging()
    run_generate(load_config(config))


@app.command()
def validate(config: str = DEFAULT_CONFIG, skip_llm: bool = False):
    """Validate all suites and write the results table."""
    configure_logging()
    run_validate(load_config(config), skip_llm=skip_llm)


@app.command()
def report(config: str = DEFAULT_CONFIG):
    """Produce summary tables and figures."""
    configure_logging()
    from dqgen.report import run_report

    run_report(load_config(config))


if __name__ == "__main__":
    app()
