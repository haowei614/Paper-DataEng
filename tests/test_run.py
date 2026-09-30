"""End-to-end orchestration test on synthetic taxi-shaped data.

Exercises the ``run_inject`` -> ``run_validate`` -> ``run_report`` pipeline
without any network access or LLM calls (``skip_llm=True`` runs only the
statistical and human baselines). Uses the real ``nyc_taxi`` injection plan so
column targeting is covered too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dqgen import ROW_ID
from dqgen.config import (
    DatasetSpec,
    ExperimentConfig,
    InjectionConfig,
    Paths,
)
from dqgen.report import run_report
from dqgen.run import _subsample, _table_refs, run_inject, run_validate


def _synthetic_taxi(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    pickup = pd.Timestamp("2024-01-15 08:00:00") + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")
    return pd.DataFrame(
        {
            ROW_ID: range(n),
            "VendorID": rng.integers(1, 3, n),
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + pd.to_timedelta(rng.integers(300, 1800, n), unit="s"),
            "passenger_count": rng.integers(1, 5, n).astype("int64"),
            "trip_distance": rng.uniform(0.5, 20.0, n),
            "RatecodeID": rng.integers(1, 6, n),
            "store_and_fwd_flag": rng.choice(["Y", "N"], n),
            "payment_type": rng.integers(1, 5, n),
            "fare_amount": rng.uniform(3.0, 80.0, n),
            "tip_amount": rng.uniform(0.0, 15.0, n),
            "total_amount": rng.uniform(3.0, 100.0, n),
        }
    )


def _make_config(tmp_path) -> ExperimentConfig:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    _synthetic_taxi().to_parquet(clean_dir / "nyc_taxi.parquet", index=False)
    return ExperimentConfig(
        models=[],  # skip_llm=True, so no models needed
        datasets=[DatasetSpec(name="taxi", kind="nyc_taxi", doc_path=None)],
        conditions=["a"],
        repetitions=1,
        injection=InjectionConfig(rates=[0.05], seeds=[1]),
        paths=Paths(
            raw_dir=str(tmp_path / "raw"),
            clean_dir=str(clean_dir),
            corrupted_dir=str(tmp_path / "corrupted"),
            cache_dir=str(tmp_path / "cache"),
            results_dir=str(tmp_path / "results"),
        ),
    )


def test_subsample_caps_and_is_deterministic(tmp_path):
    cfg = _make_config(tmp_path)
    cfg.validation_row_cap = 30
    cfg.validation_sample_seed = 7
    df = _synthetic_taxi(120)
    a = _subsample(df, cfg)
    b = _subsample(df, cfg)
    assert len(a) == 30
    assert a[ROW_ID].tolist() == b[ROW_ID].tolist()  # deterministic
    assert set(a[ROW_ID]).issubset(set(df[ROW_ID]))   # row ids preserved
    # Under the cap => returned unchanged.
    cfg.validation_row_cap = 1000
    assert len(_subsample(df, cfg)) == 120
    # Disabled => unchanged.
    cfg.validation_row_cap = None
    assert len(_subsample(df, cfg)) == 120


def test_table_refs_single_taxi(tmp_path):
    cfg = _make_config(tmp_path)
    refs = _table_refs(cfg)
    assert [r.table_id for r in refs] == ["nyc_taxi"]


def test_inject_writes_manifest_and_files(tmp_path):
    cfg = _make_config(tmp_path)
    manifest_path = run_inject(cfg)
    assert manifest_path.exists()
    lines = manifest_path.read_text().strip().splitlines()
    # 7 single-error injectors (rates=1, seeds=1) + 1 mixed dataset per seed.
    assert len(lines) == 8
    import json

    entries = [json.loads(x) for x in lines]
    assert {e["error_type"] for e in entries} >= {"missing_value", "out_of_range", "mixed"}
    for e in entries:
        assert e["data_path"].endswith(".parquet")
        assert pd.read_parquet(e["labels_path"]).shape[0] >= 0


def test_validate_baselines_only_produces_long_table(tmp_path):
    cfg = _make_config(tmp_path)
    run_inject(cfg)
    out = run_validate(cfg, skip_llm=True)
    assert out.exists()
    df = pd.read_parquet(out)
    assert not df.empty
    # Only baseline methods when LLM is skipped.
    assert set(df["method"].unique()) <= {"stats", "human"}
    # Detection + clean metrics are present.
    assert {"recall", "precision", "f1", "fp_rate_rows", "suite_size"} <= set(df["metric"].unique())
    # Stats profiler must not fire on clean data (no false positives).
    stats_fp = df[(df["method"] == "stats") & (df["metric"] == "fp_rate_rows")]["value"]
    assert (stats_fp == 0).all()


def test_report_writes_tables_and_figures(tmp_path):
    cfg = _make_config(tmp_path)
    run_inject(cfg)
    run_validate(cfg, skip_llm=True)
    results_dir = run_report(cfg)
    assert (results_dir / "tables" / "recall_by_error_type.csv").exists()
    assert (results_dir / "tables" / "detection_overall.tex").exists()
    assert (results_dir / "tables" / "executability_by_model.csv").exists()
    # Detection figure comes from baseline recall; LLM-only figures are skipped
    # when there are no LLM rows.
    assert (results_dir / "figures" / "recall_by_error_type.png").exists()
