"""Typed experiment configuration loaded from YAML.

The whole experiment grid (models, datasets, conditions, injection rates/seeds,
repetitions, paths) is described in ``configs/experiment.yaml`` and validated
into these pydantic models. Nothing about the grid is hard-coded elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """One LLM entry in the roster.

    ``name`` is the label used in results/caches; ``model`` is the
    provider-specific model id. ``backend`` selects the client. For
    OpenAI-compatible backends (Ollama, vLLM, llama.cpp, OpenAI) set
    ``base_url``; ``api_key_env`` names the environment variable holding the key
    (optional for local servers).
    """

    name: str
    backend: Literal["openai", "anthropic"]
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: float = 600.0  # large local models can take minutes per suite

    def params(self) -> dict:
        """Parameters that participate in the response-cache key."""
        return {
            "backend": self.backend,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class DatasetSpec(BaseModel):
    """A dataset to run the experiment over."""

    name: str
    kind: Literal["nyc_taxi", "tpch"]
    # nyc_taxi: month + sample size; tpch: scale factor + tables.
    month: str = "2024-01"
    sample_rows: int = 100_000
    scale_factor: float = 0.1
    tables: list[str] = Field(default_factory=lambda: ["orders", "lineitem", "customer"])
    doc_path: str | None = None
    sample_seed: int = 42
    # Optional per-table row caps (tpch only). Tables larger than their cap are
    # uniformly downsampled with ``sample_seed`` after cleaning, before row ids
    # are assigned, so the clean data is reproducible from the pipeline alone.
    # E.g. {lineitem: 100000} keeps the medium-scope lineitem tractable.
    table_row_caps: dict[str, int] = Field(default_factory=dict)


class InjectionConfig(BaseModel):
    rates: list[float] = Field(default_factory=lambda: [0.01, 0.05])
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3])


class Paths(BaseModel):
    raw_dir: str = "data/raw"
    clean_dir: str = "data/clean"
    corrupted_dir: str = "data/corrupted"
    docs_dir: str = "docs"
    cache_dir: str = "cache/llm"
    results_dir: str = "results"
    logs_dir: str = "logs"


class ExperimentConfig(BaseModel):
    models: list[ModelSpec]
    datasets: list[DatasetSpec]
    conditions: list[Literal["a", "b", "c"]] = Field(default_factory=lambda: ["a", "b", "c"])
    repetitions: int = 3
    sample_rows_in_prompt: int = 10
    # Row-level detection on very large tables (e.g. TPC-H lineitem at 600k rows)
    # is dominated by GX's COMPLETE result-format materialization. To keep the
    # validate step tractable, tables larger than this cap are uniformly
    # subsampled (fixed seed) *for validation only* — generation still sees the
    # full clean schema/sample. Cross-row error types (duplicate_row, mixed) are
    # never subsampled, since their detection needs all rows present. Set to
    # null to disable subsampling and validate on full tables.
    validation_row_cap: int | None = 50_000
    validation_sample_seed: int = 20240101
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    paths: Paths = Field(default_factory=Paths)

    def model_by_name(self, name: str) -> ModelSpec:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"no model named {name!r} in config")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from YAML."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return ExperimentConfig.model_validate(data)
