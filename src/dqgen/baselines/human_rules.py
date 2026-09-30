"""Loader for hand-written human-baseline suites.

Human suites live as JSON arrays of ``{expectation_type, kwargs, rationale}``
objects under ``baselines/human/*.json`` (same format the LLM emits). They are a
reasonable first version derived from the data dictionaries and are intended to
be refined manually.
"""

from __future__ import annotations

import json
from pathlib import Path

from dqgen.baselines.stats_profiler import build_suite_from_rules

_HUMAN_DIR = Path(__file__).parent / "human"


def load_human_rules(name: str) -> list[dict]:
    """Load the raw rule dicts for ``name`` (e.g. ``"nyc_taxi"``, ``"tpch_orders"``)."""
    path = _HUMAN_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_human_suite(name: str) -> list:
    """Load and build the GX expectation suite for ``name``."""
    return build_suite_from_rules(load_human_rules(name))
