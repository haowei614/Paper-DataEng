"""Stable hashing for the LLM response cache."""

from __future__ import annotations

import hashlib
import json


def cache_key(model: str, prompt: str, params: dict) -> str:
    """Deterministic cache key for a (model, prompt, params) triple.

    ``params`` is serialized with sorted keys so logically-equal parameter sets
    map to the same key regardless of dict ordering.
    """
    payload = json.dumps(
        {"model": model, "prompt": prompt, "params": params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
