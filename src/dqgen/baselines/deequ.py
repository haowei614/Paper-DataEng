"""Optional PyDeequ ConstraintSuggestionRunner baseline (behind a flag).

PyDeequ requires a running Spark/JVM, so this baseline is disabled by default
and only imported when explicitly enabled. When available, it runs Deequ's
constraint suggestion and converts the suggestions to equivalent GX 1.x
expectations where a mapping exists.

Enable by installing ``pydeequ`` (plus a compatible Spark) and calling
:func:`suggest_rules` with ``enabled=True``.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Whether pydeequ (and its Spark dependency) can be imported."""
    try:
        import pydeequ  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def suggest_rules(df: pd.DataFrame, *, enabled: bool = False) -> list[dict]:
    """Return GX rule dicts derived from Deequ constraint suggestions.

    Args:
        df: The (clean) dataframe to profile.
        enabled: Must be True to actually run; otherwise returns [].

    Raises:
        RuntimeError: if enabled but pydeequ/Spark is unavailable.

    Note:
        This is a stub. The Spark session setup, ConstraintSuggestionRunner
        invocation, and suggestion→GX mapping are implemented only when the
        optional dependency is present; see project docs for enabling it.
    """
    if not enabled:
        logger.info("PyDeequ baseline disabled; returning no rules.")
        return []
    if not is_available():
        raise RuntimeError(
            "PyDeequ baseline requested but pydeequ/Spark is not available. "
            "Install pydeequ and a compatible Spark, or run without the deequ flag."
        )
    raise NotImplementedError(
        "PyDeequ suggestion→GX mapping is not yet implemented; contributions welcome."
    )
