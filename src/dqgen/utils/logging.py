"""Logging helpers: console setup and a JSONL event logger."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root console logging (idempotent)."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class JsonlLogger:
    """Append structured events to a JSON Lines file.

    Each :meth:`log` call writes one JSON object (plus an ISO-8601 ``ts`` field)
    on its own line and flushes, so partial runs remain readable.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **record) -> None:
        record.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
