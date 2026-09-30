"""Dataset acquisition, cleaning, and sampling.

Two datasets are supported:

* **NYC TLC Yellow Taxi** — one month of trip records (parquet). We load the
  full month, remove rows that violate obvious integrity rules, then sample a
  fixed number of rows with a fixed seed. Cleaning happens *before* sampling so
  the sampled dataset has an exact, reproducible row count.
* **TPC-H** at a small scale factor, generated locally via DuckDB's ``tpch``
  extension. TPC-H data is clean by construction; cleaning steps are recorded
  as auditable no-op verifications.

Every returned dataframe carries a stable :data:`dqgen.ROW_ID` column so that
injected-error labels and Great Expectations' ``unexpected_index_list`` output
refer to the same identifiers, even after rows are dropped or duplicated.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from dqgen import ROW_ID

logger = logging.getLogger(__name__)

# Official NYC TLC trip-record parquet host.
NYC_TAXI_URL_TEMPLATE = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
)

TPCH_TABLES = ("orders", "lineitem", "customer")


@dataclass
class CleaningStep:
    """One auditable data-cleaning operation.

    Attributes:
        name: Short identifier for the rule.
        description: Human-readable statement of the rule being enforced.
        rule: The concrete predicate applied (as a string, for the audit log).
        rows_before: Row count before the step.
        rows_after: Row count after the step.
    """

    name: str
    description: str
    rule: str
    rows_before: int
    rows_after: int

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rows_removed"] = self.rows_removed
        return d


@dataclass
class CleaningReport:
    """Ordered record of all cleaning steps applied to a dataset."""

    dataset: str
    steps: list[CleaningStep] = field(default_factory=list)

    def add(
        self, name: str, description: str, rule: str, before: pd.DataFrame, after: pd.DataFrame
    ) -> None:
        step = CleaningStep(name, description, rule, len(before), len(after))
        logger.info(
            "[clean:%s] %s: %d -> %d (removed %d)",
            self.dataset,
            name,
            step.rows_before,
            step.rows_after,
            step.rows_removed,
        )
        self.steps.append(step)

    def to_records(self) -> list[dict]:
        return [{"dataset": self.dataset, **s.to_dict()} for s in self.steps]


def add_row_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a fresh 0..N-1 :data:`ROW_ID` column.

    Called on the *clean base* immediately before injection so that every
    corrupted variant inherits the same identifiers.
    """
    out = df.reset_index(drop=True).copy()
    out.insert(0, ROW_ID, range(len(out)))
    return out


# --------------------------------------------------------------------------- #
# NYC Yellow Taxi
# --------------------------------------------------------------------------- #
def download_nyc_taxi(month: str = "2024-01", dest_dir: Path = Path("data/raw")) -> Path:
    """Download one month of Yellow Taxi records if not already present.

    Args:
        month: ``YYYY-MM`` string, e.g. ``"2024-01"``.
        dest_dir: Directory to store the raw parquet.

    Returns:
        Path to the downloaded parquet file.
    """
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"yellow_tripdata_{month}.parquet"
    if dest.exists():
        logger.info("NYC taxi %s already downloaded at %s", month, dest)
        return dest

    url = NYC_TAXI_URL_TEMPLATE.format(month=month)
    logger.info("Downloading %s -> %s", url, dest)
    tmp = dest.with_suffix(".parquet.part")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def load_nyc_taxi(path: Path) -> pd.DataFrame:
    """Load the full month of Yellow Taxi records from parquet."""
    return pd.read_parquet(path)


def clean_nyc_taxi(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove rows violating obvious NYC taxi integrity rules.

    Each rule is documented and its row impact recorded in the returned
    :class:`CleaningReport`. Rules are intentionally conservative: they target
    only clearly invalid records, not statistical outliers.
    """
    report = CleaningReport("nyc_taxi")
    work = df.copy()

    def apply(name: str, desc: str, rule: str, mask: pd.Series) -> None:
        nonlocal work
        before = work
        work = work[mask].copy()
        report.add(name, desc, rule, before, work)

    # Timestamps must be present and dropoff strictly after pickup.
    apply(
        "valid_timestamps",
        "Pickup and dropoff timestamps present and dropoff after pickup.",
        "tpep_pickup_datetime.notna() & tpep_dropoff_datetime.notna() & "
        "(tpep_dropoff_datetime > tpep_pickup_datetime)",
        work["tpep_pickup_datetime"].notna()
        & work["tpep_dropoff_datetime"].notna()
        & (work["tpep_dropoff_datetime"] > work["tpep_pickup_datetime"]),
    )

    # Trip duration under 24 hours (longer values are data errors).
    duration_h = (
        work["tpep_dropoff_datetime"] - work["tpep_pickup_datetime"]
    ).dt.total_seconds() / 3600.0
    apply(
        "duration_under_24h",
        "Trip duration must be under 24 hours.",
        "0 < duration_hours <= 24",
        (duration_h > 0) & (duration_h <= 24),
    )

    apply(
        "passenger_count_valid",
        "Passenger count present and in [1, 8].",
        "1 <= passenger_count <= 8",
        work["passenger_count"].notna()
        & (work["passenger_count"] >= 1)
        & (work["passenger_count"] <= 8),
    )

    apply(
        "trip_distance_valid",
        "Trip distance non-negative and under 200 miles.",
        "0 <= trip_distance <= 200",
        (work["trip_distance"] >= 0) & (work["trip_distance"] <= 200),
    )

    apply(
        "fare_non_negative",
        "Fare and total amounts non-negative.",
        "fare_amount >= 0 & total_amount >= 0",
        (work["fare_amount"] >= 0) & (work["total_amount"] >= 0),
    )

    apply(
        "ratecode_valid",
        "RatecodeID is one of the six documented standard codes (1-6).",
        "RatecodeID in {1,2,3,4,5,6}",
        work["RatecodeID"].isin([1, 2, 3, 4, 5, 6]),
    )

    apply(
        "payment_type_valid",
        "payment_type is a documented code (1-6).",
        "payment_type in {1,2,3,4,5,6}",
        work["payment_type"].isin([1, 2, 3, 4, 5, 6]),
    )

    return work.reset_index(drop=True), report


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample ``n`` rows (or all, if fewer) with a fixed seed."""
    if len(df) <= n:
        logger.info("Dataset has %d <= %d rows; using all rows.", len(df), n)
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def prepare_nyc_taxi(
    month: str = "2024-01",
    n: int = 100_000,
    seed: int = 42,
    raw_dir: Path = Path("data/raw"),
    clean_dir: Path = Path("data/clean"),
) -> tuple[pd.DataFrame, CleaningReport]:
    """End-to-end: download -> load -> clean -> sample -> add row ids -> save.

    Returns the clean, sampled, row-id-bearing dataframe and its cleaning
    report. The dataframe is written to ``clean_dir/nyc_taxi.parquet`` and the
    report to ``clean_dir/nyc_taxi.cleaning.json``.
    """
    raw = download_nyc_taxi(month, raw_dir)
    full = load_nyc_taxi(raw)
    cleaned, report = clean_nyc_taxi(full)
    sampled = sample_rows(cleaned, n=n, seed=seed)
    result = add_row_ids(sampled)
    _save_clean(result, report, "nyc_taxi", clean_dir)
    return result, report


# --------------------------------------------------------------------------- #
# TPC-H
# --------------------------------------------------------------------------- #
def generate_tpch(
    sf: float = 0.1, dest_dir: Path = Path("data/raw/tpch")
) -> dict[str, Path]:
    """Generate TPC-H tables at scale factor ``sf`` via DuckDB's tpch extension.

    Only :data:`TPCH_TABLES` are exported to parquet. Regenerates only if the
    expected files are missing.
    """
    import duckdb

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = {t: dest_dir / f"{t}.parquet" for t in TPCH_TABLES}
    if all(p.exists() for p in paths.values()):
        logger.info("TPC-H sf=%s already generated in %s", sf, dest_dir)
        return paths

    con = duckdb.connect()
    try:
        con.execute("INSTALL tpch; LOAD tpch;")
        con.execute("CALL dbgen(sf=?)", [sf])
        for table, path in paths.items():
            select_sql, casts = _tpch_select_with_casts(con, table)
            con.execute(f"COPY (SELECT {select_sql} FROM {table}) TO ? (FORMAT PARQUET)", [str(path)])
            logger.info("Exported TPC-H table %s -> %s (normalized: %s)",
                        table, path, ", ".join(casts) if casts else "none")
    finally:
        con.close()
    return paths


def _tpch_select_with_casts(con, table: str) -> tuple[str, list[str]]:
    """Build a SELECT list that normalizes TPC-H storage types for pandas/GX.

    DuckDB emits ``DECIMAL`` columns as Python ``Decimal`` objects and ``DATE``
    columns as ``date`` objects (both land as pandas ``object`` dtype), which
    breaks numeric injection, temporal logic, and Great Expectations' numeric
    checks. We cast money/measure ``DECIMAL`` columns to ``DOUBLE`` (float64) and
    ``DATE`` columns to ``TIMESTAMP`` (datetime64) so downstream code sees native
    numeric/temporal dtypes — consistent with the NYC taxi timestamps. Returns
    the SELECT expression and a human-readable list of the casts applied.
    """
    schema = con.execute(f"DESCRIBE {table}").fetchall()  # (name, type, ...)
    parts: list[str] = []
    casts: list[str] = []
    for name, coltype, *_ in schema:
        t = str(coltype).upper()
        if t.startswith("DECIMAL"):
            parts.append(f'CAST("{name}" AS DOUBLE) AS "{name}"')
            casts.append(f"{name}:{t}->DOUBLE")
        elif t == "DATE":
            parts.append(f'CAST("{name}" AS TIMESTAMP) AS "{name}"')
            casts.append(f"{name}:DATE->TIMESTAMP")
        else:
            parts.append(f'"{name}"')
    return ", ".join(parts), casts


def load_tpch(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Load TPC-H tables from parquet into a dict of dataframes."""
    return {name: pd.read_parquet(p) for name, p in paths.items()}


def clean_tpch(
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], CleaningReport]:
    """Verify TPC-H invariants (clean by construction) and record the checks.

    TPC-H generated data satisfies its schema constraints by construction, so
    these steps are verifications: they assert primary-key uniqueness and
    referential integrity and record the (expected zero) row impact. If a check
    ever removes rows, that is surfaced in the audit log.
    """
    report = CleaningReport("tpch")
    out = {name: t.copy() for name, t in tables.items()}

    # Primary-key uniqueness.
    for table, key in (("orders", "o_orderkey"), ("customer", "c_custkey")):
        if table in out and key in out[table].columns:
            before = out[table]
            after = before.drop_duplicates(subset=[key])
            report.add(
                f"{table}_pk_unique",
                f"{table}.{key} is a unique primary key.",
                f"drop_duplicates(subset=[{key}])",
                before,
                after,
            )
            out[table] = after

    # Referential integrity: lineitem.l_orderkey -> orders.o_orderkey,
    # orders.o_custkey -> customer.c_custkey.
    for child, ckey, parent, pkey in (
        ("lineitem", "l_orderkey", "orders", "o_orderkey"),
        ("orders", "o_custkey", "customer", "c_custkey"),
    ):
        if child in out and parent in out:
            before = out[child]
            valid_keys = set(out[parent][pkey])
            after = before[before[ckey].isin(valid_keys)]
            report.add(
                f"{child}_fk_{parent}",
                f"{child}.{ckey} references an existing {parent}.{pkey}.",
                f"{ckey} in set({parent}.{pkey})",
                before,
                after,
            )
            out[child] = after

    return {name: t.reset_index(drop=True) for name, t in out.items()}, report


def prepare_tpch(
    sf: float = 0.1,
    raw_dir: Path = Path("data/raw/tpch"),
    clean_dir: Path = Path("data/clean"),
) -> tuple[dict[str, pd.DataFrame], CleaningReport]:
    """End-to-end TPC-H: generate -> load -> clean -> add row ids -> save.

    Row ids are added per table. Each table is written to
    ``clean_dir/tpch_<table>.parquet``.
    """
    paths = generate_tpch(sf, raw_dir)
    tables = load_tpch(paths)
    cleaned, report = clean_tpch(tables)
    result = {name: add_row_ids(t) for name, t in cleaned.items()}
    clean_dir = Path(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    for name, t in result.items():
        t.to_parquet(clean_dir / f"tpch_{name}.parquet", index=False)
    _write_cleaning_report(report, clean_dir / "tpch.cleaning.json")
    return result, report


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _save_clean(
    df: pd.DataFrame, report: CleaningReport, name: str, clean_dir: Path
) -> None:
    clean_dir = Path(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(clean_dir / f"{name}.parquet", index=False)
    _write_cleaning_report(report, clean_dir / f"{name}.cleaning.json")


def _write_cleaning_report(report: CleaningReport, path: Path) -> None:
    import json

    with open(path, "w") as fh:
        json.dump(report.to_records(), fh, indent=2)
    logger.info("Wrote cleaning report -> %s", path)
