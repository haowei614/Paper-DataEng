# dqgen — LLM-Assisted Generation of Data Quality Constraints

Reproducible experiment codebase for the paper *"LLM-Assisted Generation of
Data Quality Constraints from Schema and Documentation"*.

Given a table's schema, its data-dictionary documentation, and optionally a few
sample rows, an LLM is asked to emit a suite of [Great Expectations
1.x](https://docs.greatexpectations.io/) data-quality constraints. We then
measure how executable those constraints are, their false-positive rate on
clean data, and their ability to catch systematically injected errors —
comparing input conditions, models, and hand-written / statistical baselines.

## Research questions

| RQ  | Question |
| --- | -------- |
| RQ1 | What fraction of generated constraints are executable, and what is their false-positive rate on clean data? |
| RQ2 | Detection recall/precision on injected errors, per error type, vs. baselines. |
| RQ3 | Effect of input condition: (a) schema only, (b) schema + docs, (c) schema + docs + sample rows. |
| RQ4 | Local quantized models vs. API models. |

## Status

The full pipeline is implemented and unit-tested (122 tests):

- `dqgen.datasets` — download / generate, clean (auditable steps), sample.
- `dqgen.inject` — eight error injectors with ground-truth labels + mixed sets.
- `dqgen.context` — schema / docs / sample-row prompt contexts (conditions a/b/c).
- `dqgen.llm` — OpenAI-compatible + Anthropic backends behind a caching client.
- `dqgen.generate` / `dqgen.parse` / `dqgen.expectations` — prompt → parsed,
  allow-listed GX 1.x expectation suites (never silently dropping rules).
- `dqgen.baselines` — statistical profiler, hand-written human suites, optional
  PyDeequ (behind a flag).
- `dqgen.validate` / `dqgen.metrics` — row-level detection via GX unexpected
  indices vs. injected labels; executability, false positives, precision/recall.
- `dqgen.run` — grid orchestrator + `dqgen` CLI (`data`/`inject`/`generate`/
  `validate`/`report`), writing a corrupted-dataset manifest and a tidy
  long-format results table.
- `dqgen.report` — summary tables (CSV + LaTeX) and figures.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is provisioned
automatically.

```bash
uv sync              # create venv + install deps (Great Expectations pinned 1.23.2)
uv run pytest        # run the test suite
```

### Local models (RQ4 default roster)

The default config uses local [Ollama](https://ollama.com) models. Pull them
before running generation:

```bash
ollama pull qwen2.5-coder:32b
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
```

API models (Anthropic, OpenAI-compatible) are supported by the LLM backend and
can be added to `configs/experiment.yaml`; set the relevant API keys in the
environment first (`ANTHROPIC_API_KEY`, etc.).

## Reproduction (pipeline)

```bash
make data      # download NYC taxi + generate TPC-H, clean, sample
make inject    # produce corrupted datasets + labels (per type × rate × seed + mixed)
make generate  # LLM -> constraint suites (cached)
make validate  # run all suites on clean + corrupted data
make report    # summary tables (CSV + LaTeX) and figures
```

## Datasets

- **NYC TLC Yellow Taxi** — one month (default `2024-01`), 100k-row fixed-seed
  sample after cleaning. Data dictionary: [`docs/nyc_taxi.md`](docs/nyc_taxi.md).
- **TPC-H** at scale factor 0.1, generated via DuckDB (`orders`, `lineitem`,
  `customer`). Column descriptions: [`docs/tpch.md`](docs/tpch.md).

"Clean" data is the base after removing rows that already violate obvious
rules; every cleaning step is recorded to `data/clean/*.cleaning.json`.

## Design notes

- Every dataframe carries a stable `_row_id` so injected-error labels and GX
  `unexpected_index_list` output refer to the same identifiers.
- `referential_integrity` errors are represented as membership violations
  against the parent key set (core GX 1.x has no cross-table FK expectation);
  LLM single-table suites are expected to miss these, which is itself an RQ2
  finding.
