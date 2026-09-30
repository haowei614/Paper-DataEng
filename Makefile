## dqgen experiment pipeline.
##
## Each target runs one stage of the study end to end via the `dqgen` CLI.
## Override the config with `make CONFIG=path/to.yaml <target>`.

CONFIG ?= configs/experiment.yaml
RUN    ?= uv run dqgen

.PHONY: help data inject generate validate report all test clean-results

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

data:  ## Download/generate, clean, and sample datasets.
	$(RUN) data --config $(CONFIG)

inject:  ## Create corrupted datasets and labels (+ manifest).
	$(RUN) inject --config $(CONFIG)

generate:  ## Generate LLM constraint suites (cached).
	$(RUN) generate --config $(CONFIG)

validate:  ## Run all suites and write the results table.
	$(RUN) validate --config $(CONFIG)

report:  ## Produce summary tables (CSV + LaTeX) and figures.
	$(RUN) report --config $(CONFIG)

all: data inject generate validate report  ## Run the full pipeline.

test:  ## Run the unit tests.
	uv run pytest -q

clean-results:  ## Remove generated results (keeps data and cache).
	rm -rf results/*
