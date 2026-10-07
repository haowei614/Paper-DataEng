# Paper

Draft of **"LLM-Assisted Generation of Data Quality Constraints from Schema and Documentation."**

## Build

```bash
cd paper
tectonic main.tex      # or: latexmk -pdf main.tex
```

Produces `main.pdf`.

## Contents

- `main.tex` — the paper (self-contained; tables are embedded with the exact
  numbers from `results/tables/*.csv` of the medium-scope run).
- `references.bib` — bibliography.
- `figures/*.png` — figures copied from `results/figures/` (committed so the
  paper builds without re-running the pipeline).

## Reproducing the numbers

All results come from the medium-scope run (`configs/experiment.yaml`:
`repetitions=1`, `injection.seeds=[1,2]`, `rates=[0.01,0.05]`,
`validation_row_cap=100000`). To regenerate:

```bash
make data && make inject && make generate && make validate && make report
```

The paper reports 6,704 metric rows. Tables/figures are written to
`results/tables/` and `results/figures/`.
