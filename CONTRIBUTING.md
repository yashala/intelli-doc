# Contributing

## Setup
```bash
make setup        # uv sync for api/, npm ci for web/
make db           # pgvector container on localhost:5433 plus test and eval databases
```
Tesseract must be on `PATH` to run the API or parser tests outside Docker
(`brew install tesseract` or `apt-get install tesseract-ocr`).

## Before opening a pull request
```bash
make format lint typecheck
make test         # unit + integration (needs `make db`)
make web-test
make eval-offline # then: uv run --project api python evals/gate.py artifacts/runs/<run_id>
```
CI runs the same checks plus the offline benchmark gate and dependency audits. No API keys
are needed for any of them.

## Changing behaviour that the benchmark measures
- If a change intentionally moves offline metrics, regenerate the baseline with
  `evals/gate.py <run> --update` in the same pull request and explain the change.
- Tune thresholds on the `dev` split only. State in the pull request which split you looked at.
- Do not edit numbers in `docs/results/` or the README results block; run `make report`.

## Adding evaluation questions
Add to `evals/questions_real.jsonl` (hand-written) or to the generator for synthetic documents.
Each answerable question needs `expected_doc`, `expected_pages` and short exact `evidence`
strings taken from the page image, not from parser output.
