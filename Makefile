# Requires: uv, node 22+, docker, tesseract (for running the API outside Docker).
API := uv run --project api
EVAL_DB ?= postgresql://intellidoc:intellidoc@localhost:5433/intellidoc_eval
TEST_DB ?= postgresql://intellidoc:intellidoc@localhost:5433/intellidoc_test

.PHONY: setup up down db test test-unit lint typecheck format web-test corpus eval-offline eval-online report audit

setup:
	cd api && uv sync
	cd web && npm ci

up:
	docker compose up --build -d
	@echo "UI: http://localhost:8080  API: http://localhost:8000/health"

down:
	docker compose down

db:
	docker compose up -d --wait db
	docker compose exec -T db psql -U intellidoc -tc "SELECT 1 FROM pg_database WHERE datname='intellidoc_test'" | grep -q 1 || docker compose exec -T db createdb -U intellidoc intellidoc_test
	docker compose exec -T db psql -U intellidoc -tc "SELECT 1 FROM pg_database WHERE datname='intellidoc_eval'" | grep -q 1 || docker compose exec -T db createdb -U intellidoc intellidoc_eval

format:
	$(API) ruff format api evals

lint:
	$(API) ruff check api evals
	$(API) ruff format --check api evals

typecheck:
	$(API) mypy --config-file api/pyproject.toml api/src api/tests evals/run.py evals/report.py
	cd web && npm run typecheck

test-unit:
	cd api && uv run pytest -m "not integration"

test: db
	cd api && TEST_DATABASE_URL=$(TEST_DB) uv run pytest

web-test:
	cd web && npm test

corpus:
	$(API) python evals/corpus/generate.py

eval-offline: db
	EVAL_DATABASE_URL=$(EVAL_DB) $(API) python evals/run.py --llm offline

eval-online: db
	EVAL_DATABASE_URL=$(EVAL_DB) $(API) python evals/run.py --llm openai --judge --max-cost-usd 5

report:
	$(API) python evals/report.py

audit:
	cd api && uv export --frozen --no-hashes --no-dev > /tmp/intellidoc-req.txt && uv run pip-audit -r /tmp/intellidoc-req.txt
	cd web && npm audit --omit=dev
