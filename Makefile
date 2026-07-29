# Cele pomocnicze. Na Windows `make` zwykle nie jest dostępny —
# wtedy odpalaj prawą stronę reguły wprost w PowerShellu.

.PHONY: help db-up db-down api ui ingest eval eval-judge lint format typecheck test check

help:
	@echo "db-up      — Postgres + pgvector w Dockerze (host: 5433)"
	@echo "db-down    — zatrzymanie bazy (wolumen zostaje)"
	@echo "ingest     — PDF -> chunki -> embeddingi -> pgvector (14 plików)"
	@echo "api        — FastAPI na :8000"
	@echo "ui         — Streamlit (wymaga działającego api)"
	@echo "eval       — golden dataset, ocena keyword-match"
	@echo "eval-judge — golden dataset, ocena LLM-as-a-judge (Gemini)"
	@echo "check      — lint + typecheck + testy (to samo co CI)"

db-up:
	docker compose up -d

db-down:
	docker compose down

api:
	uv run python -m tsl_rag.api.main

# UI leży w ui.py w katalogu głównym. Poprzednio ten cel wskazywał na
# src/tsl_rag/ui/app.py — plik, którego w repo nigdy nie było.
ui:
	uv run streamlit run ui.py

ingest:
	uv run python -m tsl_rag.ingestion.cli ingest-all data/raw/

eval:
	uv run python -m evals.run_evals --output evals/results/run_latest.json

eval-judge:
	uv run python -m evals.run_evals --use-judge --output evals/results/run_latest_judge.json

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src/tsl_rag

test:
	uv run pytest -m unit

check: lint typecheck test
