.PHONY: api ui ingest eval eval-judge

api:
	uv run python -m tsl_rag.api.main

ui:
	uv run streamlit run src/tsl_rag/ui/app.py

ingest:
	uv run python -m tsl_rag.ingestion.cli ingest-all data/raw/

eval:
	uv run python -m evals.run_evals --output evals/results/run_latest.json

eval-judge:
	uv run python -m evals.run_evals --use-judge --output evals/results/run_latest_judge.json
