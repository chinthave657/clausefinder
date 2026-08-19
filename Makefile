# ClauseFinder developer targets. Everything runs through uv.

.PHONY: install lint test check ingest index eval demo run docker-build docker-up

install:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest -q

check: lint test

# Corpus download + clause-aware chunking (data/corpus -> data/parsed)
ingest:
	uv run python ingest/download_gsma.py
	uv run python ingest/chunker.py data/corpus

# Embed + build LanceDB index (data/parsed -> data/index). Needs the chunks.
index:
	uv run python ingest/embed_index.py

eval:
	uv run python eval/retrieval_eval.py

# Gradio demo (same entrypoint the container runs) — needs a built index.
demo:
	uv run python web/app.py

run: demo

docker-build:
	docker compose build

docker-up:
	docker compose up
