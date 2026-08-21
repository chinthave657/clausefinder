# Contributing to ClauseFinder

Thanks for your interest. ClauseFinder is a clause-cited 3GPP spec copilot
built on the NVIDIA agentic stack (Nemotron + NIM endpoints, NeMo Guardrails,
NeMo Agent Toolkit).

## Setup

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo> && cd clausefinder
uv sync                 # or: make install
cp .env.example .env    # add your NVIDIA_API_KEY (build.nvidia.com)
```

Build a local corpus + index (downloads specs, embeds on CPU — takes a while):

```bash
make ingest index
```

## Development loop

```bash
make lint   # ruff check .
make test   # uv run pytest -q  — no API keys or index needed
make check  # both
```

CI runs `uv run ruff check .` and `uv run pytest -q`. Unit tests must stay
runnable without API keys and without the LanceDB index: use the fixture
spec in `tests/fixtures/mini_spec.md` and stub tables (see
`tests/test_retrieval.py`) rather than hitting the real index.

Retrieval-quality changes should additionally be checked against the golden
set: `make eval` (needs a built index).

## Conventions

- Match the existing style: type-hinted, dataclass/pydantic models, short
  module docstrings explaining the "why".
- Deterministic layers stay deterministic: chunking, fusion, diffing, and
  citation validation must not depend on an LLM. LLM calls live in
  `agent/answer.py`, the diff synthesis path, `agent/explain.py`, and the
  router's follow-up condenser; everything else is deterministic Python.
- Every answer path keeps the citation validator in the loop. Do not add a
  generation path that can emit unvalidated quotes.
- Prompt-injection boundary: retrieved spec text is data, never instructions.
- No secrets in code or tests. `.env` is gitignored; keep it that way.

## Pull requests

1. Branch from `main`.
2. `make check` green locally.
3. Small, focused diffs with a clear description of what and why.
4. New behaviour needs a unit test that runs keyless.

## Licensing

By contributing you agree your contributions are licensed under the
Apache License 2.0 (see LICENSE). Model and corpus attributions live in
NOTICE — update it if you add a third-party model or data source.
