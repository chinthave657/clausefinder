# ClauseFinder — CPU app image (BYO API keys; inference runs on build.nvidia.com).
# The embedding model (OTel-Embedding-568M) runs locally on CPU for queries.
#
# Multi-stage: uv + build toolchain (needed to resolve/compile deps like
# lancedb/sentence-transformers) stay in the builder; the final image ships
# only the resulting venv and the app code on a plain slim Python base.

# --- builder ---------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Dependency layer (cached unless the lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Project layer
COPY ingest/ ingest/
COPY agent/ agent/
COPY cli/ cli/
COPY eval/ eval/
COPY web/ web/
RUN uv sync --frozen --no-dev

# --- final -------------------------------------------------------------
FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/data/hf-cache \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/ingest /app/ingest
COPY --from=builder /app/agent /app/agent
COPY --from=builder /app/cli /app/cli
COPY --from=builder /app/eval /app/eval
COPY --from=builder /app/web /app/web
COPY pyproject.toml ./

# data/ (LanceDB index + parsed corpus) is mounted as a volume — see docker-compose.yml
EXPOSE 7860
# non-root: contains blast radius of any dependency RCE (CSO audit finding #1)
RUN useradd -m app && chown -R app /app
USER app

CMD ["python", "web/app.py"]
