# ClauseFinder — CPU app image (BYO API keys; inference runs on build.nvidia.com).
# The embedding model (OTel-Embedding-568M) runs locally on CPU for queries.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/data/hf-cache \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

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

# data/ (LanceDB index + parsed corpus) is mounted as a volume — see docker-compose.yml
EXPOSE 7860
CMD ["uv", "run", "--no-dev", "python", "web/app.py"]
