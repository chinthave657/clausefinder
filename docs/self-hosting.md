# Self-hosting

ClauseFinder is designed to run entirely on your own machine, with your own
API key, no accounts on our side. This is the CPU profile the hosted demo's
"at capacity → run it yourself" fallback points to, and it's meant to work
on a 16GB laptop.

## What you need

- An API key for **either**:
  - [build.nvidia.com](https://build.nvidia.com) (`NVIDIA_API_KEY`) — free
    dev/eval tier, member rate limits apply; **or**
  - [OpenRouter](https://openrouter.ai) (`OPENROUTER_API_KEY`) — prepaid,
    `:free`-suffixed model variants exist for zero-cost dev loops.
- Docker + Docker Compose, **or** Python 3.12 + [uv](https://docs.astral.sh/uv/)
  if you'd rather run without containers.
- A built LanceDB index under `data/index`. A prebuilt index exists as a gated HF dataset (it feeds the hosted
  demo); access is gated, so building locally from the public GSMA/3GPP
  corpus (see below) is the supported self-host path. This step downloads specs and computes embeddings; expect it to
  take a while on CPU.

Nothing here requires a GPU. The `gpu` profile (below) is optional and adds
NVIDIA NIM containers for local inference — it needs an NGC key and NIM
entitlements are dev-tier only per NVIDIA's terms, so it's not the default
and isn't required to self-host.

## CPU profile (default)

```bash
git clone https://github.com/chinthave657/clausefinder && cd clausefinder
cp .env.example .env
# edit .env: set NVIDIA_API_KEY or OPENROUTER_API_KEY

make ingest       # download GSMA/3GPP markdown, clause-aware chunk it
make index        # embed chunks with OTel-Embedding-568M, build data/index

docker compose up --build
```

The app is served at `http://localhost:7860` (Gradio). LLM calls (answer
synthesis, diff synthesis) go out to build.nvidia.com or OpenRouter with
your key — nothing is cached or proxied through a third party. Only the
568M-parameter embedding model runs locally, on CPU, for query embedding at
request time; reranking is disabled by default on this profile (it adds ~5–30 s per
query on CPU); enable it with `CLAUSEFINDER_RERANK=1`, which runs
OTel-Reranker-0.6B on whatever device is available (cuda/mps/cpu).

`docker-compose.yml` mounts `./data` into the container — the index you
build on the host (or inside the container, same commands) is what the app
serves. Rebuilding the index is a `make ingest index` away; the container
itself has no state.

### Choosing which specs to index

By default `ingest/download_gsma.py` pulls series 23/24/29/33/38 across
Rel-17 and Rel-18 (the v1 scope — see the README FAQ for adding other
series, e.g. LTE/36-series, with one extra flag). To restrict further:

```bash
uv run python ingest/download_gsma.py --series 38 --releases Rel-18
uv run python ingest/chunker.py data/corpus
make index
```

A smaller scope means a faster ingest and a smaller index — useful for
trying ClauseFinder against just the series you care about.

## GPU profile (stub — local NIM inference)

For a fully local setup with no calls leaving your machine (needs an NGC
API key, NIM dev-tier entitlements, and a GPU with ≥48GB VRAM to serve
Nemotron as a NIM microservice):

```bash
docker compose --profile gpu up
```

This profile is defined in `docker-compose.yml` as a commented-out stub
today — uncomment the `nim` service, set `NGC_API_KEY`, and point the app at
it. It also turns the reranker on (OTel-Reranker-0.6B runs comfortably on a
GPU that size). This profile exists to carry the "runs fully on the NVIDIA
stack, including inference" story; it is **not** the default because NIM
entitlements are dev-only and most people self-hosting don't have a spare
48GB GPU sitting around — the CPU profile with a BYO API key is the
supported path for everyone else.

## Local dev without Docker (uv)

```bash
uv sync
cp .env.example .env
make ingest index
uv run clausefinder search "RRC reestablishment"   # retrieval only, no LLM
uv run clausefinder ask "What triggers RRC reestablishment?"
uv run python web/app.py                            # Gradio, same as the container
```

`make lint` / `make test` / `make check` run the same checks CI runs, and
`make test` needs neither an API key nor a built index (see
`CONTRIBUTING.md`).

## Troubleshooting

- **"no supporting clauses found" on everything** — the index is empty or
  the wrong path. Confirm `data/index/chunks.lance` exists and that
  `make ingest index` completed without errors; check `data/parsed/manifest.json`
  for per-spec chunk counts.
- **Slow first query** — the embedding model downloads on first run (cached
  under `HF_HOME`, which the Dockerfile points at `/app/data/hf-cache`
  inside the mounted volume, so it persists across container restarts).
- **`set OPENROUTER_API_KEY or NVIDIA_API_KEY`** — neither env var is set in
  `.env`, or `.env` wasn't copied from `.env.example` in the first place.
- **Rate limits on build.nvidia.com** — the free member tier is capped
  (~40 RPM); switch to an OpenRouter key, or a `:free`-suffixed model
  variant for dev loops, if you're iterating quickly.

## Configuration reference

| Env var | Default | Effect |
|---|---|---|
| `CLAUSEFINDER_RERANK` | off | `1` enables OTel-Reranker-0.6B (better precision; +5–30 s/query on CPU) |
| `CLAUSEFINDER_MODEL` | Nemotron Nano-30B-A3B | Answer/explain model |
| `CLAUSEFINDER_MODEL_CHAIN` | unset | Comma-separated fallback chain, primary first (the hosted demo uses `nemotron-3.5-lightning,nemotron-3-nano-30b-a3b`) |
| `CLAUSEFINDER_DIFF_MODEL` | Nemotron Super-120B | Diff synthesis model |
| `DEMO_DAILY_LIMIT` | 10 | Per-session daily query cap in the web app |

