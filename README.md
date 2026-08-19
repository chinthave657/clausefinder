# ClauseFinder

Clause-cited answers to 3GPP standards questions, plus release-to-release
clause diffs (Rel-17 vs 18 vs 19) — grounded in the official GSMA/3GPP
markdown corpus, retrieved with a hybrid BM25+vector funnel, and validated
against the source text before anything is shown to you. No hosted 3GPP
assistant exists today (prior art is self-host research repos); ClauseFinder
is built end-to-end on the NVIDIA agentic stack — NeMo Agent Toolkit,
NeMo Guardrails, and Nemotron — plus the GSMA Open Telco AI ecosystem's
corpus and domain-tuned OTel models.

**Status:** early build (P0/P1 per the [design doc](https://github.com/venkych/clausefinder)) — dev index covers a subset of TS 38.331; full 5-series corpus and hosted demo are in progress.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ CLIENTS: HF ZeroGPU Space (public demo) · CLI · docker compose│
└──────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│ AGENT SERVICE (NAT workflow)                                    │
│  Guardrails in: topic + prompt-injection (spec text = data)     │
│        │                                                         │
│  ROUTER — UI tab is the mode; regex pre-check for diff intent;  │
│           Nano structured-output fallback only on ambiguity      │
│        │                                                         │
│  RETRIEVAL (deterministic)                                       │
│   1. lexicon append (TR 21.905 + per-spec §3 terms)      [+6pp]  │
│   2. BM25 top-100 ⊕ vector top-100 → weighted RRF                │
│   3. rerank (OTel-Reranker-0.6B; CPU profile: Nano fallback)     │
│   4. parent expansion → full clause unit, deduped                │
│   5. 1-hop xref expansion (≤4 chunks)                            │
│   6. abstain check — below threshold ⇒ "no supporting clause"    │
│        │                                                         │
│  ANSWER AGENT (Nemotron Nano, reasoning off, streamed)           │
│   tagged [Ck] citations + verbatim quote per claim               │
│        │                                                         │
│  DIFF AGENT (Nemotron Super, reasoning on)                       │
│   deterministic clause diff rendered first → LLM synthesis over  │
│        │                                                         │
│  CITATION VALIDATOR (deterministic, always runs)                 │
│   tag ∈ retrieved set · quote fuzzy-matches its chunk (≥0.95) ·  │
│   {spec, clause} ∈ index manifest → repair pass → strip-on-fail  │
│  Guardrails out: citation-required, no-config-execution          │
└──────┬─────────────────────┬──────────────────────┬──────────────┘
       │                     │                      │
┌──────▼────────┐  ┌─────────▼──────────┐  ┌────────▼─────────────┐
│ INDEX          │  │ LLM GATEWAY         │  │ INGEST (offline)      │
│ LanceDB        │  │ build.nvidia.com /  │  │ GSMA/3GPP markdown →  │
│ children 200-  │  │ OpenRouter, BYO key │  │ clause-aware chunk    │
│ 350 tok +      │  │ (PKCE + paste-key)  │  │ (parents ≤1200 tok,   │
│ breadcrumb,    │  │                      │  │ children 200-350) →   │
│ parents, xref  │  │                      │  │ xref edges → OTel-    │
│ edges; tantivy │  │                      │  │ 568M embed → LanceDB  │
│ BM25 alongside │  │                      │  │                       │
└────────────────┘  └──────────────────────┘  └───────────────────────┘
```

Full design rationale (why 250-token chunks, why flat search beats HNSW,
why lexicon-append instead of substitution, etc.) is in
[`docs/architecture.md`](docs/architecture.md).

## Quickstart

### Docker (self-host, CPU, BYO key)

```bash
git clone https://github.com/venkych/clausefinder && cd clausefinder
cp .env.example .env            # add NVIDIA_API_KEY or OPENROUTER_API_KEY
make ingest index               # build a local corpus + LanceDB index (one-time)
docker compose up --build       # Gradio on http://localhost:7860
```

GPU profile (adds NIM containers, reranker on — needs an NGC key + ≥48GB
GPU, dev-tier NIM entitlements only): `docker compose --profile gpu up`.
See [`docs/self-hosting.md`](docs/self-hosting.md) for both profiles, sizing,
and troubleshooting.

### Local dev (uv)

```bash
uv sync
cp .env.example .env
make ingest index               # or point at a pre-built data/index
uv run clausefinder ask "What is the RRC Reestablishment procedure?"
make demo                       # Gradio locally (uv run python web/app.py)
```

## Try it live

`[hosted demo — coming soon]` — HF ZeroGPU Space, launching with the full
5-series corpus (see [build plan](https://github.com/venkych/clausefinder)).
Until then, run it yourself with the Docker quickstart above.

## Benchmarks

Numbers are published only once measured — no placeholders dressed up as
results. Full methodology (golden-set CI gate, TeleQnA subsets, GSMA
satellite suite, contamination caveat for OTel-2.0) lives in
[`docs/architecture.md`](docs/architecture.md#evaluation) and will land in
`docs/benchmarks.md` once P4 runs.

| Benchmark | Config | Metric | Result |
|---|---|---|---|
| Golden set (100 items, stratified) | Nano + ClauseFinder retrieval | pass rate | pending |
| Retrieval | — | recall@5 / recall@20 / MRR@10 | pending |
| TeleQnA Rel-17 (734Q) | bare Nano | accuracy | pending |
| TeleQnA Rel-17 (734Q) | Nano + ClauseFinder | accuracy | pending |
| TeleQnA Rel-18 (780Q) | bare Nano | accuracy | pending |
| TeleQnA Rel-18 (780Q) | Nano + ClauseFinder | accuracy | pending |
| TeleQnA Rel-18 (780Q) | OTel-2.0-31B (closed-book, contaminated*) | accuracy | pending |
| GSMA satellite (7 benchmarks, self-run) | Nano + ClauseFinder | per-suite | pending |

\* OTel-2.0 is trained inside the ecosystem that authors these benchmarks —
reported open-book (ours, retrieval-grounded) vs closed-book (OTel-2.0,
weights-only) for honest comparison, not a leaderboard claim.

## Why not just use a bigger context window / OTel-2.0 directly?

**OTel-2.0** (Gemma trained on 440B telecom tokens, 0.917 TeleQnA / 0.873
3GPP-TSG on the leaderboard) is a strong closed-book baseline — but it was
trained inside the ecosystem that authors those benchmarks, so its numbers
carry a contamination caveat that ours don't. ClauseFinder answers **open-book,
clause-cited, and verifiable**: every claim resolves to a specific TS/TR
clause you can check against the source. Retrieval-augmented + validated
generation, not memorized weights, is how we get grounding you can audit —
that's the OTel-2.0 comparison framed honestly, not a fight we're trying to
dodge.

## Built on the NVIDIA stack

- **NeMo Agent Toolkit (NAT)** — the router/retrieval/answer/diff workflow
  runs as a NAT workflow (Python API in-process; YAML config for the router
  variant kept for eval).
- **Nemotron** — Nano-30B-A3B answers (reasoning off, low latency/cost);
  Super-120B synthesizes release diffs (reasoning on, over a deterministic
  diff rendered first).
- **NeMo Guardrails** — input rails (topic scoping, prompt-injection —
  retrieved spec text is always treated as data, never instructions) and
  output rails (citation-required, no-config-execution).
- **OTel models** — OTel-Embedding-568M (NDCG@10 90.1 vs 57.2 general-purpose)
  and OTel-Reranker-0.6B (MRR@10 0.944), both Apache 2.0, both telecom-domain
  fine-tunes.
- **GSMA Open Telco AI corpus** — the official GSMA/3GPP markdown mirror is
  the only source ingested; see [`docs/legal.md`](docs/legal.md) for
  provenance and redistribution posture.

## FAQ

**What about LTE / the 36-series?**
The ingest pipeline is series-agnostic. Adding LTE (36-series) or any other
series is one config line:

```bash
uv run python ingest/download_gsma.py --series 36 --releases Rel-17
```

then `make index` to embed and add it to your local LanceDB. v1 ships with
series 23/24/29/33/38 (arch, NAS, core signalling, security, NR RAN); LTE
wasn't in scope for launch, not because the pipeline can't handle it.

**Why not fine-tune instead of RAG?**
We benchmark both rather than assume an answer. v1 is retrieval-augmented
generation over the base Nemotron models, with a citation validator that
makes every claim checkable against the source clause — that's the part
fine-tuning alone doesn't give you. A QLoRA fine-tune on telecom data is
planned as **P5** (see the build plan), evaluated against bare Nano,
Nano+RAG, and OTel-2.0 on the same suite so the comparison is apples-to-apples,
not a marketing number.

## Docs

- [`docs/architecture.md`](docs/architecture.md) — the funnel, stage by
  stage, with the "why" behind each constraint.
- [`docs/self-hosting.md`](docs/self-hosting.md) — Docker/uv self-host guide,
  CPU and GPU profiles.
- [`docs/legal.md`](docs/legal.md) — corpus provenance, citation posture,
  takedown contact.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — dev setup, test conventions.
- [`NOTICE`](NOTICE) — third-party model and data attribution.

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Third-party model/corpus attribution
in [`NOTICE`](NOTICE).
