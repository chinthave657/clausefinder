# ClauseFinder

Clause-cited answers to 3GPP standards questions, plus release-to-release
clause diffs (Rel-17 vs 18 vs 19) — grounded in the official GSMA/3GPP
markdown corpus, retrieved with a hybrid BM25+vector funnel, and validated
against the source text before anything is shown to you. No hosted 3GPP
assistant exists today (prior art is self-host research repos); ClauseFinder
is built end-to-end on the NVIDIA agentic stack — NeMo Agent Toolkit,
NeMo Guardrails, and Nemotron — plus the GSMA Open Telco AI ecosystem's
corpus and domain-tuned OTel models.

**Status:** core build complete ([architecture](docs/architecture.md)) — full Rel-17+18 corpus indexed (391k chunks, 5 series); hosted demo in progress.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ CLIENTS: HF ZeroGPU Space (public demo) · CLI · docker compose│
└──────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│ AGENT SERVICE (NAT workflow)                                    │
│  Guardrails (NAT workflow path): topic + prompt-injection (spec text = data)     │
│        │                                                         │
│  ROUTER — UI tab is the mode; regex pre-check for diff intent;  │
│           Nano structured-output fallback only on ambiguity      │
│        │                                                         │
│  RETRIEVAL (deterministic)                                       │
│   1. lexicon append (TR 21.905 + per-spec §3 terms)      [+6pp]  │
│   2. BM25 top-100 ⊕ vector top-100 → weighted RRF                │
│   3. rerank (OTel-Reranker-0.6B; off by default on CPU — enable with CLAUSEFINDER_RERANK=1)     │
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
│ 350 tok +      │  │ (paste-key)  │  │ (parents ≤1200 tok,   │
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

> **Expected footprint:** the default ingest (5 series × Rel-17/18) downloads ~0.6 GB of markdown and embeds ~390k chunks (hours on CPU, ~1.5 h on a T4 — see `colab/embed_corpus.ipynb`). Fast dev path: `--spec 38331 --releases Rel-18` (~3k chunks, minutes).

### Docker (self-host, CPU, BYO key)

```bash
git clone https://github.com/chinthave657/clausefinder && cd clausefinder
cp .env.example .env            # add NVIDIA_API_KEY or OPENROUTER_API_KEY
make ingest index               # build a local corpus + LanceDB index (one-time)
docker compose up --build       # Gradio on http://localhost:7860
```

GPU profile (stub — uncomment the `nim` service in docker-compose.yml; needs an NGC key and a ≥48 GB GPU; see docs/self-hosting.md) | Benchmark | Config | Metric | Result |
|---|---|---|---|
| Golden set (starter 9, ask mode, 322k-chunk index at time of run (corpus since grown to 391k; full-corpus judged rerun pending)) | Nano + ClauseFinder (rerank on) | validator-pass + gold-cited | 8/9 |
| Retrieval — full corpus (391,487 chunks, 776 specs, Rel-17+18; 107-question golden set (112 rows incl. 5 abstain probes)) | rerank@50 + identifier leg | recall@5 / recall@20 / MRR@10 | **0.87 / 0.91 / 0.78** |
| Retrieval — depth-100 ablation (rejected: top-5 regression at 2× latency) | rerank@100 | recall@5 / recall@20 / MRR@10 | 0.85 / 0.91 / 0.78 |
| Abstain rail (5 out-of-corpus adversarial questions) | calibrated two-floor gate | refused-or-caveated | **5/5** (was 0/5 pre-rail) |
| TeleQnA (ot-lite, 1000Q mixed-source) | Nemotron Nano closed-book | accuracy | 72.2% (3GPP subset n=191: 60.2%) |
| TeleQnA (ot-lite, 1000Q mixed-source) | Nano + naive RAG (always-on) | accuracy | 68.8% (3GPP: **64.9%**, non-3GPP: 70.4% — context distraction) |
| TeleQnA (ot-lite, 1000Q mixed-source) | Nano + **selective RAG** (τ=0.15, fires 16%) | accuracy | **73.8%** (3GPP: **68.6%**, non-3GPP: 75.7%) |
| TeleQnA (leaderboard, reference) | OTel-2.0-31B closed-book (trained in-ecosystem*) | accuracy | 91.7% |

The selective-RAG threshold is the abstain rail's soft floor, **calibrated a
priori on the golden set** — the TeleQnA sweep independently confirms it sits on
the optimum plateau (0.15–0.20). Naive always-on RAG is reported because the
field usually hides it: retrieval helps only where the corpus covers the
question (+8.4pp on 3GPP with the gate) and hurts elsewhere without one
(−5.3pp). Full sweep + caveats: `eval/reports/teleqna_summary.json`.
GSMA satellite suite (7 benchmarks, self-run): pending.

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
- **Nemotron** — model fallback chain via `CLAUSEFINDER_MODEL_CHAIN` (the
  hosted demo leads with Lightning-3.5; Nano-30B-A3B is the guaranteed
  fallback and in-repo default). Nano answers (reasoning off, low latency/cost);
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
planned as a follow-up, evaluated against bare Nano,
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
