# Architecture

This is a prose walkthrough of the ClauseFinder pipeline, stage by stage,
with the reasoning behind each design choice — not just what the code does,
but why it does it that way. The numbered constraints below are pinned in
the [design document](https://github.com/venkych/clausefinder) §2; this doc
explains how they show up in the funnel. Code references point at the
current implementation in `ingest/`, `agent/tools/retrieval.py`,
`agent/answer.py`, `agent/diff.py`, and `agent/validators/citations.py`.

## Guiding principle: validator-first

Every layer that *can* be deterministic, is. The LLM only ever runs on the
residual — synthesis over already-retrieved, already-structured text — and
its output is checked by a deterministic validator before it reaches you.
This isn't a style preference: constrained decoding that guarantees valid
citations doesn't exist cross-provider (OpenRouter's `json_schema` support
is heterogeneous — see §2), so post-generation validation is the only
approach that works regardless of which backend serves the model. The
citation validator (`agent/validators/citations.py`) runs on every answer,
every time, with no bypass path.

## Ingest: clause-aware chunking, not fixed-window

The GSMA/3GPP markdown corpus (`ingest/download_gsma.py`) is converted into
retrieval units by `ingest/chunker.py`. Two design choices come straight
from published measurements on 3GPP retrieval specifically, not general RAG
folklore:

- **125–250-token child chunks, not 500.** Telco-oRAG measured 125–250-token
  chunks beating 500-token chunks on 3GPP text (79.6 vs 78.8). Spec clauses
  are dense and procedural — a 500-token window usually straddles two
  distinct requirements, diluting the embedding for both. ClauseFinder
  targets 300 tokens per child (`CHILD_TARGET`, capped at 350) as the
  practical midpoint of that measured range, splitting on paragraph
  boundaries within a clause rather than blindly on token count.
- **Clause numbers drive structure, not markdown heading depth.** The GSMA
  markdown conversion is not internally consistent about heading levels
  (`## 5.2.2` sits next to `### 4.2.1` in the same document), so the parser
  derives the clause hierarchy from the clause *number* itself
  (`_clause_rank` / the breadcrumb stack in `chunker.py`), never from `#`
  nesting. This is what makes breadcrumbs like `TS 38.331 Rel-18 — 5.3 RRC
  procedures > 5.3.5.3 Reception…` reliable.
- **Tables stay atomic.** TSpec-LLM measured preserving tables intact rather
  than splitting them lifting retrieval quality (60→75%). `_split_children`
  never breaks a table row across chunks, and tables that would otherwise
  overflow the target size are still kept whole.
- **The 1400-token embed cap is a fine-tune constraint, not a preference.**
  OTel-Embedding-568M was fine-tuned at a max sequence length of 1500
  tokens. Anything embedded longer than that is truncated by the tokenizer,
  silently dropping content from the vector. Chunking enforces a 1400-token
  ceiling (`EMBED_MAX`) on breadcrumb+text with margin, and any chunk that
  would exceed it is flagged (`over_embed_limit`) and logged at ingest time
  rather than silently truncated — see `ingest/chunker.py`'s per-spec stats.
- **Parent units (≤1200 tokens) sit above child chunks.** This is the
  small-to-big pattern: children are the retrieval unit (small, precise
  embeddings), but generation reads the full parent clause so the model
  isn't reasoning over a truncated fragment of a requirement.

## Query enhancement: append, never substitute

Before retrieval, `enhance_query` (`agent/tools/retrieval.py`) appends
acronym expansions drawn from TR 21.905 vocabulary and per-spec §3
definitions to the query — it never rewrites or replaces the original text.
Telco-RAG measured this lexicon step alone worth +6pp (84.8→90.8) on 3GPP
QA. Appending rather than substituting matters: the original wording still
drives the BM25 leg (exact spec terminology often *is* the query), while the
expansion gives the vector leg extra surface area for questions phrased in
plain English instead of 3GPP jargon.

## Hybrid retrieval: BM25 ⊕ vector, weighted RRF, no HNSW

Two independent searches run over the same LanceDB table: tantivy BM25 (FTS)
and OTel-Embedding-568M vector search, each to depth 100 (`CAND_DEPTH`). They
fuse via **weighted** reciprocal-rank fusion — vector at 0.7, BM25 at 0.3
(`VEC_W` / `FTS_W` in `retrieval.py`) — rather than the naive unweighted RRF
most hybrid-search writeups use.

The weighting exists because unweighted RRF systematically favors *dual
presence* over *single-leg excellence*: a chunk ranked mediocre-mediocre on
both legs can outscore a chunk ranked #1 on vector search alone. OTel-568M's
measured retrieval quality (NDCG@10 90.1 vs 57.2 for a general-purpose
embedder) is strong enough that its top hits shouldn't be diluted by that
effect, so vector search gets the larger weight. Fusion is capped at rank 50
(`FUSE_DEPTH`) per leg — Telco-RAG-adjacent literature found ranks beyond
that add fusion noise rather than signal.

**Flat (brute-force) search, not HNSW, and no post-filter.** Two decisions
here, both evidence-driven rather than default-library behavior:

- Telco-oRAG measured flat search beating HNSW on 3GPP-scale corpora — an
  approximate index buys speed you don't need yet at this corpus size, at an
  accuracy cost you do pay. LanceDB's exact search is used deliberately; a
  flat-vs-IVF recall check is planned for CI once the full corpus is built,
  to catch the point where that tradeoff should flip.
- Release/series filtering is a LanceDB **prefilter**, not a post-filter
  (`vq.where(where, prefilter=True)` in `Retriever.search`) — the constraint
  narrows the candidate set *before* the ANN search runs, so a release
  filter can't silently starve recall by cutting into an already-ranked
  result list. And it's a *soft* series boost when a TS number appears in
  the query, never a hard filter — trained-router approaches (Telco-oRAG's
  own router) measurably confused series 24/38 and 23/29, which are exactly
  the series ClauseFinder ships. A hard filter would inherit that failure
  mode; a boost degrades gracefully instead.

## Rerank, parent expansion, xref expansion

- **Rerank** (OTel-Reranker-0.6B, MRR@10 0.944) runs on the fused top-50,
  narrowing to the top 5–8 chunks that actually reach generation. On the
  self-host CPU profile without a GPU, this step falls back to a Nano
  listwise rerank instead of running the 0.6B model on CPU.
- **Parent expansion** deduplicates by `parent_id` and returns one hit per
  clause unit — the full parent text (≤1200 tokens), not just the winning
  child fragment. This is the small-to-big pattern completing: retrieval
  precision comes from small children, but the model synthesizes over
  complete clauses.
- **1-hop cross-reference expansion** adds up to 4 more chunks
  (`XREF_EXTRA`) by following "see clause X" / "see TS nn.nnn" edges
  extracted at ingest time (`XrefEdge`, built via regex over each clause's
  body) from the top-3 results. This captures most of the multi-hop value a
  graph-RAG approach would provide, at a fraction of the engineering cost —
  see "explicitly rejected" below.
- **Abstain check:** if the top fused score falls below a calibrated
  threshold, the answer agent is told to report "no supporting clause found"
  rather than search-and-hope. A wrong answer with confident-looking
  citations is worse than an honest abstain.

## Answer generation: prompt structure, not just prompt content

`agent/answer.py` assembles context in a specific order — **query → clause
excerpts → the query again, repeated → instruction** — rather than the more
common instruction-first or single-query layout. Telco-RAG measured this
structured ordering worth +4.6pp over an unstructured prompt on the same
task. Reasoning is explicitly requested off for the answer call
(`NO_THINK`), since Nemotron-3 models are reasoning-default and this path
needs low latency, not deliberation — the retrieval funnel already did the
hard reasoning work by narrowing to the right clauses.

## Diff mode: deterministic diff first, LLM synthesizes over it

`agent/diff.py` resolves clause sets on both release sides independently
(release-filtered retrieval, ≤15 chunks/side), aligns clauses primarily by
clause number with a title-similarity fallback for renumbered clauses across
releases, and computes a **deterministic** unified diff (`difflib` over
normalized lines) before any LLM call. That rendered diff is shown to the
user immediately — synthesis (Super-120B, reasoning **on**, the one
reasoning-on call in the system) runs over the diff and the source excerpts
to produce prose, but the ground truth was already computed and displayed
without waiting on the model. An empty diff short-circuits entirely: "no
changes" is the correct answer and no synthesis can improve on it.

## Citation validation: three checks, in order of strength

`agent/validators/citations.py` runs after every generation call, ask or
diff:

1. every cited `[Ck]` tag must exist in the retrieved set (exact match);
2. every quoted span must fuzzy-match (≥0.95, `rapidfuzz`) *within that
   specific chunk* — not "somewhere in the retrieved context," which would
   let a model attribute a real quote to the wrong clause;
3. every `{spec, clause}` pair the answer names must exist in the index
   manifest.

On failure, one repair pass asks the model to fix only the failed citations
(`agent/answer.py`'s second `_chat` call); if that still fails, quotes that
can't be verified anywhere in the retrieved set are stripped
(`strip_failed_quotes`) rather than shown — the claim keeps its clause
reference but loses the fabricated-looking quote. Validation happens before
any output rail and is never skippable by prompt or config.

## Explicitly rejected (recorded to pre-empt re-litigating them)

- **Graph-RAG on GSMA knowledge graphs** — the available `telecom-kg-rel19`
  is Rel-19-only, which can't serve a Rel-17-vs-18 diff. The 1-hop xref
  table captures most of the multi-hop retrieval value at a fraction of the
  engineering cost.
- **Late chunking** — incompatible with OTel-568M's pooled-contrastive
  fine-tune at a fixed 1500-token window.
- **A trained neural spec router** — Chat3GPP's hybrid+RRF approach beat
  Telco-RAG's trained router without any training at all, and Telco-oRAG's
  own router measurably confused exactly the series ClauseFinder ships
  (24/38, 23/29). A soft boost captures the useful signal without the
  failure mode.
- **HyDE-style dual-round candidate-answer retrieval** — Telco-oRAG measured
  a +1.2–3.4pp gain for 75–80% more retrieval latency. Not worth it at
  current recall; revisit only if TeleQnA accuracy drops below ~75%.
- **Trusting `json_schema` mode across OpenRouter providers** — support is
  heterogeneous across the providers OpenRouter routes to. `require_parameters:
  true` plus enum-tolerant parsing plus a Pydantic re-ask on failure is used
  instead of relying on constrained decoding working everywhere.

## Evaluation

Three independent measurement layers, each gating a different thing:

1. **Golden set** (100 stratified Q&A items, ~20/series, ≥20/mode, ≥10
   adversarial including 5–10 out-of-corpus abstain cases) — CI gate is
   **paired per-item regression** (any previously-passing item flipping to
   failing blocks the change), not an aggregate score threshold. At n=100
   the aggregate Wilson 95% CI is ±4–5pp, wide enough that an aggregate gate
   can't reliably see a 5pp regression; per-item pairing can.
2. **Retrieval-stage metrics** — recall@5, recall@20, MRR@10 at clause
   granularity, with a pinned parent-match rule (a retrieved child under
   gold clause 5.2.1 counts as matching gold 5.2.1 even if it's specifically
   5.2.1.3). Chunk-size sweep across {150, 250, 400} tokens recorded once;
   flat-vs-IVF recall checked in CI.
3. **Citation faithfulness** — the deterministic validator runs in CI on
   every change; an offline ALCE-style entailment check runs non-gating on
   paraphrase-path citations specifically, since that's the fallback path
   where an unfaithful citation could accumulate undetected.
4. **Public comparability** — TeleQnA Rel-17 (734Q) and Rel-18 (780Q) 3GPP
   subsets, the split every prior 3GPP-RAG system reports results on, run
   for three configs: bare Nano, Nano+ClauseFinder, and OTel-2.0-31B
   (pinned, with the contamination caveat stated verbatim wherever the
   number appears — see the README).
5. **GSMA satellite suite** — all 7 official benchmarks run locally (not
   submitted to the leaderboard, which accepts models only, not RAG
   configs) via an Inspect ModelAPI wrapper; results published with suite
   version and dataset revision hashes pinned so they're reproducible.

None of these have run against the full corpus yet — see the benchmark
table in the README for what's pending and why no numbers are fabricated in
the meantime.
