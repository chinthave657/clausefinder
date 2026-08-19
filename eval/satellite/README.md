# GSMA `satellite` — self-run protocol

Design §5.5 / §5 point 5: the official [GSMA leaderboard](https://github.com/gsma-labs/satellite)
accepts **models only, all-7-benchmarks-or-nothing** (TeleQnA 1000 +
TeleLogs/TeleMath/TeleTables/3GPP-TSG 100 each + ORANBench/srsRANBench 150;
every one of the 89 current entries is a bare model). ClauseFinder is a RAG
*system*, not a model — it cannot be an official leaderboard entry. So we run
`satellite` ourselves, locally, against our own configs, and publish the
numbers in `docs/benchmarks.md` with the suite version and dataset revision
hashes pinned (frozen inputs — a moving benchmark makes trend comparisons
meaningless).

**Official leaderboard PR stays reserved for models we actually control**:
bare Nano first (check whether it's already on the board), P5 QLoRA is the
flagship entry once it exists (design §6, P5). Neither of those needs this
directory — that PR is a model submission through satellite's normal
contribution flow, not a local run.

## What "self-run" means here

`satellite` (per its own docs) wraps each benchmark as an
[Inspect AI](https://inspect.ai-safety-institute.org.uk/) task and evaluates
a `ModelAPI` against it. Inspect's `openai/*` provider family works against
**any OpenAI-compatible chat-completions endpoint** via `base_url` — that's
the integration point for both of our configs below.

## The 3 configs (see `run_config.md` for the exact comparison protocol)

### 1. Bare Nano (no RAG)
`nvidia/nemotron-3-nano-30b-a3b` called directly, closed-book — no corpus,
no retrieval, no citations. This is the "what does the base model already
know" control.

```
inspect eval <benchmark_task> \
  --model openai/nvidia/nemotron-3-nano-30b-a3b \
  --model-base-url https://integrate.api.nvidia.com/v1 \
  --model-api-key $NVIDIA_API_KEY
```

### 2. Nano + ClauseFinder (our RAG config) — via the local shim
Satellite/Inspect only knows how to talk to a model endpoint; ClauseFinder is
a pipeline (retrieve → rerank → answer → validate). `eval/satellite/shim.py`
is a small FastAPI app that exposes `POST /v1/chat/completions` in the exact
OpenAI wire shape, but internally runs `agent.answer.ask()` against our
LanceDB index instead of calling an LLM directly. From Inspect's point of
view it's just another OpenAI-compatible model — it never needs to know
retrieval happened underneath.

**Shim design, one request = one question:**
- take the last `role: user` message as the question (satellite's tasks are
  single-turn Q&A, so there's no multi-turn state to carry)
- run it through `Retriever` (BM25+vector RRF, rerank, parent+xref expansion)
  and `agent.answer.ask()` (Nano-30B, reasoning off, citation-validated)
  against `data/index` — same code path the CLI and demo use, not a
  reimplementation
- return the answer text as `choices[0].message.content`, non-streamed
- `temperature`/`max_tokens` in the request are accepted for wire
  compatibility and ignored — `ask()` already runs its own fixed decoding
  params (design §3)
- an extra `clausefinder_validator` field carries the citation-validator
  result for local debugging; OpenAI-strict clients ignore unknown
  top-level keys, so this doesn't break Inspect's parsing

```
# terminal 1 — run the shim against the local dev/full index
uv run uvicorn eval.satellite.shim:app --port 8091

# terminal 2 — point satellite at it
inspect eval <benchmark_task> \
  --model openai/clausefinder-local \
  --model-base-url http://127.0.0.1:8091/v1 \
  --model-api-key unused
```

### 3. OTel-2.0-31B (benchmark rival, closed-book)
Called directly the same way as config 1 but against OTel-2.0-31B. See the
**contamination caveat** below — this is not an apples-to-apples baseline.

## Contamination caveat (state this verbatim wherever config-3 numbers appear)

> OTel-2.0-31B was trained inside the ecosystem that authors these
> benchmarks (Gemma-4 + 440B telecom tokens, published leaderboard numbers of
> 0.873 on 3GPP-TSG and 0.917 on TeleQnA). Our comparison is **open-book
> (Nano+ClauseFinder, retrieval over the frozen GSMA/3GPP corpus) vs.
> closed-book (bare Nano, OTel-2.0)** — not same-conditions. Differences
> **under ~6pp on 3GPP-TSG (n=100, SE≈3pp)** between configs are not
> statistically distinguishable and should not be read as a ranking.

See `run_config.md` for the full protocol this caveat is attached to.

## Pinning

Every self-run publishes, alongside the numbers:
- `satellite` git commit / release tag used
- Inspect AI version (`uv run pip show inspect-ai` or equivalent)
- dataset revision hash for each of the 7 benchmarks (satellite pins these;
  copy them, don't re-resolve "latest")
- ClauseFinder index build id (which corpus/chunking run `data/index` came from)
- `JUDGE_MODEL` / answer model revision (`agent.answer.py`'s pinned model id)

Rationale: design §8 risk table — "NAT API churn / OTel weight drift" is
mitigated by pinning everything; an unpinned satellite run is not a
reproducible number, it's a snapshot nobody can audit later.

## What this directory does NOT do

- **No bulk runs from this repo's CI or agents.** The 7-benchmark suite is
  large (TeleQnA alone is 1000 items); this is a manual, deliberate,
  cost-aware local run, not something CI triggers.
- **No official leaderboard submission logic.** That's a separate,
  human-driven PR against `gsma-labs/satellite` for models only (see above).
- **No corpus/index building.** The shim reads whatever `data/index` already
  is; it does not run ingest.
