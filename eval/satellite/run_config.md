# Satellite self-run: 3-config comparison protocol

Companion to `README.md` (which covers *how* each config is wired). This
file pins *what* gets compared and *how the numbers get reported* so a
future run (or a reviewer) can reproduce the shape of the comparison exactly.

## The 3 configs

| # | Config | Book | Endpoint | Purpose |
|---|--------|------|----------|---------|
| 1 | Bare Nano | closed | `nvidia/nemotron-3-nano-30b-a3b` direct | base-model control: what Nano knows with zero retrieval |
| 2 | Nano + ClauseFinder | open | `eval/satellite/shim.py` → `agent.answer.ask()` over `data/index` | our actual system under test |
| 3 | OTel-2.0-31B | closed | OTel-2.0-31B direct | benchmark-native rival, contamination caveat applies |

"Open/closed book" is the load-bearing distinction, not "good/bad model" —
config 2 gets to read the corpus at answer time, configs 1 and 3 don't.

## Benchmarks run (all 7, satellite's own suite)

TeleQnA (1000) · TeleLogs (100) · TeleMath (100) · TeleTables (100) ·
3GPP-TSG (100) · ORANBench (150) · srsRANBench (150)

Run all 7 per config for a complete satellite pass. If time/cost forces a
subset for an interim check, **3GPP-TSG and TeleQnA are the two to keep** —
they're the ones design §5.4 already benchmarks against published prior art
(Telco-oRAG/Chat3GPP/Telco-RAG), so a partial run still lands on a number
that means something.

## Fixed protocol per run

1. Pin inputs first (README "Pinning" section) — satellite commit, Inspect
   version, per-benchmark dataset hashes, ClauseFinder index build id, model
   revisions. Record these in the results file before running anything.
2. Run configs 1 and 3 directly against their provider endpoints (no shim).
3. Start the shim (`uv run uvicorn eval.satellite.shim:app --port 8091`)
   against the **same `data/index`** that's pinned in step 1, then run
   config 2 through it.
4. Same benchmark set, same Inspect settings (temperature/max_tokens are
   accepted by the shim but not meaningful for config 2 — `ask()` fixes its
   own decoding params; set them consistently for configs 1/3 anyway so the
   report can state one number per axis).
5. Report per-benchmark accuracy for all 3 configs side by side in
   `docs/benchmarks.md`, plus the pinned inputs from step 1.

## Reading the results — required framing

- **Never present config 2 vs config 3 as a plain ranking.** Attach the
  contamination caveat (verbatim block in README.md) every time OTel-2.0
  numbers appear next to ours.
- **State the significance floor.** 3GPP-TSG is n=100 per design §5.5,
  SE≈3pp — differences under ~6pp between any two configs on that benchmark
  are noise, not signal. Say so in the write-up rather than letting the
  reader infer a ranking from a small gap.
- **Config 1 vs config 2 is the number that matters for the product story**:
  it isolates what retrieval + citation-validation buys over the same base
  model with nothing else changed. That comparison has no contamination
  caveat — both configs start from the same Nano weights.
- **Public comparability numbers (TeleQnA Rel-17/18 3GPP subsets, design §5.4)
  are reported separately** from the full 7-benchmark satellite pass — they
  use the published subset splits (734Q/780Q) that Telco-oRAG/Chat3GPP/
  Telco-RAG report against, not satellite's own TeleQnA sampling. Don't
  conflate the two TeleQnA numbers in the write-up; label which split each
  one is.

## Cost/time note

This is a manual, human-triggered run (README "What this directory does NOT
do"). Budget per design §7: "self-run satellite + TeleQnA runs (3 configs;
mostly on dev credits) ~$20-50". Use `build.nvidia.com` dev credits for
configs 1 and 2's Nano calls (sanctioned use per design §5.6); OTel-2.0
access/cost is whatever its serving provider charges — check before running
all 7 benchmarks against it.
