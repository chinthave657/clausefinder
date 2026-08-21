---
title: 3GPP ClauseFinder
emoji: 📡
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: "6.24.0"
python_version: "3.11"
app_file: web/app.py
startup_duration_timeout: 30m
license: apache-2.0
short_description: Clause-cited 3GPP answers, diffs, explanations
tags:
  - telecom
  - 3gpp
  - rag
  - nvidia
  - nemotron
---

# ClauseFinder

Clause-cited 3GPP answers and release diffs on the NVIDIA agentic stack
(Nemotron-3 Nano/Super via the NVIDIA API, OTel-Embedding-568M, LanceDB
hybrid retrieval with weighted RRF).

## Modes

| Tab | What it does |
|-----|--------------|
| **Ask** | Hybrid BM25 ⊕ vector retrieval over indexed 3GPP specs, Nemotron-3 Nano synthesis, deterministic citation validator (verbatim-quote fuzzy match ≥95) |
| **Diff** | Release-to-release clause diff: deterministic normalized diff first, then Nemotron-3 Super-120B synthesis with citations on both sides |
| **Explain** | Enter a clause reference (e.g. `TS 38.331 5.3.5.3`); the clause is fetched directly by metadata — no vector search — and explained in plain English with validated citations |

Every answer carries inline `[Ck]` citations. The sources panel lists each
cited chunk's spec / clause / release / version with a link to the official
3GPP spec page, and the validator report shows per-citation badges:
green = verbatim quote verified, amber = paraphrased or re-anchored.

## Configuration

Set `NVIDIA_API_KEY` (or `OPENROUTER_API_KEY`) as a Space secret. The
LanceDB index is expected at `data/index/` (distributed as a tarball from a
gated HF dataset and unpacked at boot).

Embedding of the user query runs in a `@spaces.GPU` function on ZeroGPU;
the app also runs on plain CPU (locally: `uv run python web/app.py`).

> **Answers cite official 3GPP specs — verify against the referenced clause
> before use.** ClauseFinder is a research assistant, not a normative source.
