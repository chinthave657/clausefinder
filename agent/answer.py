"""Ask-mode answer agent: tag-cited synthesis over retrieved clauses.

Prompt structure follows the measured Telco-RAG format (+4.6pp):
query → context → REPEATED query → instruction. Context capped ~2000 tokens.
LLM via any OpenAI-compatible endpoint (NVIDIA build endpoints for dev,
OpenRouter for the demo). Reasoning is requested OFF for answer calls.
"""
from __future__ import annotations

import os
import re

from openai import OpenAI

from agent.tools.retrieval import Retrieved, Retriever, enhance_query
from agent.validators.citations import ValidatorReport, strip_failed_quotes, validate

# 2000 was Telco-RAG's number for 125-tok chunks; with ~1200-tok parents it
# starves the model to 1-2 sources. 5 parents ≈ 4500-6000 tok ≈ $0.0003 on Nano.
CONTEXT_CAP_TOKENS = 6000

# Abstain floors (design §3 step 6), calibrated 2026-08-19 on the golden set:
# abstain questions score <= 0.110, answerable >= 0.119 on OTel-Reranker-0.6B.
# Two floors because the boundary is thin: below HARD -> refuse outright;
# HARD..SOFT -> answer but lead with an explicit weak-match caveat.
ABSTAIN_HARD = 0.05
ABSTAIN_SOFT = 0.15
ABSTAIN_TIP = (
    "Tip: phrasing the question in 3GPP terminology (procedure names, message "
    "names, TS numbers) usually retrieves better than industry or marketing "
    "terms.")
WEAK_MATCH_CAVEAT = (
    "> **Weak match:** the indexed specs may not directly cover this question — "
    "the excerpts below are the closest material found, but verify the "
    "technology/release context before relying on this answer.\n\n")

# Nemotron-3 is reasoning-default; ask/router run with thinking off (design §2).
# Endpoint-specific knobs (verified 2026-08-21): NVIDIA honors
# chat_template_kwargs; OpenRouter providers vary wildly, so we pin providers
# with clean reasoning-channel behavior, strip via reasoning.exclude, and give
# headroom for billed-but-stripped reasoning tokens. /no_think in SYSTEM covers
# providers that respect in-band directives. _chat's degenerate-retry backstops.
NO_THINK_NVIDIA = {"chat_template_kwargs": {"thinking": False}}
NO_THINK_OPENROUTER = {"reasoning": {"exclude": True},
                       "provider": {"order": ["Novita", "DeepInfra"],
                                    "allow_fallbacks": True}}
MAX_TOKENS_OPENROUTER = 1600  # answer ~900 + stripped reasoning headroom
# App attribution on OpenRouter rankings (free discovery channel)
OR_HEADERS = {"HTTP-Referer": "https://github.com/chinthave657/clausefinder",
              "X-Title": "3GPP ClauseFinder"}
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _content(resp) -> str:
    """Message content with any reasoning block stripped defensively."""
    return _THINK_RE.sub("", resp.choices[0].message.content or "").strip()


_REASONING_TELLS = ("We need to", "The user asks", "The user wants", "Let me ",
                    "I should ", "First, I", "Okay, the user")


def _degenerate(text: str) -> bool:
    """Empty/<unk> output, or reasoning leaked into the content channel
    (provider ignored every suppression knob) — all retryable."""
    return ("<unk>" in text or len(text) < 40
            or text.lstrip().startswith(_REASONING_TELLS))


def _chat(client, model: str, messages: list[dict], max_tokens: int = 900) -> str:
    """One completion; degenerate-output guard (empty/short/<unk> — reasoning
    swallowing the budget, or the NVIDIA long-context serving bug) retries
    twice with the endpoint's no-think config, then once bare."""
    onrouter = "openrouter" in str(client.base_url)
    no_think = NO_THINK_OPENROUTER if onrouter else NO_THINK_NVIDIA
    if onrouter:
        max_tokens = max(max_tokens, MAX_TOKENS_OPENROUTER)
    ladder = ((no_think, no_think, no_think) if onrouter  # bare = reasoning
              else (no_think, no_think, {}))
    for extra in ladder:
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=0.0, max_tokens=max_tokens, extra_body=extra)
        text = _content(resp)
        if not _degenerate(text):
            return text
    return text

# Domain rules below adapted from lugasia/3gpp-skill (MIT) — see NOTICE.
SYSTEM = """/no_think
You are ClauseFinder, a 3GPP standards assistant for telecom engineers.
Rules:
- Answer ONLY from the provided clause excerpts. If they don't contain the answer, say so and name the specs you searched.
- Cite every factual claim inline with its chunk tag, e.g. [C2]. Include one short verbatim quote (<= 1 paragraph, in double quotes) per key claim, anchored to its tag.
- Quotes must be CONTIGUOUS verbatim spans copied exactly from an excerpt: no ellipses (...), no edits, no stitching separate sentences. Never place the user's question in quotes.
- Use precise 3GPP terminology (handover not handoff; UE, gNB; cite specs as TS).
- Distinguish what the standard REQUIRES, what it ALLOWS, and what vendors typically implement.
- Never output configuration commands as authoritative; recommend and cite instead.
- Be concise. No preamble. Do not think step-by-step out loud."""


def _client(api_key: str | None = None) -> tuple[OpenAI, str]:
    """api_key: optional per-request BYO OpenRouter key (web demo session
    state) — overrides env vars for just this call, never persisted."""
    if api_key:
        return (OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key,
                       default_headers=OR_HEADERS),
                os.environ.get("CLAUSEFINDER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free"))
    # NVIDIA first: dev machines carry both keys and evals must spend the
    # dev/eval credit lane; the demo Space carries only OPENROUTER_API_KEY.
    if os.environ.get("NVIDIA_API_KEY"):
        return (OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                       api_key=os.environ["NVIDIA_API_KEY"]),
                os.environ.get("CLAUSEFINDER_MODEL", "nvidia/nemotron-3-nano-30b-a3b"))
    if os.environ.get("OPENROUTER_API_KEY"):
        return (OpenAI(base_url="https://openrouter.ai/api/v1",
                       api_key=os.environ["OPENROUTER_API_KEY"],
                       default_headers=OR_HEADERS),
                os.environ.get("CLAUSEFINDER_MODEL", "nvidia/nemotron-3-nano-30b-a3b"))
    raise SystemExit("set OPENROUTER_API_KEY or NVIDIA_API_KEY")


def _context_block(retriever: Retriever, hits: list[Retrieved]) -> tuple[str, dict[str, dict]]:
    parts, tagmap, tok = [], {}, 0
    for r in hits:
        parent = retriever.parent_text(r)
        t = int(len(parent.split()) * 1.3)
        if tok + t > CONTEXT_CAP_TOKENS and parts:
            break
        c = r.chunk
        header = f"{r.tag} {c['spec']} §{c['clause']} {c['clause_title']} ({c['release']} V{c['version']})"
        parts.append(f"{header}\n{parent}")
        tagmap[r.tag] = {**c, "text": parent}
        tok += t
    return "\n\n---\n\n".join(parts), tagmap


def ask(question: str, retriever: Retriever, release: str | None = None,
        acronyms: dict | None = None,
        api_key: str | None = None,
        hits: list | None = None) -> tuple[str, ValidatorReport, dict[str, dict]]:
    """hits: optionally precomputed retrieval results — lets GPU-quota hosts
    (ZeroGPU) run retrieval inside the GPU window and LLM synthesis outside."""
    if hits is None:
        q = enhance_query(question, acronyms or {})
        hits = retriever.search(q, release=release)
    if not hits:
        return ("No supporting clauses found in the indexed specs.",
                ValidatorReport(checks=[], passed=False), {})

    # Abstain rail: cross-encoder confidence gate (only meaningful w/ rerank on)
    top_rr = max((h.rerank_score for h in hits if h.rerank_score is not None),
                 default=None)
    if top_rr is not None and top_rr < ABSTAIN_HARD:
        searched = sorted({h.chunk["spec"] for h in hits})
        return (f"The indexed specifications don't appear to cover this "
                f"question (best relevance {top_rr:.2f}). Closest specs "
                f"searched: {', '.join(searched[:6])}. If the answer lives in "
                f"a spec series ClauseFinder doesn't index (e.g. 36-series "
                f"LTE, O-RAN, 3GPP2, IEEE), that's why — see the README for "
                f"how to self-host with additional series. " + ABSTAIN_TIP,
                ValidatorReport(checks=[], passed=True), {})
    weak = top_rr is not None and top_rr < ABSTAIN_SOFT

    context, tagmap = _context_block(retriever, hits)

    user = (f"Question: {question}\n\n"
            f"Clause excerpts:\n{context}\n\n"
            f"Question (again): {question}\n\n"
            "Answer with inline [Ck] citations and verbatim quotes per the rules.")

    client, model = _client(api_key)
    answer = _chat(client, model,
                   [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user}])

    report = validate(answer, tagmap)
    if not report.passed:
        answer2 = _chat(client, model, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
            {"role": "user", "content":
                ("Your answer contains NO [Ck] citation tags. Every factual claim needs one. "
                 if not report.checks else "Some citations failed validation:\n"
                 + "\n".join(c.reason for c in report.checks if not c.ok))
                + "\nRewrite the answer fixing ONLY the failed citations/quotes. "
                  "Quote text verbatim from the excerpts or drop the quote."},
        ])
        report2 = validate(answer2, tagmap)
        if report2.passed or sum(c.ok for c in report2.checks) > sum(c.ok for c in report.checks):
            answer, report = answer2, report2
    if not report.passed:
        answer = strip_failed_quotes(answer, tagmap)
        report = validate(answer, tagmap)
    if weak:
        answer = WEAK_MATCH_CAVEAT + answer
    return answer, report, tagmap
