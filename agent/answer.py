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

from agent.tools.retrieval import Retriever, Retrieved, enhance_query
from agent.validators.citations import ValidatorReport, strip_failed_quotes, validate

# 2000 was Telco-RAG's number for 125-tok chunks; with ~1200-tok parents it
# starves the model to 1-2 sources. 5 parents ≈ 4500-6000 tok ≈ $0.0003 on Nano.
CONTEXT_CAP_TOKENS = 6000

# Nemotron-3 is reasoning-default; ask/router run with thinking off (design §2).
NO_THINK = {"chat_template_kwargs": {"thinking": False}}
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _content(resp) -> str:
    """Message content with any reasoning block stripped defensively."""
    return _THINK_RE.sub("", resp.choices[0].message.content or "").strip()


def _degenerate(text: str) -> bool:
    return "<unk>" in text or len(text) < 40


def _chat(client, model: str, messages: list[dict], max_tokens: int = 900) -> str:
    """One completion; on degenerate output (<unk> runs — intermittent serving
    bug under long context) retry once, then once more without NO_THINK."""
    for extra in (NO_THINK, NO_THINK, {}):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=0.0, max_tokens=max_tokens, extra_body=extra)
        text = _content(resp)
        if not _degenerate(text):
            return text
    return text

SYSTEM = """You are ClauseFinder, a 3GPP standards assistant for telecom engineers.
Rules:
- Answer ONLY from the provided clause excerpts. If they don't contain the answer, say so and name the specs you searched.
- Cite every factual claim inline with its chunk tag, e.g. [C2]. Include one short verbatim quote (<= 1 paragraph, in double quotes) per key claim, anchored to its tag.
- Quotes must be CONTIGUOUS verbatim spans copied exactly from an excerpt: no ellipses (...), no edits, no stitching separate sentences. Never place the user's question in quotes.
- Use precise 3GPP terminology (handover not handoff; UE, gNB; cite specs as TS).
- Distinguish what the standard REQUIRES, what it ALLOWS, and what vendors typically implement.
- Never output configuration commands as authoritative; recommend and cite instead.
- Be concise. No preamble. Do not think step-by-step out loud."""


def _client() -> tuple[OpenAI, str]:
    if os.environ.get("OPENROUTER_API_KEY"):
        return (OpenAI(base_url="https://openrouter.ai/api/v1",
                       api_key=os.environ["OPENROUTER_API_KEY"]),
                os.environ.get("CLAUSEFINDER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free"))
    if os.environ.get("NVIDIA_API_KEY"):
        return (OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                       api_key=os.environ["NVIDIA_API_KEY"]),
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
        acronyms: dict | None = None) -> tuple[str, ValidatorReport, dict[str, dict]]:
    q = enhance_query(question, acronyms or {})
    hits = retriever.search(q, release=release)
    if not hits:
        return ("No supporting clauses found in the indexed specs.",
                ValidatorReport(checks=[], passed=False), {})
    context, tagmap = _context_block(retriever, hits)

    user = (f"Question: {question}\n\n"
            f"Clause excerpts:\n{context}\n\n"
            f"Question (again): {question}\n\n"
            "Answer with inline [Ck] citations and verbatim quotes per the rules.")

    client, model = _client()
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
    return answer, report, tagmap
