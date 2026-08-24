"""Diff-mode agent: release-to-release clause diff with cited synthesis.

Pipeline (design §2, diff agent):
  1. resolve_clause_sets — Retriever with release filter, <=15 chunks/side
  2. align_clauses (agent.tools.clause_align) — clause-number match, title-
     similarity fallback for renumbering
  3. DETERMINISTIC diff (difflib over normalized lines) rendered first
  4. Super-120B synthesis over the rendered diff, reasoning ON,
     [Ck] citations on both sides, validated by the citation validator

An empty deterministic diff short-circuits before the LLM: the honest
answer is "no changes", and no synthesis can improve on it.
"""
from __future__ import annotations

import os

from openai import OpenAI

from agent.answer import NO_THINK, _content, _degenerate
from agent.tools.clause_align import ClausePair, ClauseSide, align_clauses, render_diff
from agent.tools.retrieval import Retriever, enhance_query
from agent.validators.citations import ValidatorReport, strip_failed_quotes, validate

PER_SIDE = 15               # design cap: <=15 chunks per release side
PARENT_CAP_TOKENS = 900     # per-clause text cap (both sides, symmetric)

__all__ = ["ClausePair", "ClauseSide", "resolve_clause_sets", "diff_releases"]


def resolve_clause_sets(query: str, releases: tuple[str, str], retriever: Retriever,
                        acronyms: dict | None = None,
                        per_side: int = PER_SIDE) -> dict[str, list[ClauseSide]]:
    """One release-filtered retrieval per side; tags are assigned sequentially
    across side A then side B so every [Ck] is globally unique."""
    q = enhance_query(query, acronyms or {})
    # self-diff smoke case: one retrieval serves both sides (distinct keys/tags)
    rels = releases if releases[0] != releases[1] else (releases[0] + "#a",
                                                       releases[1] + "#b")
    sides: dict[str, list[ClauseSide]] = {}
    k = 0
    for key, rel in zip(rels, releases):
        hits = retriever.search(q, release=rel, top=per_side)[:per_side]
        out = []
        for h in hits:
            k += 1
            out.append(ClauseSide(
                tag=f"[C{k}]", release=rel,
                clause=h.chunk["clause"], title=h.chunk["clause_title"],
                text=retriever.parent_text(h, cap_tokens=PARENT_CAP_TOKENS),
                chunk=h.chunk))
        sides[key] = out
    return sides


# --- synthesis -----------------------------------------------------------

SYSTEM = """You are ClauseFinder in DIFF MODE, comparing two 3GPP releases for telecom engineers.
Rules:
- You are given clause excerpts from BOTH releases and a DETERMINISTIC diff computed over them. Report ONLY differences supported by that diff and those excerpts. Do not invent changes.
- Cite every claim inline with its chunk tag, e.g. [C2]. When describing a change, cite BOTH sides (old-release tag and new-release tag) where both exist.
- Include one short verbatim quote (<= 1 paragraph, in double quotes) per key change, anchored to its tag. Quotes must be CONTIGUOUS verbatim spans copied exactly from an excerpt: no ellipses, no edits, no stitching.
- Structure the answer as: added / removed / changed, most significant first.
- Use precise 3GPP terminology (UE, gNB, cite specs as TS).
- If the diff shows no differences relevant to the question, say so plainly.
- Be concise. No preamble."""


def _client() -> tuple[OpenAI, str]:
    if os.environ.get("OPENROUTER_API_KEY"):
        return (OpenAI(base_url="https://openrouter.ai/api/v1",
                       api_key=os.environ["OPENROUTER_API_KEY"]),
                os.environ.get("CLAUSEFINDER_DIFF_MODEL", "nvidia/nemotron-3-super-120b-a12b"))
    if os.environ.get("NVIDIA_API_KEY"):
        return (OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                       api_key=os.environ["NVIDIA_API_KEY"]),
                os.environ.get("CLAUSEFINDER_DIFF_MODEL", "nvidia/nemotron-3-super-120b-a12b"))
    raise SystemExit("set OPENROUTER_API_KEY or NVIDIA_API_KEY")


def _chat(client, model: str, messages: list[dict], max_tokens: int = 4096) -> str:
    """Reasoning ON (the one thinking-on call in the system — design §2):
    no NO_THINK on the first two attempts. Degenerate-output retry mirrors
    answer.py; the last-resort attempt flips thinking off."""
    # last rung drops the provider pin NO_THINK may carry — the pinned
    # providers serve the answer models, not necessarily Super-120B
    last = {k: v for k, v in NO_THINK.items() if k != "provider"} or NO_THINK
    text = ""
    for extra in ({}, {}, last):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, max_tokens=max_tokens, extra_body=extra)
        except Exception:  # provider unavailable for this model — next rung
            continue
        text = _content(resp)
        if not _degenerate(text):
            return text
    return text


def _context_block(sides: dict[str, list[ClauseSide]]) -> tuple[str, dict[str, dict]]:
    parts, tagmap = [], {}
    for rel_sides in sides.values():
        for s in rel_sides:
            c = s.chunk
            header = (f"{s.tag} {c['spec']} §{s.clause} {s.title} "
                      f"({s.release} V{c['version']})")
            parts.append(f"{header}\n{s.text}")
            tagmap[s.tag] = {**c, "text": s.text}
    return "\n\n---\n\n".join(parts), tagmap


def diff_releases(question: str, retriever: Retriever, release_a: str, release_b: str,
                  acronyms: dict | None = None, sides: dict | None = None
                  ) -> tuple[str, ValidatorReport, dict[str, dict], str]:
    """Full diff pipeline. Returns (answer, validator report, tagmap,
    rendered deterministic diff). sides: optionally precomputed clause sets —
    lets ZeroGPU hosts run retrieval inside the GPU window, synthesis outside."""
    if sides is None:
        sides = resolve_clause_sets(question, (release_a, release_b), retriever, acronyms)
    keys = list(sides)
    side_a, side_b = sides[keys[0]], sides[keys[-1]]

    for rel, side in ((release_a, side_a), (release_b, side_b)):
        if not side:
            return (f"No supporting clause found in the indexed specs for {rel}.",
                    ValidatorReport(checks=[], passed=False), {}, "")

    diffs = align_clauses(side_a, side_b)
    rendered = render_diff(diffs, release_a, release_b)
    if not rendered:
        n = sum(d.kind == "unchanged" for d in diffs)
        return (f"No changes: the {n} clause(s) retrieved for this question are "
                f"textually identical between {release_a} and {release_b}.",
                ValidatorReport(checks=[], passed=True), {}, rendered)

    context, tagmap = _context_block(sides)
    user = (f"Question: {question}\n"
            f"Compare {release_a} (old) vs {release_b} (new).\n\n"
            f"Clause excerpts (both releases):\n{context}\n\n"
            f"Deterministic diff ({release_a} -> {release_b}):\n{rendered}\n\n"
            f"Question (again): {question}\n\n"
            "Summarize the release differences relevant to the question, with inline "
            "[Ck] citations on both sides and verbatim quotes per the rules.")

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
                ("Your answer contains NO [Ck] citation tags. Every claim needs one. "
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
    return answer, report, tagmap, rendered
