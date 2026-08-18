"""Diff-mode agent: release-to-release clause diff with cited synthesis.

Pipeline (design §2, diff agent):
  1. resolve_clause_sets — Retriever with release filter, <=15 chunks/side
  2. align clauses by number, title-similarity fallback for renumbering
  3. DETERMINISTIC diff (difflib over normalized lines) rendered first
  4. Super-120B synthesis over the rendered diff, reasoning ON,
     [Ck] citations on both sides, validated by the citation validator

An empty deterministic diff short-circuits before the LLM: the honest
answer is "no changes", and no synthesis can improve on it.
"""
from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field

from openai import OpenAI

from agent.answer import NO_THINK, _content, _degenerate
from agent.tools.retrieval import Retriever, enhance_query
from agent.validators.citations import ValidatorReport, strip_failed_quotes, validate

PER_SIDE = 15               # design cap: <=15 chunks per release side
PARENT_CAP_TOKENS = 900     # per-clause text cap (both sides, symmetric)
MAX_DIFF_LINES = 40         # per-clause unified-diff cap in the render
TITLE_SIM_THRESHOLD = 85.0  # renumbering fallback: title match score

# --- normalization (same neutralizations as the citation validator's _norm,
# but line-wise and case-preserving so the rendered diff stays readable) ---
_MD_NOISE = re.compile(r"[*_`|#]|\\(?=[_*])")
_WS = re.compile(r"\s+")


def _norm_line(s: str) -> str:
    s = _MD_NOISE.sub("", s)
    s = (s.replace("‘", "'").replace("’", "'")
         .replace("“", '"').replace("”", '"')
         .replace("–", "-").replace("—", "-"))
    return _WS.sub(" ", s).strip()


def _norm_lines(text: str) -> list[str]:
    return [n for n in (_norm_line(l) for l in text.splitlines()) if n]


@dataclass
class ClauseSide:
    tag: str        # [Ck] — unique across BOTH sides
    release: str
    clause: str
    title: str
    text: str       # full clause unit (parent text, capped)
    chunk: dict


@dataclass
class ClauseDiff:
    kind: str                   # "added" | "removed" | "changed" | "same"
    a: ClauseSide | None = None
    b: ClauseSide | None = None
    renumbered: bool = False
    diff_lines: list[str] = field(default_factory=list)


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


def _title_sim(a: str, b: str) -> float:
    from rapidfuzz import fuzz
    return fuzz.token_sort_ratio(_norm_line(a).lower(), _norm_line(b).lower())


def align_clauses(side_a: list[ClauseSide], side_b: list[ClauseSide]) -> list[ClauseDiff]:
    """Primary key: clause number. Fallback: best title match >= threshold
    among the leftovers (catches renumbering between releases)."""
    by_a = {s.clause: s for s in side_a}
    by_b = {s.clause: s for s in side_b}
    diffs: list[ClauseDiff] = []

    matched_b: set[str] = set()
    for num, a in by_a.items():
        if num in by_b:
            matched_b.add(num)
            diffs.append(_pair_diff(a, by_b[num], renumbered=False))

    left_a = [a for n, a in by_a.items() if n not in by_b]
    left_b = [b for n, b in by_b.items() if n not in matched_b]
    for a in left_a:
        best, best_score = None, 0.0
        for b in left_b:
            sc = _title_sim(a.title, b.title)
            if sc > best_score:
                best, best_score = b, sc
        if best is not None and best_score >= TITLE_SIM_THRESHOLD:
            left_b.remove(best)
            diffs.append(_pair_diff(a, best, renumbered=True))
        else:
            diffs.append(ClauseDiff(kind="removed", a=a))
    for b in left_b:
        diffs.append(ClauseDiff(kind="added", b=b))
    return diffs


def _pair_diff(a: ClauseSide, b: ClauseSide, renumbered: bool) -> ClauseDiff:
    la, lb = _norm_lines(a.text), _norm_lines(b.text)
    if la == lb:
        return ClauseDiff(kind="same", a=a, b=b, renumbered=renumbered)
    lines = [l for l in difflib.unified_diff(la, lb, lineterm="", n=1)
             if not l.startswith(("---", "+++"))]
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + ["… (diff truncated)"]
    return ClauseDiff(kind="changed", a=a, b=b, renumbered=renumbered,
                      diff_lines=lines)


def render_diff(diffs: list[ClauseDiff], rel_a: str, rel_b: str) -> str:
    """Structured, deterministic diff report — the ground truth the LLM
    synthesizes over (and the artifact shown to the user first)."""
    parts: list[str] = []
    for d in sorted((d for d in diffs if d.kind != "same"),
                    key=lambda d: (d.b or d.a).clause):
        if d.kind == "added":
            parts.append(f"== ADDED in {rel_b}: §{d.b.clause} {d.b.title} {d.b.tag}")
        elif d.kind == "removed":
            parts.append(f"== REMOVED since {rel_a}: §{d.a.clause} {d.a.title} {d.a.tag}")
        else:
            head = (f"== CHANGED: §{d.a.clause} {d.a.title} "
                    f"({d.a.tag} {rel_a} vs {d.b.tag} {rel_b})")
            if d.renumbered:
                head += f" [renumbered §{d.a.clause} -> §{d.b.clause}]"
            parts.append(head + "\n" + "\n".join(d.diff_lines))
    return "\n\n".join(parts)


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
    text = ""
    for extra in ({}, {}, NO_THINK):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=0.0, max_tokens=max_tokens, extra_body=extra)
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
                  acronyms: dict | None = None
                  ) -> tuple[str, ValidatorReport, dict[str, dict], str]:
    """Full diff pipeline. Returns (answer, validator report, tagmap,
    rendered deterministic diff)."""
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
        n = sum(d.kind == "same" for d in diffs)
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
