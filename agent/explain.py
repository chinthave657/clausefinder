"""Explain-mode agent: plain-English explanation of ONE named clause.

No vector search — the clause reference ("TS 38.331 5.3.5.3") is resolved
directly against LanceDB metadata (spec + clause equality, subclause prefix
fallback). Same Nano model, same NO_THINK call path, same citation validator
as ask-mode (design §2: reasoning off everywhere except diff synthesis).
"""
from __future__ import annotations

import re

from agent.answer import NO_THINK, _chat, _client  # shared call path incl. degenerate retry
from agent.tools.retrieval import Retrieved, Retriever
from agent.validators.citations import ValidatorReport, strip_failed_quotes, validate

_ = NO_THINK  # re-exported intent: explain runs thinking-off via _chat

# "TS 38.331 5.3.5.3" | "ts38.331 §5.3.5.3" | "38.331 clause 5.3.5.3"
REF_RE = re.compile(
    r"(?:TS\s*)?(\d{2}\.\d{3})(?:\s*(?:§|clause)\s*|\s+)([\dA-Za-z]+(?:\.[\dA-Za-z]+)*)",
    re.IGNORECASE)

MAX_CHUNKS = 12          # a clause unit is rarely bigger; keeps context ~3k tok
CAP_TOKENS = 6000

SYSTEM = """You are ClauseFinder, a 3GPP standards assistant for telecom engineers.
You will be given the full text of ONE spec clause, split into tagged excerpts.
Rules:
- Explain what this clause specifies in plain English for an engineer who has not read it.
- Ground EVERY claim in the excerpts: cite inline with the chunk tag, e.g. [C2], and include one short verbatim quote (<= 1 paragraph, in double quotes) per key point, anchored to its tag.
- Quotes must be CONTIGUOUS verbatim spans copied exactly from an excerpt: no ellipses (...), no edits, no stitching separate sentences.
- Use precise 3GPP terminology (handover not handoff; UE, gNB; cite specs as TS).
- Distinguish what the clause REQUIRES (shall), what it ALLOWS (may), and what is left to implementation.
- Be concise. No preamble. Do not think step-by-step out loud."""


def parse_ref(ref: str) -> tuple[str, str] | None:
    """'TS 38.331 5.3.5.3' -> ('TS 38.331', '5.3.5.3'); None if unparseable."""
    m = REF_RE.search(ref.strip())
    if not m:
        return None
    return f"TS {m.group(1)}", m.group(2).rstrip(".")


def fetch_clause(retriever: Retriever, spec: str, clause: str,
                 release: str | None = None) -> list[dict]:
    """Clause chunks straight from LanceDB metadata — exact clause first,
    then subclause prefix (gold 5.2.1 is answered by 5.2.1.3 chunks)."""
    rel = f" AND release = '{release}'" if release else ""
    exact = (retriever.tbl.search()
             .where(f"spec = '{spec}' AND clause = '{clause}'{rel}")
             .limit(MAX_CHUNKS).to_list())
    if not exact:
        exact = (retriever.tbl.search()
                 .where(f"spec = '{spec}' AND clause LIKE '{clause}.%'{rel}")
                 .limit(MAX_CHUNKS).to_list())
    exact.sort(key=lambda c: (c["clause"], c["id"]))
    return exact


def explain(ref: str, retriever: Retriever,
            release: str | None = None) -> tuple[str, ValidatorReport, dict[str, dict]]:
    """Same return shape as answer.ask(): (text, report, tagmap)."""
    parsed = parse_ref(ref)
    if not parsed:
        return (f"Could not parse a clause reference from '{ref}'. "
                "Use the form 'TS 38.331 5.3.5.3'.",
                ValidatorReport(checks=[], passed=False), {})
    spec, clause = parsed
    chunks = fetch_clause(retriever, spec, clause, release=release)
    if not chunks:
        where = f"{spec} (release {release})" if release else spec
        return (f"No clause {clause} found in the indexed copy of {where}.",
                ValidatorReport(checks=[], passed=False), {})

    parts, tagmap, tok = [], {}, 0
    for i, c in enumerate(chunks):
        t = int(len(c["text"].split()) * 1.3)
        if tok + t > CAP_TOKENS and parts:
            break
        tag = f"[C{i + 1}]"
        header = (f"{tag} {c['spec']} §{c['clause']} {c['clause_title']} "
                  f"({c['release']} V{c['version']})")
        parts.append(f"{header}\n{c['text']}")
        tagmap[tag] = {k: v for k, v in c.items() if k != "vector"}
        tok += t
    context = "\n\n---\n\n".join(parts)

    user = (f"Clause to explain: {spec} §{clause}\n\n"
            f"Clause text:\n{context}\n\n"
            f"Explain {spec} §{clause} in plain English "
            "with inline [Ck] citations and verbatim quotes per the rules.")

    client, model = _client()
    text = _chat(client, model,
                 [{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}])

    report = validate(text, tagmap)
    if not report.passed:
        text2 = _chat(client, model, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": text},
            {"role": "user", "content":
                ("Your answer contains NO [Ck] citation tags. Every claim needs one. "
                 if not report.checks else "Some citations failed validation:\n"
                 + "\n".join(c.reason for c in report.checks if not c.ok))
                + "\nRewrite, fixing ONLY the failed citations/quotes. "
                  "Quote text verbatim from the excerpts or drop the quote."},
        ])
        report2 = validate(text2, tagmap)
        if report2.passed or sum(c.ok for c in report2.checks) > sum(c.ok for c in report.checks):
            text, report = text2, report2
    if not report.passed:
        text = strip_failed_quotes(text, tagmap)
        report = validate(text, tagmap)
    return text, report, tagmap


__all__ = ["explain", "parse_ref", "fetch_clause", "Retrieved"]
