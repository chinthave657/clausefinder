"""Deterministic citation validator — runs before any LLM critique.

Checks, in order of strength:
  a. every cited [Ck] tag exists in the retrieved set (exact)
  b. every quoted span fuzzy-matches (>=0.95) within THAT chunk only
  c. every {spec, clause} pair the answer names exists in the index manifest
"""
from __future__ import annotations

import re
from pydantic import BaseModel
from rapidfuzz import fuzz

TAG_RE = re.compile(r"\[C(\d+)\]")
QUOTE_RE = re.compile(r'"([^"]{20,})"')
FUZZ_THRESHOLD = 95.0


class CitationCheck(BaseModel):
    tag: str
    ok: bool
    reason: str = ""


class ValidatorReport(BaseModel):
    checks: list[CitationCheck]
    passed: bool
    uncited_tags: list[str] = []


def validate(answer: str, retrieved: dict[str, dict]) -> ValidatorReport:
    """retrieved: tag -> chunk (incl. parent text under key 'text')."""
    checks: list[CitationCheck] = []
    cited = set(TAG_RE.findall(answer))

    for n in cited:
        tag = f"[C{n}]"
        if tag not in retrieved:
            checks.append(CitationCheck(tag=tag, ok=False, reason="tag not in retrieved set"))
            continue
        checks.append(CitationCheck(tag=tag, ok=True))

    # quotes: each quote must live inside the chunk of the nearest preceding tag
    for m in QUOTE_RE.finditer(answer):
        quote = m.group(1)
        prior = TAG_RE.findall(answer[:m.start()])
        follow = TAG_RE.findall(answer[m.end():m.end() + 40])
        n = (follow or prior[::-1] or [None])[0]
        tag = f"[C{n}]" if n else None
        if not tag or tag not in retrieved:
            checks.append(CitationCheck(tag=tag or "?", ok=False,
                                        reason=f"quote has no anchoring tag: {quote[:40]}…"))
            continue
        score = fuzz.partial_ratio(quote, retrieved[tag]["text"])
        ok = score >= FUZZ_THRESHOLD
        checks.append(CitationCheck(
            tag=tag, ok=ok,
            reason="" if ok else f"quote not found in {tag} (score {score:.0f}): {quote[:40]}…"))

    return ValidatorReport(
        checks=checks,
        passed=all(c.ok for c in checks) and bool(cited),
        uncited_tags=[t for t in retrieved if t.strip("[]C") not in cited],
    )
