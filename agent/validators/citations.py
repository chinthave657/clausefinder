"""Deterministic citation validator — runs before any LLM critique.

Checks, in order of strength:
  a. every cited [Ck] tag exists in the retrieved set (exact)
  b. every quoted span fuzzy-matches (>=0.95) within that chunk, with
     re-anchoring to another retrieved chunk when the attribution was off
"""
from __future__ import annotations

import re

from pydantic import BaseModel
from rapidfuzz import fuzz

TAG_RE = re.compile(r"\[C(\d+)\]")
QUOTE_RE = re.compile(r'"([^"]{20,})"')
FUZZ_THRESHOLD = 95.0

_MD_NOISE = re.compile(r"[*_`|>#]|\\(?=[_*])")
_WS = re.compile(r"\s+")
# quote elision markers: "...", "[...]", "[…]", "…" — split and verify each
# segment verbatim; elision is legitimate, invention is not
_ELLIPSIS = re.compile(r"\[?(?:\.{3}|…)\]?")
_BULLET = re.compile(r"\b\d+>\s*")  # 3GPP procedural level markers "1> " "2> "


def _norm(s: str) -> str:
    """Neutralize markdown emphasis, escapes, smart punctuation, line wraps."""
    # _BULLET must run BEFORE _MD_NOISE: the noise class strips '>', which
    # would leave '\d+>' unmatchable and orphan a stray digit token
    s = _BULLET.sub("", s)
    s = _MD_NOISE.sub("", s)
    s = (s.replace("‘", "'").replace("’", "'")
         .replace("“", '"').replace("”", '"')
         .replace("–", "-").replace("—", "-"))
    return _WS.sub(" ", s).strip().lower()


def _quote_score(quote: str, text: str) -> float:
    """Min segment score: every elision-separated fragment must be verbatim."""
    ntext = _norm(text)
    segs = [s for s in (_norm(p) for p in _ELLIPSIS.split(quote)) if len(s) >= 15]
    if not segs:
        return fuzz.partial_ratio(_norm(quote), ntext)
    return min(fuzz.partial_ratio(s, ntext) for s in segs)


class CitationCheck(BaseModel):
    tag: str
    ok: bool
    reason: str = ""


class ValidatorReport(BaseModel):
    checks: list[CitationCheck]
    passed: bool
    uncited_tags: list[str] = []


def strip_failed_quotes(answer: str, retrieved: dict[str, dict]) -> str:
    """Design §3 layer 3: a quote that validates nowhere in the retrieved set
    is removed — the claim keeps its clause reference, loses the fake quote."""
    def _sub(m: re.Match) -> str:
        quote = m.group(1)
        scores = {t: _quote_score(quote, c["text"]) for t, c in retrieved.items()}
        if scores and max(scores.values()) >= FUZZ_THRESHOLD:
            return m.group(0)
        # point the reader at the closest source instead of leaving a husk
        if scores:
            best = max(scores, key=scores.get)
            c = retrieved[best]
            ref = (f"{c['spec']} §{c['clause']} " if c.get("spec") and c.get("clause")
                   else "")
            return f"(quote omitted — see {ref}via {best})"
        return "(quote omitted — no verbatim source)"
    return QUOTE_RE.sub(_sub, answer)


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
        score = _quote_score(quote, retrieved[tag]["text"])
        if score >= FUZZ_THRESHOLD:
            checks.append(CitationCheck(tag=tag, ok=True))
            continue
        # verbatim in a different retrieved chunk? re-anchor: grounding holds,
        # only the tag attribution was off
        others = {t: _quote_score(quote, c["text"])
                  for t, c in retrieved.items() if t != tag}
        best = max(others, key=others.get) if others else None
        if best and others[best] >= FUZZ_THRESHOLD:
            checks.append(CitationCheck(
                tag=best, ok=True,
                reason=f"re-anchored {tag}->{best}"))
            continue
        checks.append(CitationCheck(
            tag=tag, ok=False,
            reason=f"quote not found in any retrieved chunk (best {max(score, *(others.values() or [0])):.0f}): {quote[:40]}…"))

    return ValidatorReport(
        checks=checks,
        passed=all(c.ok for c in checks) and bool(cited),
        uncited_tags=[t for t in retrieved if t.strip("[]C") not in cited],
    )
