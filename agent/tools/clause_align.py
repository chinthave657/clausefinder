"""Deterministic clause-set alignment between two releases (design §3, diff agent).

Given per-release clause sides (from retrieval hits or an explicit spec+clause
request), aligns pairs by clause number; a clause missing under its own number
on the other side falls back to title-similarity matching (rapidfuzz
token_set_ratio >= TITLE_SIM_THRESHOLD) to catch renumbering across releases.
Each pair is classified unchanged / modified / added / removed. The diff
itself is a normalized `difflib.unified_diff` over text run through
`agent.validators.citations._norm` — the same normalization the citation
validator uses, so rendered diff lines and validator quote-matching agree on
what counts as "the same text".
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Literal

from rapidfuzz import fuzz

from agent.validators.citations import _norm

TITLE_SIM_THRESHOLD = 80.0  # design: rapidfuzz token_set_ratio >= 80
MAX_DIFF_LINES = 40         # per-clause unified-diff cap in the render

Kind = Literal["unchanged", "modified", "added", "removed"]


@dataclass
class ClauseSide:
    tag: str        # [Ck] — unique across BOTH sides
    release: str
    clause: str
    title: str
    text: str       # full clause unit (parent text, capped by the caller)
    chunk: dict = field(default_factory=dict)


@dataclass
class ClausePair:
    kind: Kind
    a: ClauseSide | None = None
    b: ClauseSide | None = None
    renumbered: bool = False
    diff_lines: list[str] = field(default_factory=list)


def _title_sim(a: str, b: str) -> float:
    return fuzz.token_set_ratio(_norm(a), _norm(b))


def _norm_lines(text: str) -> list[str]:
    return [n for n in (_norm(l) for l in text.splitlines()) if n]


def _pair(a: ClauseSide, b: ClauseSide, renumbered: bool) -> ClausePair:
    la, lb = _norm_lines(a.text), _norm_lines(b.text)
    if la == lb:
        return ClausePair(kind="unchanged", a=a, b=b, renumbered=renumbered)
    lines = [l for l in difflib.unified_diff(la, lb, lineterm="", n=1)
             if not l.startswith(("---", "+++"))]
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + ["… (diff truncated)"]
    return ClausePair(kind="modified", a=a, b=b, renumbered=renumbered, diff_lines=lines)


def align_clauses(side_a: list[ClauseSide], side_b: list[ClauseSide]) -> list[ClausePair]:
    """Primary key: clause number, exact match. Fallback: best title match
    >= TITLE_SIM_THRESHOLD among the leftovers on both sides (renumbering).
    Unmatched leftovers become removed (a-only) / added (b-only)."""
    by_a = {s.clause: s for s in side_a}
    by_b = {s.clause: s for s in side_b}
    pairs: list[ClausePair] = []

    matched_b: set[str] = set()
    for num, a in by_a.items():
        if num in by_b:
            matched_b.add(num)
            pairs.append(_pair(a, by_b[num], renumbered=False))

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
            pairs.append(_pair(a, best, renumbered=True))
        else:
            pairs.append(ClausePair(kind="removed", a=a))
    for b in left_b:
        pairs.append(ClausePair(kind="added", b=b))
    return pairs


def render_diff(pairs: list[ClausePair], rel_a: str, rel_b: str) -> str:
    """Structured, deterministic diff report — the ground truth the LLM
    synthesizes over (and the artifact shown to the user first). Unchanged
    pairs are omitted; empty return means no relevant difference."""
    parts: list[str] = []
    for p in sorted((p for p in pairs if p.kind != "unchanged"),
                    key=lambda p: (p.b or p.a).clause):
        if p.kind == "added":
            parts.append(f"== ADDED in {rel_b}: §{p.b.clause} {p.b.title} {p.b.tag}")
        elif p.kind == "removed":
            parts.append(f"== REMOVED since {rel_a}: §{p.a.clause} {p.a.title} {p.a.tag}")
        else:
            head = (f"== MODIFIED: §{p.a.clause} {p.a.title} "
                    f"({p.a.tag} {rel_a} vs {p.b.tag} {rel_b})")
            if p.renumbered:
                head += f" [renumbered §{p.a.clause} -> §{p.b.clause}]"
            parts.append(head + "\n" + "\n".join(p.diff_lines))
    return "\n\n".join(parts)
