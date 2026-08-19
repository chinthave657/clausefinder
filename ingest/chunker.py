"""Clause-aware chunking of GSMA/3GPP markdown.

Clause numbers live in heading TEXT; markdown heading levels are inconsistent
in the GSMA conversion (e.g. `## 5.2.2` next to `### 4.2.1`), so hierarchy is
derived from the clause number itself, never from `#` depth.

Output: parent units (one per clause, capped) and child chunks (retrieval
units, 200-350 tokens, breadcrumb-prefixed) plus cross-reference edges.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

# `## 5.2.2.3.3a Title` / `## A.1 Title` / `## Annex B (informative): Title`
CLAUSE_RE = re.compile(r"^#{1,6}\s+\**((?:\d+|[A-Z])(?:\.\d+)*[a-z]?)\**\s+(.+?)\s*$")
ANNEX_RE = re.compile(r"^#{1,6}\s+\**Annex\s+([A-Z])\**\s*(?:\((normative|informative)\))?:?\s*(.*?)\s*$", re.I)
# GSMA markdown has two header shapes:
#   (a) "# 3GPP TS 38.331 V18.5.0"          — single heading line
#   (b) "**3GPP TS**\n\n**24.501 V18.5.0**" — split bold lines (~24% of specs)
TITLE_RE = re.compile(
    r"(?:#\s+|\*\*)3GPP\s+(TS|TR)(?:\*\*)?\s+(?:\*\*)?([\d.\-]+)\s+V([\d.]+)",
    re.DOTALL)
XREF_CLAUSE_RE = re.compile(r"\bclauses?\s+((?:\d+|[A-Z])(?:\.\d+)+[a-z]?)", re.I)
XREF_SPEC_RE = re.compile(r"\bTS\s?(\d{2}\.\d{3}(?:-\d+)?)")

SKIP_TITLES = {"contents", "foreword", "references"}
CHILD_TARGET = 300   # tokens (approx)
CHILD_MAX = 350
PARENT_MAX = 1200
EMBED_MAX = 1400     # OTel-568M fine-tune length guard


def ntokens(text: str) -> int:
    """Cheap token estimate (~1.3 tokens/word for technical text)."""
    return int(len(text.split()) * 1.3)


class Chunk(BaseModel):
    id: str
    parent_id: str | None = None
    spec: str            # "TS 38.331"
    series: int
    release: str         # "Rel-18"
    version: str         # "18.0.0"
    clause: str          # "5.2.2.1" or "A.1"
    clause_title: str
    breadcrumb: str
    text: str
    is_table: bool = False
    annex: bool = False
    normative: bool = True
    over_embed_limit: bool = False
    url: str = ""


class XrefEdge(BaseModel):
    src_spec: str
    src_clause: str
    dst_spec: str        # same spec unless a TS reference is nearby
    dst_clause: str


@dataclass
class _Section:
    clause: str
    title: str
    lines: list[str] = field(default_factory=list)
    annex: bool = False
    normative: bool = True


def _clause_rank(clause: str) -> list[str]:
    return clause.split(".")


def _spec_url(spec_digits: str) -> str:
    # Deep link to the official 3GPP spec page.
    return f"https://www.3gpp.org/DynaReport/{spec_digits.replace('.', '').replace('-', '')}.htm"


def parse_spec(md_path: Path, release: str, series: int) -> tuple[list[Chunk], list[XrefEdge], dict]:
    text = md_path.read_text(errors="replace")
    lines = text.splitlines()

    # Identity: prefer the GSMA path when the parent dir is a spec dir
    # (".../<series>_series/<digits>/raw.md") — header formats vary too much
    # to trust (single-line, split-bold, broken Word-field conversions all
    # occur; trusting headers silently skipped 290 specs incl. TS 24.501).
    # Non-GSMA layouts (tests, ad-hoc files) fall back to header parsing.
    raw_digits = md_path.parent.name          # e.g. "24501" or "38523-3"
    m = TITLE_RE.search(text[:2000])
    if re.fullmatch(r"\d{5}(-\d+)?", raw_digits):
        base, _, suffix = raw_digits.partition("-")
        spec_digits = f"{base[:2]}.{base[2:]}" + (f"-{suffix}" if suffix else "")
        if m:
            doc_type, _, version = m.groups()
        else:
            doc_type = "TR" if "Technical Report" in text[:3000] else "TS"
            vm = re.search(r"\bV(\d+\.\d+\.\d+)", text[:5000])
            version = vm.group(1) if vm else "unknown"
    else:
        if not m:
            raise ValueError(f"no 3GPP title line in {md_path}")
        doc_type, spec_digits, version = m.groups()
    spec = f"{doc_type} {spec_digits}"
    url = _spec_url(spec_digits)

    # --- split into clause sections -------------------------------------
    sections: list[_Section] = []
    cur: _Section | None = None
    annex_ctx: tuple[str, bool] | None = None  # (letter, normative)

    for ln in lines:
        am = ANNEX_RE.match(ln)
        cm = CLAUSE_RE.match(ln) if not am else None
        if am:
            letter, kind, title = am.groups()
            annex_ctx = (letter, (kind or "").lower() != "informative")
            cur = _Section(letter, title or f"Annex {letter}", annex=True, normative=annex_ctx[1])
            sections.append(cur)
            continue
        if cm:
            clause, title = cm.group(1), cm.group(2).replace("*", "").strip()
            is_annex = bool(annex_ctx) and clause[0].isalpha()
            normative = annex_ctx[1] if is_annex and annex_ctx else True
            if not clause[0].isalpha():
                annex_ctx = None
                is_annex, normative = False, True
            cur = _Section(clause, title, annex=is_annex, normative=normative)
            sections.append(cur)
            continue
        if cur is not None:
            cur.lines.append(ln)

    # --- build chunks ----------------------------------------------------
    chunks: list[Chunk] = []
    edges: list[XrefEdge] = []
    over_limit = 0
    path: list[tuple[str, str]] = []  # (clause, title) stack for breadcrumbs

    for sec in sections:
        # ancestors are strict clause-number prefixes (5.3.5.3 ⇒ 5 > 5.3 > 5.3.5);
        # maintain the stack for EVERY section — container clauses often have no body
        while path and not (sec.clause + ".").startswith(path[-1][0] + "."):
            path.pop()
        crumb_mid = " > ".join(f"{c} {t}" for c, t in path)
        path.append((sec.clause, sec.title))

        if sec.title.strip().lower() in SKIP_TITLES:
            continue
        body = "\n".join(sec.lines).strip()
        if not body:
            continue

        breadcrumb = f"{spec} {release} — " + (f"{crumb_mid} > " if crumb_mid else "") + f"{sec.clause} {sec.title}"
        pid = hashlib.sha1(f"{spec}|{release}|{sec.clause}".encode()).hexdigest()[:16]

        # xref edges from this clause's body
        for xm in XREF_CLAUSE_RE.finditer(body):
            ctx = body[max(0, xm.start() - 60):xm.start()]
            sm = XREF_SPEC_RE.search(ctx)
            edges.append(XrefEdge(
                src_spec=spec, src_clause=sec.clause,
                dst_spec=f"TS {sm.group(1)}" if sm else spec,
                dst_clause=xm.group(1),
            ))

        common = dict(
            parent_id=pid, spec=spec, series=series, release=release,
            version=version, clause=sec.clause, clause_title=sec.title,
            breadcrumb=breadcrumb, annex=sec.annex, normative=sec.normative, url=url,
        )

        for i, (piece, is_table) in enumerate(_split_children(body)):
            over = ntokens(breadcrumb + piece) > EMBED_MAX
            over_limit += over
            chunks.append(Chunk(id=f"{pid}-{i}", text=piece, is_table=is_table,
                                over_embed_limit=over, **common))

    stats = {
        "spec": spec, "release": release, "version": version,
        "clauses": len(sections), "chunks": len(chunks),
        "edges": len(edges), "over_embed_limit": over_limit,
    }
    return chunks, edges, stats


def _split_children(body: str) -> list[tuple[str, bool]]:
    """Split clause body into 200-350-token pieces; tables stay atomic."""
    blocks: list[tuple[str, bool]] = []   # (block, is_table)
    cur_lines: list[str] = []
    in_table = False
    for ln in body.splitlines():
        is_t = ln.lstrip().startswith("|")
        if is_t != in_table and cur_lines:
            blocks.append(("\n".join(cur_lines).strip(), in_table))
            cur_lines = []
        in_table = is_t
        cur_lines.append(ln)
    if cur_lines:
        blocks.append(("\n".join(cur_lines).strip(), in_table))

    out: list[tuple[str, bool]] = []
    buf, buf_tok = [], 0
    for block, is_table in blocks:
        if not block:
            continue
        if is_table:
            if buf:
                out.append(("\n".join(buf), False))
                buf, buf_tok = [], 0
            out.append((block, True))
            continue
        for para in re.split(r"\n{2,}", block):
            t = ntokens(para)
            if buf_tok + t > CHILD_MAX and buf:
                out.append(("\n".join(buf), False))
                buf, buf_tok = [], 0
            buf.append(para)
            buf_tok += t
            if buf_tok >= CHILD_TARGET:
                out.append(("\n".join(buf), False))
                buf, buf_tok = [], 0
    if buf:
        out.append(("\n".join(buf), False))
    return [(p, t) for p, t in out if p.strip()]


def parse_corpus(corpus_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []
    with (out_dir / "chunks.jsonl").open("w") as cf, (out_dir / "edges.jsonl").open("w") as ef:
        for md in sorted(corpus_root.rglob("raw.md")):
            release = next(p for p in md.parts if p.startswith("Rel-"))
            series = int(next(p for p in md.parts if p.endswith("_series")).split("_")[0])
            try:
                chunks, edges, stats = parse_spec(md, release, series)
            except ValueError as e:
                print(f"SKIP {md}: {e}")
                continue
            for c in chunks:
                cf.write(c.model_dump_json() + "\n")
            for e in edges:
                ef.write(e.model_dump_json() + "\n")
            all_stats.append(stats)
            print(f"{stats['spec']} {release}: {stats['chunks']} chunks, "
                  f"{stats['edges']} edges, {stats['over_embed_limit']} over-limit")
    (out_dir / "manifest.json").write_text(json.dumps(all_stats, indent=1))


if __name__ == "__main__":
    import typer
    typer.run(lambda corpus: parse_corpus(Path(corpus), Path("data/parsed")))
