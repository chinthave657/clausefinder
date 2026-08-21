"""Deterministic retrieval funnel.

BM25 top-100 ⊕ vector top-100 → RRF → parent expansion (full clause unit,
deduped) → 1-hop xref expansion (≤4 extra chunks). Reranker is a config flag
(P1). Soft release filter via LanceDB prefilter — never a hard series gate.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import lancedb

CAND_DEPTH = 100
FUSE_DEPTH = 50      # ranks beyond this add dual-presence noise, not signal
RRF_K = 60
VEC_W = 0.7          # OTel-568M (NDCG@10 90.1) outranks BM25 on NL questions;
FTS_W = 0.3          # weighted RRF prevents mediocre dual-presence from
                     # beating single-leg excellence (no reranker until P1)
ID_W = 0.6           # identifier leg: exact spec tokens (T304, SIB1, 5QI) are
                     # the strongest relevance signal an engineer query carries;
                     # without this leg, generic-term BM25 mass ("timer",
                     # "expires") buries the one chunk naming the identifier

# spec identifiers: T304, SIB1, N2, 5QI, F1AP, Msg3, KAMF … but not acronyms
# that are pure words (UE, AMF) — those the vector leg handles
_IDENT_RE = re.compile(r"\b(?=\w*\d)(?:[A-Za-z]{1,6}\d{1,4}[A-Za-z]{0,4}|5Q[Ii]|Msg[1-5])\b")
_IDENT_STOP = {"3GPP", "5G", "4G", "2G", "3G", "5GS", "5GC", "5GMM", "5GSM",
               "Rel17", "Rel18", "Rel19", "TS23", "TS24", "TS29", "TS33", "TS38"}


def _identifiers(query: str) -> list[str]:
    out = []
    for m in _IDENT_RE.findall(query):
        if m not in _IDENT_STOP and not re.fullmatch(r"\d+", m):
            out.append(m)
    return list(dict.fromkeys(out))
TOP_FINAL = 8
XREF_EXTRA = 4

# OTel-Reranker-0.6B (Qwen3 cross-encoder, MRR@10 0.944 on telecom eval).
# Design §3 step 3: rerank fused top-50 before parent expansion. Off by
# default on CPU-only paths (adds ~5-30s); on for eval/GPU via env or flag.
RERANK_MODEL = "farbodtavakkoli/OTel-Reranker-0.6B"
RERANK_DEPTH = 50

_FTS_SANITIZE = str.maketrans({c: " " for c in "?!:;()[]{}\"'-/\\*^~"})


def _fts_query(query: str) -> str:
    """Strip tantivy operators — '-' negates, '?' and quotes are syntax."""
    return " ".join(query.translate(_FTS_SANITIZE).split())


@dataclass
class Retrieved:
    tag: str            # [C1] …
    chunk: dict
    score: float
    rerank_score: float | None = None  # cross-encoder relevance; None if rerank off


class Retriever:
    def __init__(self, db_path: Path, model=None, rerank: bool | None = None):
        self.db = lancedb.connect(db_path)
        self.tbl = self.db.open_table("chunks")
        self.edges = (
            self.db.open_table("edges").to_arrow().to_pylist()
            if "edges" in self.db.table_names() else []
        )
        if model is None:
            from ingest.embed_index import load_model
            model = load_model()
        self.model = model
        self.rerank = (rerank if rerank is not None
                       else os.environ.get("CLAUSEFINDER_RERANK", "0") == "1")
        self._reranker = None

    def to_device(self, device: str) -> None:
        """Move embedder (+reranker if loaded) — ZeroGPU pattern: weights load
        on CPU at boot; each GPU window hops them to cuda (seconds, not the
        30-60s cold load that blew the declared duration)."""
        if getattr(self, "_device", None) == device:
            return
        self.model.to(device)
        if self._reranker is not None:
            self._reranker.model.to(device)
        self._device = device

    def _rerank_hits(self, query: str, hits: list[dict]) -> list[dict]:
        if self._reranker is None:
            import torch
            from sentence_transformers import CrossEncoder
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
            self._reranker = CrossEncoder(RERANK_MODEL, trust_remote_code=True,
                                          max_length=1024, device=device)
            # Qwen3 reranker ships no pair-rendering template; ST requires one
            self._reranker.processor.chat_template = (
                '<Query>: {{ messages | selectattr("role", "eq", "query")'
                ' | map(attribute="content") | first }}\n'
                '<Document>: {{ messages | selectattr("role", "eq", "document")'
                ' | map(attribute="content") | first }}')
        scores = self._reranker.predict(
            [(query, h["breadcrumb"] + "\n" + h["text"]) for h in hits],
            batch_size=8, show_progress_bar=False)
        ranked = sorted(zip(scores, hits), key=lambda p: -float(p[0]))
        for s, h in ranked:
            h["_rerank_score"] = float(s)
        return [h for _, h in ranked]

    def search(self, query: str, release: str | None = None,
               top: int = TOP_FINAL) -> list[Retrieved]:
        where = f"release = '{release}'" if release else None

        qvec = self.model.encode([query], normalize_embeddings=True)[0]
        vq = self.tbl.search(qvec).limit(CAND_DEPTH)
        fq = self.tbl.search(_fts_query(query), query_type="fts").limit(CAND_DEPTH)
        if where:
            vq = vq.where(where, prefilter=True)
            fq = fq.where(where)
        vec_hits = vq.to_list()
        fts_hits = fq.to_list()

        # identifier leg: exact-token FTS on spec identifiers in the query
        id_hits: list[dict] = []
        idents = _identifiers(query)
        if idents:
            iq = self.tbl.search(" ".join(idents), query_type="fts").limit(CAND_DEPTH)
            if where:
                iq = iq.where(where)
            id_hits = iq.to_list()

        # Weighted reciprocal-rank fusion over capped depth
        scores: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for hits, w in ((vec_hits, VEC_W), (fts_hits, FTS_W), (id_hits, ID_W)):
            for rank, h in enumerate(hits[:FUSE_DEPTH]):
                cid = h["id"]
                scores[cid] = scores.get(cid, 0.0) + w / (RRF_K + rank + 1)
                by_id.setdefault(cid, h)

        ranked = sorted(scores, key=scores.get, reverse=True)

        # Optional cross-encoder rerank of fused top-RERANK_DEPTH (design §3.3)
        if getattr(self, "rerank", False):  # tests build Retriever via __new__
            cands = [by_id[c] for c in ranked[:RERANK_DEPTH]]
            reranked = self._rerank_hits(query, cands)
            ranked = [h["id"] for h in reranked] + ranked[RERANK_DEPTH:]

        # Parent expansion: one result per clause unit (parent), best child wins
        seen_parents: set[str] = set()
        results: list[dict] = []
        for cid in ranked:
            h = by_id[cid]
            if h["parent_id"] in seen_parents:
                continue
            seen_parents.add(h["parent_id"])
            results.append(h)
            if len(results) >= top:
                break

        # 1-hop xref expansion from top-3
        extra: list[dict] = []
        for h in results[:3]:
            for e in self.edges:
                if e["src_spec"] == h["spec"] and e["src_clause"] == h["clause"]:
                    hits = self._by_clause(e["dst_spec"], e["dst_clause"], h["release"])
                    for x in hits:
                        if x["parent_id"] not in seen_parents:
                            seen_parents.add(x["parent_id"])
                            extra.append(x)
                if len(extra) >= XREF_EXTRA:
                    break
            if len(extra) >= XREF_EXTRA:
                break

        out = []
        for i, h in enumerate(results + extra):
            out.append(Retrieved(tag=f"[C{i+1}]", chunk=h, score=scores.get(h["id"], 0.0),
                                 rerank_score=h.get("_rerank_score")))
        return out

    def parent_text(self, r: Retrieved, cap_tokens: int = 1200) -> str:
        """Full clause unit for generation: all sibling children, capped."""
        sibs = (self.tbl.search()
                .where(f"parent_id = '{r.chunk['parent_id']}'")
                .limit(50).to_list())
        sibs.sort(key=lambda s: s["id"])
        text, tok = [], 0
        for s in sibs:
            t = int(len(s["text"].split()) * 1.3)
            if tok + t > cap_tokens:
                break
            text.append(s["text"])
            tok += t
        return "\n".join(text)

    def _by_clause(self, spec: str, clause: str, release: str) -> list[dict]:
        return (self.tbl.search()
                .where(f"spec = '{spec}' AND clause = '{clause}' AND release = '{release}'")
                .limit(1).to_list())


def load_acronyms(parsed_dir: Path) -> dict[str, str]:
    p = parsed_dir / "acronyms.json"
    return json.loads(p.read_text()) if p.exists() else {}


# Curated industry-term → spec-terminology pointers (append-only, reviewed by
# hand). These are RELATED-terminology hints for retrieval, NOT equivalences —
# the answer is still grounded in whatever the retrieved clauses actually say.
# E.g. "AI-RAN" (industry: converged AI+RAN infrastructure) is a distinct
# concept from 3GPP's "AI/ML for NG-RAN" (Rel-18 data-collection support);
# the alias only helps retrieval surface the adjacent spec material.
TERM_ALIASES: dict[str, str] = {
    "AI-RAN": "related 3GPP terminology: AI/ML for NG-RAN, RAN intelligence",
    "AIRAN": "related 3GPP terminology: AI/ML for NG-RAN, RAN intelligence",
}


def enhance_query(query: str, acronyms: dict[str, str]) -> str:
    """Append (never substitute) expansions — Telco-RAG lexicon step.
    Acronym matching is exact/case-sensitive (3GPP acronyms are case-bearing);
    alias matching is punctuation-tolerant and case-insensitive."""
    raw_tokens = query.split()
    hits = [f"{a} = {x}" for a, x in acronyms.items() if a in raw_tokens]
    norm = {t.strip("?,.!;:()\"'").lower() for t in raw_tokens}
    hits += [f"{k} — {v}" for k, v in TERM_ALIASES.items() if k.lower() in norm]
    return query + ("\n(" + "; ".join(hits) + ")" if hits else "")
