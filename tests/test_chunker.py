"""Chunker tests over tests/fixtures/mini_spec.md — no index, no API keys."""
from pathlib import Path

import pytest

from ingest.chunker import parse_spec

FIXTURE = Path(__file__).parent / "fixtures" / "mini_spec.md"


@pytest.fixture(scope="module")
def parsed():
    return parse_spec(FIXTURE, release="Rel-18", series=38)


def test_spec_metadata(parsed):
    chunks, _, stats = parsed
    assert stats["spec"] == "TS 38.331"
    assert stats["version"] == "18.0.0"
    assert all(c.spec == "TS 38.331" and c.release == "Rel-18" for c in chunks)


def test_contents_skipped_and_empty_containers_dropped(parsed):
    chunks, _, _ = parsed
    clauses = {c.clause for c in chunks}
    assert "1" in clauses
    assert "5.3.5.3" in clauses
    # Contents heading is skipped; bodiless container clauses emit no chunks
    titles = {c.clause_title.lower() for c in chunks}
    assert "contents" not in titles
    assert "5" not in clauses and "5.3" not in clauses


def test_breadcrumb_uses_clause_hierarchy(parsed):
    chunks, _, _ = parsed
    c = next(c for c in chunks if c.clause == "5.3.5.3")
    # ancestors come from clause-number prefixes, not markdown depth
    assert c.breadcrumb.startswith("TS 38.331 Rel-18 — ")
    assert "5 Procedures" in c.breadcrumb
    assert "5.3 RRC connection control" in c.breadcrumb
    assert "5.3.5 RRC reconfiguration" in c.breadcrumb
    assert c.breadcrumb.endswith("5.3.5.3 Reception of an RRCReconfiguration by the UE")


def test_children_share_parent_id_and_text_prefix(parsed):
    chunks, _, _ = parsed
    kids = [c for c in chunks if c.clause == "5.3.5.3"]
    assert len(kids) >= 2  # prose + atomic table at minimum
    assert len({c.parent_id for c in kids}) == 1
    assert [c.id for c in kids] == [f"{kids[0].parent_id}-{i}" for i in range(len(kids))]


def test_table_stays_atomic(parsed):
    chunks, _, _ = parsed
    tables = [c for c in chunks if c.is_table]
    assert len(tables) == 1
    t = tables[0]
    assert t.clause == "5.3.5.3"
    assert "fullConfig" in t.text and "masterCellGroup" in t.text
    # no prose leaked into the table chunk
    assert "UE shall submit" not in t.text


def test_xref_edges(parsed):
    _, edges, _ = parsed
    pairs = {(e.dst_spec, e.dst_clause) for e in edges}
    # "TS 38.321 clause 5.4.1" -> cross-spec edge
    assert ("TS 38.321", "5.4.1") in pairs
    # "clause 5.3.5.5" with no nearby TS -> same-spec edge
    assert ("TS 38.331", "5.3.5.5") in pairs
    assert all(e.src_spec == "TS 38.331" for e in edges)


def test_annex_flags(parsed):
    chunks, _, _ = parsed
    a = next(c for c in chunks if c.clause == "A.1")
    assert a.annex is True
    assert a.normative is False  # "(informative)" annex
    n = next(c for c in chunks if c.clause == "1")
    assert n.annex is False and n.normative is True
