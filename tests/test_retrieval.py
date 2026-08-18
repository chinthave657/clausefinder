"""Retrieval fusion tests with a stubbed LanceDB table — no index, no model."""
from agent.tools.retrieval import (
    FTS_W,
    FUSE_DEPTH,
    RRF_K,
    VEC_W,
    Retriever,
    _fts_query,
    enhance_query,
)

# --- stubs -----------------------------------------------------------------

def _chunk(cid, parent=None, clause="1", spec="TS 38.331", release="Rel-18"):
    return {"id": cid, "parent_id": parent or f"p-{cid}", "spec": spec,
            "clause": clause, "release": release, "text": f"text {cid}"}


class _StubQuery:
    def __init__(self, hits):
        self._hits = hits

    def limit(self, n):
        self._hits = self._hits[:n]
        return self

    def where(self, *_a, **_k):
        return self

    def to_list(self):
        return list(self._hits)


class _StubTable:
    def __init__(self, vec_hits, fts_hits):
        self.vec_hits, self.fts_hits = vec_hits, fts_hits

    def search(self, query=None, query_type=None):
        if query_type == "fts":
            return _StubQuery(self.fts_hits)
        return _StubQuery(self.vec_hits)


class _StubModel:
    def encode(self, texts, **_k):
        return [[0.0, 1.0]] * len(texts)


def make_retriever(vec_hits, fts_hits, edges=()):
    r = object.__new__(Retriever)  # skip __init__: no lancedb connection
    r.tbl = _StubTable(vec_hits, fts_hits)
    r.edges = list(edges)
    r.model = _StubModel()
    return r


# --- weighted RRF fusion ---------------------------------------------------

def test_vec_top_hit_outranks_fts_top_hit():
    a, b = _chunk("a"), _chunk("b")
    out = make_retriever([a], [b]).search("q")
    assert [h.chunk["id"] for h in out] == ["a", "b"]
    assert out[0].tag == "[C1]" and out[1].tag == "[C2]"
    assert abs(out[0].score - VEC_W / (RRF_K + 1)) < 1e-9
    assert abs(out[1].score - FTS_W / (RRF_K + 1)) < 1e-9


def test_dual_presence_sums_both_legs():
    a, b = _chunk("a"), _chunk("b")
    out = make_retriever([a, b], [b, a]).search("q")
    # both appear in both legs at ranks 0/1 — scores are the leg sums
    want_a = VEC_W / (RRF_K + 1) + FTS_W / (RRF_K + 2)
    got = {h.chunk["id"]: h.score for h in out}
    assert abs(got["a"] - want_a) < 1e-9


def test_single_leg_excellence_beats_mediocre_dual_presence():
    # "solo" is vec rank 0 only; "dual" sits at rank 30 in BOTH legs
    solo, dual = _chunk("solo"), _chunk("dual")
    filler_v = [_chunk(f"v{i}") for i in range(30)]
    filler_f = [_chunk(f"f{i}") for i in range(30)]
    r = make_retriever([solo] + filler_v + [dual], filler_f + [dual])
    out = r.search("q", top=40)
    ids = [h.chunk["id"] for h in out]
    assert ids.index("solo") < ids.index("dual")


def test_fuse_depth_caps_contribution():
    deep = _chunk("deep")
    vec = [_chunk(f"v{i}") for i in range(FUSE_DEPTH)] + [deep]
    out = make_retriever(vec, []).search("q", top=100)
    assert "deep" not in {h.chunk["id"] for h in out}


def test_parent_dedup_keeps_best_child():
    c1, c2 = _chunk("k1", parent="P"), _chunk("k2", parent="P")
    other = _chunk("o")
    out = make_retriever([c1, c2, other], []).search("q")
    ids = [h.chunk["id"] for h in out]
    assert ids == ["k1", "o"]  # k2 collapsed into its parent's best child


def test_top_limits_results():
    vec = [_chunk(f"v{i}") for i in range(20)]
    assert len(make_retriever(vec, []).search("q", top=5)) == 5


# --- FTS sanitization ------------------------------------------------------

def test_fts_query_strips_tantivy_operators():
    q = 'what is "msg3" re-transmission? (see 5.1.3) - or *not*!'
    s = _fts_query(q)
    for ch in '?"()-*!':
        assert ch not in s
    assert "msg3" in s and "5.1.3" in s
    assert "  " not in s


def test_fts_query_plain_text_unchanged():
    assert _fts_query("random access preamble") == "random access preamble"


# --- acronym query enhancement --------------------------------------------

def test_enhance_query_appends_never_substitutes():
    out = enhance_query("what is BWP adaptation", {"BWP": "Bandwidth Part"})
    assert out.startswith("what is BWP adaptation")
    assert "BWP = Bandwidth Part" in out


def test_enhance_query_no_hit_is_identity():
    q = "random access procedure"
    assert enhance_query(q, {"BWP": "Bandwidth Part"}) == q
