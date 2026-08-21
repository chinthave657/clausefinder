"""Citation validator tests — pure functions, no index, no API keys."""
from agent.validators.citations import (
    _quote_score,
    strip_failed_quotes,
    validate,
)

CHUNK_A = {
    "text": (
        "The UE shall **submit** the *RRCReconfigurationComplete* message to "
        "lower layers for transmission upon which the procedure ends.\n"
        "1> if the RRCReconfiguration includes the fullConfig:\n"
        "2> perform the full configuration procedure as specified in 5.3.5.11;"
    )
}
CHUNK_B = {
    "text": (
        "The network configures the UE with a measurement gap pattern applicable "
        "to all frequencies as specified in clause 9.1.2 of TS 38.133."
    )
}
RETRIEVED = {"[C1]": CHUNK_A, "[C2]": CHUNK_B}


# --- _quote_score: normalization + elision segments -----------------------

def test_quote_score_ignores_markdown_and_bullets():
    q = "the UE shall submit the RRCReconfigurationComplete message to lower layers"
    assert _quote_score(q, CHUNK_A["text"]) >= 95


def test_quote_score_elided_segments_each_verbatim():
    q = ("if the RRCReconfiguration includes the fullConfig ... "
         "perform the full configuration procedure as specified in 5.3.5.11")
    assert _quote_score(q, CHUNK_A["text"]) >= 95


def test_quote_score_elision_min_semantics():
    # first segment verbatim, second invented -> min() drags score down
    q = ("if the RRCReconfiguration includes the fullConfig [...] "
         "the UE shall immediately detach from the network and power off")
    assert _quote_score(q, CHUNK_A["text"]) < 95


def test_quote_score_invented_text_fails():
    assert _quote_score("the UE shall ignore all reconfiguration messages entirely",
                        CHUNK_A["text"]) < 95


# --- validate: tags, anchoring, re-anchoring ------------------------------

def test_validate_passes_on_good_answer():
    ans = ('The UE completes the procedure: "the UE shall submit the '
           'RRCReconfigurationComplete message to lower layers for transmission" [C1].')
    rep = validate(ans, RETRIEVED)
    assert rep.passed
    assert all(c.ok for c in rep.checks)


def test_validate_unknown_tag_fails():
    rep = validate("Per the spec [C7], anything goes.", RETRIEVED)
    assert not rep.passed
    assert any(c.tag == "[C7]" and not c.ok for c in rep.checks)


def test_validate_reanchors_quote_to_other_chunk():
    # quote is verbatim in C2 but cited as C1 -> grounding holds, tag re-anchored
    ans = ('"the network configures the UE with a measurement gap pattern '
           'applicable to all frequencies" [C1].')
    rep = validate(ans, RETRIEVED)
    assert rep.passed
    assert any(c.ok and "re-anchored [C1]->[C2]" in c.reason for c in rep.checks)


def test_validate_fabricated_quote_fails():
    ans = ('"the UE shall transmit on all beams simultaneously without any '
           'network configuration whatsoever" [C1].')
    rep = validate(ans, RETRIEVED)
    assert not rep.passed


def test_validate_uncited_tags_reported():
    rep = validate('See "the UE shall submit the RRCReconfigurationComplete '
                   'message to lower layers for transmission" [C1].', RETRIEVED)
    assert rep.uncited_tags == ["[C2]"]


def test_validate_no_citations_fails():
    assert not validate("An answer with zero citations.", RETRIEVED).passed


# --- strip_failed_quotes ---------------------------------------------------

def test_strip_failed_quotes_removes_only_fabrications():
    good = ("the UE shall submit the RRCReconfigurationComplete message to "
            "lower layers for transmission")
    bad = "the UE shall levitate above the gNB while measuring RSRP values"
    ans = f'Good: "{good}" [C1]. Bad: "{bad}" [C1].'
    out = strip_failed_quotes(ans, RETRIEVED)
    assert good in out
    assert bad not in out
    assert "(quote omitted — see " in out   # points at the closest source
    assert "via [C1])" in out
    assert out.count("[C1]") >= 2  # clause references survive (marker adds one)
