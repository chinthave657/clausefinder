"""Table-driven tests for the regex router tier ONLY -- no LLM calls.

route() falls back to _llm_route() when regex_route() returns None; those
cases are exercised here with _llm_route patched to raise, which proves the
regex tier alone is deciding (a real fallback call would fail the test).
"""
from __future__ import annotations

import pytest

from agent.router import Mode, regex_route, route

# (input text, expected Mode | None) -- None means "ambiguous, LLM tier decides"
REGEX_CASES = [
    # -- unambiguous diff: keyword + >=2 release tokens --
    ("diff Rel-17 vs Rel-18 for RRC reconfiguration", Mode.diff),
    ("compare Release 17 and Release 18 handling of SIB1", Mode.diff),
    ("what changed between rel-16 and rel-17 in NAS registration", Mode.diff),
    ("show me the differences between Rel-17 and Rel-18 for TS 38.331 5.3.5.3", Mode.diff),
    ("Rel-15 vs Rel-16 vs Rel-17: delta in AMF selection", Mode.diff),

    # -- unambiguous explain: single clause ref, no diff signal --
    ("Explain TS 38.331 5.3.5.3", Mode.explain),
    ("What does 38.331 clause 5.3.5.3 say?", Mode.explain),
    ("walk me through 38.331 §5.3.5.3", Mode.explain),

    # -- ambiguous: one release token only --
    ("How does 5G handle handover in Rel-17?", None),
    ("What's new in Rel-18?", None),

    # -- ambiguous: two releases but no diff keyword --
    ("Rel-17 and Rel-18 both support SIB1 broadcast", None),

    # -- ambiguous: diff keyword but no release tokens --
    ("compare RRC reconfiguration procedures", None),
    ("what changed in the AMF selection procedure", None),

    # -- ambiguous: plain question, no signal at all --
    ("What is a gNB?", None),
    ("", None),

    # -- diff keyword wins over an incidental clause-shaped number --
    ("diff Rel-17 vs Rel-18 for 38.331", Mode.diff),
]


@pytest.mark.parametrize("text,expected", REGEX_CASES)
def test_regex_route(text, expected):
    assert regex_route(text) == expected


@pytest.mark.parametrize("text,expected", [c for c in REGEX_CASES if c[1] is not None])
def test_route_short_circuits_on_unambiguous_input(text, expected, monkeypatch):
    """route() must not touch the LLM tier when the regex tier is decisive."""
    def _boom(_text):
        raise AssertionError("LLM fallback must not be called for unambiguous input")

    monkeypatch.setattr("agent.router._llm_route", _boom)
    assert route(text) == expected


def test_route_falls_back_when_ambiguous(monkeypatch):
    monkeypatch.setattr("agent.router._llm_route", lambda _text: Mode.diff)
    assert route("What is a gNB?") == Mode.diff


def test_llm_route_defaults_ask_with_no_api_key(monkeypatch):
    """_llm_route() itself, exercised with no key set -- no network call is
    made (short-circuits before the client is constructed), so this stays
    within "no LLM in tests" while proving the default=ask-on-failure path."""
    from agent.router import _llm_route

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert _llm_route("anything") == Mode.ask


@pytest.mark.parametrize("raw,expected", [
    ("ask", Mode.ask),
    ("Diff", Mode.diff),
    ("EXPLAIN", Mode.explain),
    ("  diff  \n", Mode.diff),
    ("the answer is: explain", Mode.explain),
    ("", Mode.ask),
    ("banana", Mode.ask),
])
def test_llm_route_never_called_but_parse_is_enum_tolerant(raw, expected):
    from agent.router import _parse_mode
    assert _parse_mode(raw) == expected


def test_term_alias_expansion():
    from agent.tools.retrieval import enhance_query
    out = enhance_query("explain abot AI-RAN?", {})
    assert "AI/ML for NG-RAN" in out
    assert out.startswith("explain abot AI-RAN?")  # append, never substitute


def test_term_alias_case_and_punct():
    from agent.tools.retrieval import enhance_query
    assert "AI/ML for NG-RAN" in enhance_query("What is ai-ran, exactly?", {})
    assert "AI/ML" not in enhance_query("What is NWDAF?", {})
