"""Rule-first mode router (design §3, "ROUTER").

Two tiers, cheapest first:
  1. regex_route() -- deterministic, free, runs on every unified-chat query.
     Fires only when the text is unambiguous: diff keywords + >=2 distinct
     release tokens => Mode.diff; a single explicit clause reference with no
     diff signal => Mode.explain. Anything else is left ambiguous (None).
  2. LLM fallback (Nemotron Nano, reasoning off, max_tokens=10) -- called
     ONLY when the regex tier returns None. Enum-tolerant parse (substring
     match against the three labels); default=ask on ANY failure (missing
     key, network error, empty/garbage completion) -- the router must never
     raise and must never block a query (design: "default=ask on any
     failure"). Kept in NAT YAML + eval per design ("LLM router kept in NAT
     YAML + eval for the story").

UI tabs bypass this entirely (design: "UI tab is the mode") -- this module
is only for the single unified-chat surface.
"""
from __future__ import annotations

import os
import re
from enum import Enum


class Mode(str, Enum):
    ask = "ask"
    diff = "diff"
    explain = "explain"


# "Rel-17", "Release 18", "rel 19" -- NOT a bare "17" (too weak a signal on
# its own outside a Rel-/Release- context; see design's "two release tokens").
RELEASE_RE = re.compile(r"\bRel(?:ease)?[-\s]?(\d{1,2})\b", re.IGNORECASE)

DIFF_KEYWORDS_RE = re.compile(
    r"\b(diff|differences?|compare|comparison|changed|changes|delta)\b",
    re.IGNORECASE,
)

# Single clause reference, same shape as agent.explain.REF_RE:
# "TS 38.331 5.3.5.3" | "38.331 §5.3.5.3" | "38.331 clause 5.3.5.3"
CLAUSE_REF_RE = re.compile(
    r"(?:TS\s*)?\d{2}\.\d{3}\s*(?:§|clause\s*)?[\dA-Za-z]+(?:\.[\dA-Za-z]+)*",
    re.IGNORECASE,
)


def regex_route(text: str) -> Mode | None:
    """Deterministic tier. Returns a Mode when the text unambiguously signals
    one, else None (caller should fall back to the LLM tier)."""
    releases = {m for m in RELEASE_RE.findall(text)}
    has_diff_kw = bool(DIFF_KEYWORDS_RE.search(text))

    if has_diff_kw and len(releases) >= 2:
        return Mode.diff
    if not has_diff_kw and len(releases) < 2 and CLAUSE_REF_RE.search(text):
        return Mode.explain
    return None


def _parse_mode(raw: str) -> Mode:
    """Enum-tolerant parse: first label whose value appears in the (lowered,
    stripped) completion wins. Empty/unrecognized text -> Mode.ask."""
    raw = (raw or "").strip().lower()
    for m in Mode:
        if m.value in raw:
            return m
    return Mode.ask


_ROUTER_SYSTEM = (
    "Classify the user's telecom-standards query into exactly one label: "
    "ask, diff, or explain.\n"
    "diff = comparing behaviour/text across two or more 3GPP releases.\n"
    "explain = walking through what ONE named spec clause says.\n"
    "ask = anything else (a normal standards question).\n"
    "Answer with ONLY that one word, nothing else."
)


def _llm_route(text: str) -> Mode:
    """LLM fallback tier. Never raises: any failure returns Mode.ask."""
    try:
        api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return Mode.ask
        from openai import OpenAI

        if os.environ.get("NVIDIA_API_KEY"):
            base_url, model = ("https://integrate.api.nvidia.com/v1",
                               os.environ.get("CLAUSEFINDER_ROUTER_MODEL",
                                               "nvidia/nemotron-3-nano-30b-a3b"))
        else:
            base_url, model = ("https://openrouter.ai/api/v1",
                               os.environ.get("CLAUSEFINDER_ROUTER_MODEL",
                                               "nvidia/nemotron-3-nano-30b-a3b:free"))

        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=10,
            extra_body={"chat_template_kwargs": {"thinking": False}},
        )
        raw = resp.choices[0].message.content or ""
    except Exception:
        return Mode.ask
    return _parse_mode(raw)


def route(text: str) -> Mode:
    """Rule-first mode routing. Regex tier first; LLM tier only on ambiguity."""
    ruled = regex_route(text)
    if ruled is not None:
        return ruled
    return _llm_route(text)


__all__ = ["Mode", "regex_route", "route", "RELEASE_RE", "DIFF_KEYWORDS_RE", "CLAUSE_REF_RE"]
