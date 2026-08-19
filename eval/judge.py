"""Pinned LLM judge for ask-mode answer correctness.

Design §5.1: "pinned judge" — the judge model, temperature, and rubric are
frozen constants. A judge that silently drifts (provider swaps the model
behind an alias, someone bumps the version to chase quality) makes the CI
gate rot silently: pass rates move because grading changed, not because the
system did. JUDGE_MODEL is pinned deliberately; changing it is a reviewed,
deliberate action, never a side effect of a provider update.

Two grading paths:
  - gold_answer_points present -> LLM grades content on the 0/1/2 rubric.
  - gold_answer_points absent  -> "citation-only" mode (design §5.1: "some
    rows may lack gold_answer_points - judge those as citation-only"): no
    content to grade, so score is derived deterministically from whether
    required gold clauses were actually cited. No LLM call, no flake.

required_clauses_cited is ALWAYS computed deterministically (never asked of
the LLM) using the same parent-clause match rule pinned in retrieval_eval.py
(gold 5.2.1 matched by a 5.2.1.3 chunk) — citation grounding is a fact about
the retrieved/cited set, not a judgment call.
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel

# Pinned deliberately (see module docstring). Bump only as a reviewed change,
# never implicitly via a provider-side default/alias update.
JUDGE_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
JUDGE_TEMPERATURE = 0.0

_SCORE_LABEL = {0: "wrong", 1: "partial", 2: "correct"}

RUBRIC = """You are grading a 3GPP standards assistant's answer for factual correctness.

Question: {question}

Required points the answer must cover (gold, from a human-authored key):
{points}

Candidate answer:
{answer}

Score the candidate answer:
0 = wrong — misses or contradicts the required points
1 = partial — covers some but not all required points, or is vague/hedged where the gold is specific
2 = correct — covers all required points accurately

Respond with ONLY a JSON object, no other text:
{{"score": 0, "rationale": "<one sentence>"}}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeVerdict(BaseModel):
    score: int  # 0/1/2 wrong/partial/correct
    label: Literal["wrong", "partial", "correct"]
    required_clauses_cited: bool
    citation_only: bool = False  # True: no gold_answer_points, score is citation-derived only
    rationale: str
    parse_failed: bool = False  # True: judge output never parsed; score is a safe default


def _clause_match(cited_clause: str, gold_clause: str) -> bool:
    """Same rule as eval/retrieval_eval.py::clause_match — pinned together."""
    return cited_clause == gold_clause or cited_clause.startswith(gold_clause + ".")


def required_clauses_cited(gold_clauses: list[dict], cited_chunks: list[dict]) -> bool:
    """True if every REQUIRED gold clause (or a descendant) is among the
    chunks the answer actually cited (not merely retrieved). No required
    clauses -> vacuously true (nothing to check)."""
    required = [g for g in gold_clauses if g.get("required")]
    if not required:
        return True
    return all(
        any(c.get("spec") == g["spec"] and _clause_match(c.get("clause", ""), g["clause"])
            for c in cited_chunks)
        for g in required
    )


def _parse(raw: str) -> tuple[int, str] | None:
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        score = int(obj["score"])
        if score not in (0, 1, 2):
            return None
        return score, str(obj.get("rationale", ""))[:300]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _call(client, question: str, answer: str, points_text: str) -> str:
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": RUBRIC.format(
            question=question, points=points_text, answer=answer)}],
        temperature=JUDGE_TEMPERATURE,
        max_tokens=200,
        extra_body={"chat_template_kwargs": {"thinking": False}},
    )
    return (resp.choices[0].message.content or "").strip()


def judge(client, question: str, answer: str,
          gold_answer_points: list[str] | None,
          gold_clauses: list[dict],
          cited_chunks: list[dict]) -> JudgeVerdict:
    """client: OpenAI-compatible client pinned to JUDGE_MODEL's endpoint
    (agent.answer._client() gives one). cited_chunks: chunk dicts (with
    'spec'/'clause') for tags actually cited in `answer` — NOT the full
    retrieved set; an uncited retrieval doesn't ground a claim."""
    cited_ok = required_clauses_cited(gold_clauses, cited_chunks)

    if not gold_answer_points:
        score = 2 if cited_ok else 0
        return JudgeVerdict(
            score=score, label=_SCORE_LABEL[score], required_clauses_cited=cited_ok,
            citation_only=True,
            rationale="citation-only: no gold_answer_points; scored on required-clause citation alone.")

    points_text = "\n".join(f"- {p}" for p in gold_answer_points)

    # Retry once with a shortened prompt on parse failure (design: "retry/
    # shorten on parse failure") — long verbose answers most often confuse
    # the judge into wrapping JSON in prose; truncating removes that.
    for attempt_answer in (answer, answer[:600]):
        raw = _call(client, question, attempt_answer, points_text)
        parsed = _parse(raw)
        if parsed:
            score, rationale = parsed
            return JudgeVerdict(
                score=score, label=_SCORE_LABEL[score], required_clauses_cited=cited_ok,
                rationale=rationale)

    return JudgeVerdict(
        score=1, label="partial", required_clauses_cited=cited_ok,
        rationale="judge output did not parse after retry; defaulted to partial.",
        parse_failed=True)
