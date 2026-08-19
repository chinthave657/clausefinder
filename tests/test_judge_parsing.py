"""Judge parsing tests — mocked LLM responses, no network, no API keys.

Covers: JSON parse happy path, retry-on-parse-failure, parse-failure
fallback, citation-only mode (no gold_answer_points -> no LLM call at all),
and the deterministic required_clauses_cited match rule.
"""
from eval.judge import JudgeVerdict, judge, required_clauses_cited

GOLD_CLAUSES = [
    {"spec": "TS 38.331", "clause": "5.3.5.3", "required": True},
    {"spec": "TS 38.331", "clause": "5.3.7.1", "required": False},
]


class _StubMessage:
    def __init__(self, content):
        self.content = content


class _StubChoice:
    def __init__(self, content):
        self.message = _StubMessage(content)


class _StubResp:
    def __init__(self, content):
        self.choices = [_StubChoice(content)]


class _StubClient:
    """Returns each entry in `responses` in order, one per .create() call."""
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResp(self._responses.pop(0))


# --- required_clauses_cited -------------------------------------------------

def test_required_clauses_cited_exact_match():
    cited = [{"spec": "TS 38.331", "clause": "5.3.5.3"}]
    assert required_clauses_cited(GOLD_CLAUSES, cited) is True


def test_required_clauses_cited_descendant_match():
    cited = [{"spec": "TS 38.331", "clause": "5.3.5.3.2"}]
    assert required_clauses_cited(GOLD_CLAUSES, cited) is True


def test_required_clauses_cited_missing():
    cited = [{"spec": "TS 38.331", "clause": "5.3.7.1"}]  # only the non-required one
    assert required_clauses_cited(GOLD_CLAUSES, cited) is False


def test_required_clauses_cited_no_required_is_vacuously_true():
    only_optional = [{"spec": "TS 38.331", "clause": "5.3.7.1", "required": False}]
    assert required_clauses_cited(only_optional, []) is True


def test_required_clauses_cited_wrong_spec_does_not_match():
    cited = [{"spec": "TS 38.101-1", "clause": "5.3.5.3"}]
    assert required_clauses_cited(GOLD_CLAUSES, cited) is False


# --- judge(): citation-only mode (no LLM call) ------------------------------

def test_judge_citation_only_no_llm_call_when_cited():
    client = _StubClient([])  # would raise IndexError if judge() called .create()
    v = judge(client, "q", "answer text [C1]",
              gold_answer_points=None, gold_clauses=GOLD_CLAUSES,
              cited_chunks=[{"spec": "TS 38.331", "clause": "5.3.5.3"}])
    assert isinstance(v, JudgeVerdict)
    assert v.citation_only is True
    assert v.required_clauses_cited is True
    assert v.score == 2 and v.label == "correct"
    assert client.calls == []


def test_judge_citation_only_scores_wrong_when_not_cited():
    client = _StubClient([])
    v = judge(client, "q", "answer text", gold_answer_points=[],
              gold_clauses=GOLD_CLAUSES, cited_chunks=[])
    assert v.citation_only is True
    assert v.required_clauses_cited is False
    assert v.score == 0 and v.label == "wrong"


# --- judge(): LLM-graded path with gold_answer_points -----------------------

def test_judge_parses_clean_json_on_first_try():
    client = _StubClient(['{"score": 2, "rationale": "covers all points"}'])
    v = judge(client, "q", "a", gold_answer_points=["point A", "point B"],
              gold_clauses=[], cited_chunks=[])
    assert v.score == 2 and v.label == "correct"
    assert v.parse_failed is False
    assert v.citation_only is False
    assert len(client.calls) == 1


def test_judge_parses_json_wrapped_in_prose():
    client = _StubClient(['Sure thing — here you go:\n{"score": 1, "rationale": "partial"} \nhope that helps'])
    v = judge(client, "q", "a", gold_answer_points=["point A"], gold_clauses=[], cited_chunks=[])
    assert v.score == 1 and v.label == "partial"


def test_judge_retries_once_on_parse_failure_then_succeeds():
    client = _StubClient(["not json at all", '{"score": 0, "rationale": "wrong"}'])
    v = judge(client, "q", "a" * 5000, gold_answer_points=["point A"], gold_clauses=[], cited_chunks=[])
    assert v.score == 0 and v.label == "wrong"
    assert v.parse_failed is False
    assert len(client.calls) == 2
    # second attempt used the shortened answer
    assert len(client.calls[1]["messages"][0]["content"]) < len(client.calls[0]["messages"][0]["content"])


def test_judge_falls_back_to_partial_after_both_attempts_fail():
    client = _StubClient(["garbage", "still garbage"])
    v = judge(client, "q", "a", gold_answer_points=["point A"], gold_clauses=GOLD_CLAUSES,
              cited_chunks=[{"spec": "TS 38.331", "clause": "5.3.5.3"}])
    assert v.parse_failed is True
    assert v.score == 1 and v.label == "partial"
    assert len(client.calls) == 2
    # required_clauses_cited is still computed deterministically, judge or not
    assert v.required_clauses_cited is True


def test_judge_rejects_out_of_range_score_and_retries():
    client = _StubClient(['{"score": 5, "rationale": "bad"}', '{"score": 2, "rationale": "ok"}'])
    v = judge(client, "q", "a", gold_answer_points=["point A"], gold_clauses=[], cited_chunks=[])
    assert v.score == 2
    assert len(client.calls) == 2


def test_judge_uses_pinned_model_and_temperature():
    client = _StubClient(['{"score": 2, "rationale": "ok"}'])
    judge(client, "q", "a", gold_answer_points=["point A"], gold_clauses=[], cited_chunks=[])
    from eval.judge import JUDGE_MODEL, JUDGE_TEMPERATURE
    assert client.calls[0]["model"] == JUDGE_MODEL
    assert client.calls[0]["temperature"] == JUDGE_TEMPERATURE
