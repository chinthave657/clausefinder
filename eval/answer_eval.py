"""Ask-mode answer eval: runs agent.answer.ask() over the golden set and
scores validator pass rate, gold-clause-citation rate, and judge distribution.

Reuses the run pattern from eval/retrieval_eval.py and eval/diff_eval.py
(module-level `run()`, plain prints, JSON summary at the end) and the golden
file (eval/golden/starter.jsonl). Only rows with mode == "ask" are run here —
diff/abstain rows belong to eval/diff_eval.py and the abstain path.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from agent.answer import _client, ask
from agent.tools.retrieval import Retriever, load_acronyms
from eval.judge import JudgeVerdict, judge

TAG_RE = re.compile(r"\[C(\d+)\]")

GOLDEN = Path("eval/golden/starter.jsonl")
DB = Path("data/index")
PARSED = Path("data/parsed")
REPORTS_DIR = Path("eval/reports")


def _cited_chunks(answer: str, tagmap: dict[str, dict]) -> list[dict]:
    """Chunks the answer actually cited (tag appears in the text) — not the
    full retrieved set. An uncited retrieval doesn't ground a claim."""
    cited_tags = {f"[C{n}]" for n in TAG_RE.findall(answer)}
    return [c for t, c in tagmap.items() if t in cited_tags]


def run(golden: Path = GOLDEN, db: Path = DB, parsed: Path = PARSED,
        limit: int | None = None) -> dict:
    load_dotenv()
    retriever = Retriever(db)
    acronyms = load_acronyms(parsed)
    client, model = _client()

    rows = [json.loads(l) for l in golden.open() if l.strip()]
    rows = [r for r in rows if r.get("mode") == "ask"]
    if limit:
        rows = rows[:limit]

    results: list[dict] = []
    validator_pass = 0
    gold_cited = 0
    scores: Counter[int] = Counter()

    for r in rows:
        answer, report, tagmap = ask(r["question"], retriever, release=r.get("release"),
                                     acronyms=acronyms)
        cited = _cited_chunks(answer, tagmap)
        verdict: JudgeVerdict = judge(
            client, r["question"], answer,
            gold_answer_points=r.get("gold_answer_points"),
            gold_clauses=r.get("gold_clauses", []),
            cited_chunks=cited)

        validator_pass += report.passed
        gold_cited += verdict.required_clauses_cited
        scores[verdict.score] += 1

        status = "✓" if report.passed else "✗"
        cite_status = "✓" if verdict.required_clauses_cited else "✗"
        print(f"{status} validator | {cite_status} cited | judge={verdict.label:7s} "
              f"| {r['id']} {r['question'][:55]}")

        results.append({
            "id": r["id"], "question": r["question"],
            "validator_passed": report.passed,
            "required_clauses_cited": verdict.required_clauses_cited,
            "judge_score": verdict.score, "judge_label": verdict.label,
            "citation_only": verdict.citation_only,
            "judge_parse_failed": verdict.parse_failed,
            "judge_rationale": verdict.rationale,
        })

    n = len(rows)
    out = {
        "timestamp": datetime.now(UTC).isoformat(),
        "n": n,
        "validator_pass_rate": validator_pass / n if n else 0.0,
        "gold_cited_rate": gold_cited / n if n else 0.0,
        "judge_score_distribution": {
            "wrong": scores[0], "partial": scores[1], "correct": scores[2]},
        "judge_mean_score": sum(s * c for s, c in scores.items()) / n if n else 0.0,
        "results": results,
    }

    print("\n" + "=" * 60)
    print(f"{'metric':<24} {'value':>10}")
    print(f"{'n':<24} {n:>10}")
    print(f"{'validator pass rate':<24} {out['validator_pass_rate']:>10.1%}")
    print(f"{'gold-clause cited rate':<24} {out['gold_cited_rate']:>10.1%}")
    print(f"{'judge mean score':<24} {out['judge_mean_score']:>10.2f}")
    print(f"{'judge: wrong/partial/correct':<24} "
          f"{scores[0]:>3}/{scores[1]:>3}/{scores[2]:>3}")
    print("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"answer_eval_{stamp}.json"
    report_path.write_text(json.dumps(out, indent=1))
    print(f"\nreport written: {report_path}")

    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N ask-mode golden rows (cheap CI runs)")
    p.add_argument("--golden", type=Path, default=GOLDEN)
    p.add_argument("--db", type=Path, default=DB)
    p.add_argument("--parsed", type=Path, default=PARSED)
    args = p.parse_args()
    run(golden=args.golden, db=args.db, parsed=args.parsed, limit=args.limit)
