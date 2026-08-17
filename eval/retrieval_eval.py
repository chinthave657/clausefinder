"""Retrieval-stage eval: recall@k and MRR@10 at clause granularity.

Labels come free from the golden set's gold_clauses. Parent-clause matching
rule (pinned): a retrieved chunk matches gold clause X if its clause == X or
its clause is a descendant of X (gold 5.2.1 matched by a 5.2.1.3 chunk).
"""
from __future__ import annotations

import json
from pathlib import Path

from agent.tools.retrieval import Retriever, enhance_query, load_acronyms


def clause_match(retrieved_clause: str, gold_clause: str) -> bool:
    return retrieved_clause == gold_clause or retrieved_clause.startswith(gold_clause + ".")


def run(golden: Path = Path("eval/golden/starter.jsonl"),
        db: Path = Path("data/index"), parsed: Path = Path("data/parsed")) -> dict:
    retriever = Retriever(db)
    acronyms = load_acronyms(parsed)
    rows = [json.loads(l) for l in golden.open() if l.strip()]
    rows = [r for r in rows if r["gold_clauses"]]

    r_at_5 = r_at_20 = mrr = 0.0
    for r in rows:
        hits = retriever.search(enhance_query(r["question"], acronyms),
                                release=r.get("release"), top=20)
        required = [g for g in r["gold_clauses"] if g.get("required")] or r["gold_clauses"]

        def hit_rank(gold):
            for i, h in enumerate(hits):
                if h.chunk["spec"] == gold["spec"] and clause_match(h.chunk["clause"], gold["clause"]):
                    return i
            return None

        required_g = [g for g in r["gold_clauses"] if g.get("required")]
        ranks = [hit_rank(g) for g in required]
        found = [x for x in ranks if x is not None]
        if required_g:   # all required clauses must surface
            r_at_5 += all(x is not None and x < 5 for x in ranks)
            r_at_20 += all(x is not None for x in ranks)
        else:            # sufficient semantics: any one gold clause counts
            r_at_5 += any(x is not None and x < 5 for x in ranks)
            r_at_20 += bool(found)
        mrr += 1.0 / (min(found) + 1) if found else 0.0
        status = "✓" if found else "✗"
        print(f"{status} {r['id']} ranks={ranks} {r['question'][:60]}")

    n = len(rows)
    out = {"n": n, "recall@5": r_at_5 / n, "recall@20": r_at_20 / n, "mrr@10": mrr / n}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run()
