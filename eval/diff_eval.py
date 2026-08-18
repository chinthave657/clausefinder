"""Diff-mode eval: pipeline mechanics + golden diff questions.

Two parts:
  1. Self-diff smoke test (always runs, no LLM call): Rel-X vs Rel-X must
     produce an empty deterministic diff and a "No changes" answer.
  2. Golden diff questions (eval/golden/diff_starter.jsonl): each runs the
     full pipeline IF both releases exist in the index; otherwise SKIPPED
     (the P0 index carries Rel-18 only — Rel-17 lands with the big ingest).
"""
from __future__ import annotations

import json
from pathlib import Path

from agent.diff import diff_releases
from agent.tools.retrieval import Retriever, load_acronyms


def index_releases(retriever: Retriever) -> set[str]:
    rows = retriever.tbl.to_arrow().select(["release"]).to_pylist()
    return {r["release"] for r in rows}


def smoke_self_diff(retriever: Retriever, acronyms: dict, release: str) -> bool:
    """Rel-X vs Rel-X exercises retrieval, alignment, and the deterministic
    diff; identical sides must short-circuit to 'No changes' before any LLM."""
    q = "What actions does the UE take upon reception of an RRCReconfiguration message?"
    answer, report, _tagmap, rendered = diff_releases(q, retriever, release, release,
                                                     acronyms=acronyms)
    ok = rendered == "" and answer.lower().startswith("no changes") and report.passed
    print(f"{'✓' if ok else '✗'} smoke self-diff {release} vs {release}: {answer[:90]}")
    return ok


def run(golden: Path = Path("eval/golden/diff_starter.jsonl"),
        db: Path = Path("data/index"), parsed: Path = Path("data/parsed")) -> dict:
    retriever = Retriever(db)
    acronyms = load_acronyms(parsed)
    avail = index_releases(retriever)
    print(f"releases in index: {sorted(avail)}")

    smoke_ok = smoke_self_diff(retriever, acronyms, sorted(avail)[-1])

    rows = [json.loads(l) for l in golden.open() if l.strip()]
    ran = skipped = passed = 0
    for r in rows:
        rel_a, rel_b = r["releases"]
        missing = [x for x in (rel_a, rel_b) if x not in avail]
        if missing:
            skipped += 1
            print(f"- {r['id']} SKIP (releases not in index: {', '.join(missing)})")
            continue
        ran += 1
        answer, report, _tagmap, rendered = diff_releases(
            r["question"], retriever, rel_a, rel_b, acronyms=acronyms)
        ok = report.passed
        passed += ok
        print(f"{'✓' if ok else '✗'} {r['id']} diff_lines={len(rendered.splitlines())} "
              f"validator={'pass' if ok else 'FAIL'} {r['question'][:60]}")

    out = {"smoke_self_diff": smoke_ok, "n": len(rows), "ran": ran,
           "skipped": skipped, "validator_passed": passed}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run()
