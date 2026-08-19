"""Exit-code CI gate: compares the latest eval/reports/answer_eval_*.json
against eval/baseline.json. Fails (exit 1) if validator pass rate or
gold-clause-cited rate drops more than 5 percentage points.

This is deliberately a *paired* gate on the two harness-computed rates, not
the aggregate judge score (design §5.1: aggregate judge score is reported
informational-only with a Wilson CI — n=100 can't see a 5pp regression on the
judge axis reliably; validator pass rate and gold-cited rate are exact
deterministic counts and can).

Usage:
  uv run python eval/ci_gate.py                # gate latest report vs baseline
  uv run python eval/ci_gate.py --set-baseline  # run a fresh eval, save it as baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from eval.answer_eval import DB, GOLDEN, PARSED, REPORTS_DIR
from eval.answer_eval import run as run_answer_eval

BASELINE_PATH = Path("eval/baseline.json")
DROP_THRESHOLD_PP = 0.05  # 5 percentage points


def find_latest_report(reports_dir: Path = REPORTS_DIR) -> Path | None:
    reports = sorted(reports_dir.glob("answer_eval_*.json"))
    return reports[-1] if reports else None


def _rates(d: dict) -> dict:
    return {"validator_pass_rate": d["validator_pass_rate"], "gold_cited_rate": d["gold_cited_rate"]}


def gate(baseline: dict, current: dict) -> list[str]:
    failures = []
    for key, label in (("validator_pass_rate", "validator pass rate"),
                        ("gold_cited_rate", "gold-clause cited rate")):
        drop = baseline[key] - current[key]
        if drop > DROP_THRESHOLD_PP:
            failures.append(
                f"{label} dropped {drop:.1%}: baseline {baseline[key]:.1%} -> "
                f"current {current[key]:.1%} (threshold {DROP_THRESHOLD_PP:.0%})")
    return failures


def set_baseline(limit: int | None, golden: Path, db: Path, parsed: Path) -> dict:
    result = run_answer_eval(golden=golden, db=db, parsed=parsed, limit=limit)
    baseline = {
        **_rates(result),
        "n": result["n"],
        "set_from_timestamp": result["timestamp"],
        "set_at": datetime.now(UTC).isoformat(),
    }
    BASELINE_PATH.write_text(json.dumps(baseline, indent=1))
    print(f"baseline written: {BASELINE_PATH} ({baseline})")
    return baseline


def run(report: Path | None = None, baseline_path: Path = BASELINE_PATH) -> int:
    report = report or find_latest_report()
    if report is None:
        print(f"WARN: no report found under {REPORTS_DIR} — nothing to gate. "
              "Run eval/answer_eval.py first. Not failing CI on a missing report.")
        return 0
    if not baseline_path.exists():
        print(f"WARN: no baseline at {baseline_path} — run with --set-baseline "
              "to create one. Not failing CI on a missing baseline.")
        return 0

    current = _rates(json.loads(report.read_text()))
    baseline = json.loads(baseline_path.read_text())
    print(f"report:   {report}")
    print(f"baseline: {baseline_path} ({baseline})")
    print(f"current:  {current}")

    failures = gate(baseline, current)
    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nGATE PASSED (no regression > 5pp).")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-baseline", action="store_true",
                   help="run a fresh answer_eval against the current dev-index state "
                        "and write eval/baseline.json from its metrics")
    p.add_argument("--limit", type=int, default=None,
                   help="forwarded to answer_eval when --set-baseline is used")
    p.add_argument("--report", type=Path, default=None,
                   help="explicit report to gate (default: latest in eval/reports/)")
    p.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    p.add_argument("--golden", type=Path, default=GOLDEN)
    p.add_argument("--db", type=Path, default=DB)
    p.add_argument("--parsed", type=Path, default=PARSED)
    args = p.parse_args()

    if args.set_baseline:
        set_baseline(args.limit, args.golden, args.db, args.parsed)
        sys.exit(0)

    sys.exit(run(report=args.report, baseline_path=args.baseline))
