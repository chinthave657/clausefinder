"""TeleQnA (GSMA ot-lite) MCQ benchmark: closed-book vs ClauseFinder-RAG.

Two configs on the same 1,000 questions:
  closed — Nemotron Nano answers from parametric knowledge only
  rag    — ClauseFinder retrieval context (rerank on) prepended

Honest-reporting notes baked in:
  * TeleQnA mixes sources (3GPP, IEEE, ITU, ETSI...). Our corpus covers five
    3GPP series, so results are reported overall AND split by source tag —
    retrieval can only help where the corpus covers the source.
  * Checkpointed per-question (JSONL append); rerun resumes. MoE serving is
    nondeterministic at temp 0 — single-run numbers carry that.
  * Answer extraction is deterministic regex over a forced JSON format, with
    one repair retry; unparseable after retry scores wrong (counted, reported).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL = os.environ.get("CLAUSEFINDER_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
NO_THINK = {"chat_template_kwargs": {"thinking": False}}
_ANS_RE = re.compile(r'"answer"\s*:\s*(\d+)')
_TAG_RE = re.compile(r"\[([A-Za-z0-9 .\-]+?)[\s\d.,]*\]\s*$")

CLOSED_SYS = ("You are a telecom standards expert. Answer the multiple-choice "
              "question. Reply with ONLY a JSON object: {\"answer\": <0-based "
              "choice index>}. No explanation.")
RAG_SYS = ("You are a telecom standards expert. Excerpts from 3GPP "
           "specifications are provided; use them when relevant, and your own "
           "knowledge otherwise. Reply with ONLY a JSON object: "
           "{\"answer\": <0-based choice index>}. No explanation.")


def _client():
    from openai import OpenAI
    if os.environ.get("NVIDIA_API_KEY"):
        return OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                      api_key=os.environ["NVIDIA_API_KEY"])
    raise SystemExit("set NVIDIA_API_KEY")


def _pick(client, system: str, user: str) -> int | None:
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0.0, max_tokens=800,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                extra_body=NO_THINK)
            text = resp.choices[0].message.content or ""
        except Exception as e:  # rate limit / 5xx: back off once
            time.sleep(20 if attempt == 0 else 60)
            if attempt:
                print(f"  API error, skipping: {type(e).__name__}")
                return None
            continue
        m = _ANS_RE.search(text)
        if m:
            return int(m.group(1))
        user = user + "\n\nReply with ONLY {\"answer\": <index>}."
    return None


def _source_tag(question: str) -> str:
    m = _TAG_RE.search(question.strip())
    tag = (m.group(1).strip().upper() if m else "UNTAGGED")
    for fam in ("3GPP", "IEEE", "ITU", "ETSI"):
        if tag.startswith(fam):
            return fam
    return "OTHER"


def _load_rows(limit: int | None) -> list[dict]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("GSMA/ot-lite", "test_teleqna.json",
                        repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    rows = json.load(open(p))
    return rows[:limit] if limit else rows


def _resume_set(out: Path) -> set[int]:
    """Crash-tolerant: a truncated final line must not kill the resume."""
    done: set[int] = set()
    if out.exists():
        for l in out.open():
            try:
                done.add(json.loads(l)["i"])
            except Exception:
                pass
    return done


def run(mode: str, limit: int | None, out_dir: Path) -> None:
    rows = _load_rows(limit)

    out = out_dir / f"teleqna_{mode}.jsonl"
    done = _resume_set(out)
    client = _client()

    retriever = acronyms = None
    if mode == "rag":
        from agent.tools.retrieval import Retriever, enhance_query, load_acronyms
        retriever = Retriever(Path("data/index"))
        acronyms = load_acronyms(Path("data/parsed"))

    with out.open("a") as f:
        for i, row in enumerate(rows):
            if i in done:
                continue
            q = row["question"]
            choices = "\n".join(f"{j}. {c}" for j, c in enumerate(row["choices"]))
            if mode == "closed":
                user = f"{q}\n\nChoices:\n{choices}"
            else:
                from agent.tools.retrieval import enhance_query
                hits = retriever.search(enhance_query(q, acronyms), top=6)
                ctx = "\n\n".join(
                    f"[{h.chunk['spec']} §{h.chunk['clause']}] "
                    f"{retriever.parent_text(h)[:1200]}" for h in hits[:4])
                user = (f"Excerpts:\n{ctx}\n\nQuestion: {q}\n\n"
                        f"Choices:\n{choices}")
            pred = _pick(client, CLOSED_SYS if mode == "closed" else RAG_SYS, user)
            rec = {"i": i, "pred": pred, "gold": row["answer"],
                   "correct": pred == row["answer"],
                   "source": _source_tag(q), "parse_fail": pred is None}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"{mode}: {i+1}/{len(rows)}", flush=True)

    recs = [json.loads(l) for l in out.open()]
    n = len(recs)
    acc = sum(r["correct"] for r in recs) / n
    by = {}
    for r in recs:
        by.setdefault(r["source"], []).append(r["correct"])
    print(json.dumps({
        "mode": mode, "model": MODEL, "n": n, "accuracy": round(acc, 4),
        "parse_failures": sum(r["parse_fail"] for r in recs),
        "by_source": {k: {"n": len(v), "acc": round(sum(v)/len(v), 4)}
                      for k, v in sorted(by.items())},
    }, indent=1))


def run_scores(limit: int | None, out_dir: Path) -> None:
    """Retrieval-only pass: top reranker score per question (no LLM calls).
    Feeds the selective mode; requires CLAUSEFINDER_RERANK=1."""
    from agent.tools.retrieval import Retriever, enhance_query, load_acronyms
    rows = _load_rows(limit)
    r = Retriever(Path("data/index"))
    acr = load_acronyms(Path("data/parsed"))
    out = out_dir / "teleqna_scores.jsonl"
    done = _resume_set(out)
    with out.open("a") as f:
        for i, row in enumerate(rows):
            if i in done:
                continue
            hits = r.search(enhance_query(row["question"], acr), top=6)
            top = max((h.rerank_score for h in hits
                       if h.rerank_score is not None), default=0.0)
            f.write(json.dumps({"i": i, "top_score": round(float(top), 4)}) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                print(f"scores: {i+1}/{len(rows)}", flush=True)
    print("scores complete")


def run_selective(tau: float, out_dir: Path) -> None:
    """Post-hoc combine: pick the rag prediction where top_score >= tau, else
    the closed prediction. Needs closed + rag + scores JSONLs (no LLM calls).
    tau defaults to the abstain rail's a-priori soft floor (answer.ABSTAIN_SOFT)."""
    closed = {json.loads(l)["i"]: json.loads(l) for l in (out_dir / "teleqna_closed.jsonl").open()}
    rag = {json.loads(l)["i"]: json.loads(l) for l in (out_dir / "teleqna_rag.jsonl").open()}
    scores = {json.loads(l)["i"]: json.loads(l)["top_score"] for l in (out_dir / "teleqna_scores.jsonl").open()}
    ids = sorted(set(closed) & set(rag) & set(scores))
    by, correct, fired = {}, 0, 0
    for i in ids:
        pick = rag[i] if scores[i] >= tau else closed[i]
        fired += scores[i] >= tau
        correct += pick["correct"]
        by.setdefault(closed[i]["source"], []).append(pick["correct"])
    print(json.dumps({
        "mode": "selective", "tau": tau, "n": len(ids),
        "accuracy": round(correct / len(ids), 4),
        "rag_fired": round(fired / len(ids), 4),
        "by_source": {k: {"n": len(v), "acc": round(sum(v) / len(v), 4)}
                      for k, v in sorted(by.items())},
    }, indent=1))


if __name__ == "__main__":
    from agent.answer import ABSTAIN_SOFT
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["closed", "rag", "scores", "selective"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tau", type=float, default=ABSTAIN_SOFT)
    ap.add_argument("--out", type=Path, default=Path("eval/reports"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.mode == "scores":
        run_scores(a.limit, a.out)
    elif a.mode == "selective":
        run_selective(a.tau, a.out)
    else:
        run(a.mode, a.limit, a.out)
