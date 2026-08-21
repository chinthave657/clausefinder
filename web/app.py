"""ClauseFinder — Gradio app for HF Spaces (ZeroGPU) and local CPU.

Three tabs = router modes (design §2, rule-first: the UI tab IS the mode):
  Ask     — hybrid retrieval + Nemotron synthesis (CLAUSEFINDER_MODEL_CHAIN) with citation validator
  Diff    — release diff (deterministic clause diff + Super-120B synthesis)
  Explain — one clause fetched by metadata (no vector search) + Nano explain

Run locally:  uv run python web/app.py
"""
from __future__ import annotations

import contextlib
import datetime as dt
import html
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import gradio as gr

try:  # ZeroGPU on HF Spaces; identity decorator on local CPU
    import spaces
    GPU = spaces.GPU
except ImportError:
    def GPU(fn=None, **_kw):
        return fn if fn is not None else (lambda f: f)

from web.theme import CSS, theme

DB = ROOT / "data" / "index"
PARSED = ROOT / "data" / "parsed"
INDEX_DATASET = os.environ.get("CLAUSEFINDER_INDEX_DATASET", "chinthave/clausefinder-index")


def _bootstrap_index() -> None:
    """On HF Spaces (ephemeral disk) pull the prebuilt index from the gated
    dataset at boot. Version-proof: lists files via the paginated HTTP tree
    API (the hub library preinstalled in Space images returns EMPTY legacy
    'siblings' for repos >1k files, so snapshot_download sees nothing) and
    downloads each explicitly via hf_hub_download."""
    if DB.exists() or not os.environ.get("HF_TOKEN"):
        return
    from concurrent.futures import ThreadPoolExecutor

    import requests
    from huggingface_hub import hf_hub_download

    token = os.environ["HF_TOKEN"]
    url = (f"https://huggingface.co/api/datasets/{INDEX_DATASET}"
           "/tree/main/data?recursive=1&expand=0")
    paths: list[str] = []
    while url:
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"},
                         timeout=60)
        r.raise_for_status()
        paths += [e["path"] for e in r.json() if e.get("type") == "file"]
        url = r.links.get("next", {}).get("url")
    wanted = [p for p in paths
              if (p.startswith("data/index/")
                  or (p.startswith("data/parsed/")
                      and p.endswith((".json", ".jsonl"))))
              and "/chunks.jsonl" not in p]      # 600MB, app never reads it
    print(f"bootstrapping {len(wanted)} files from {INDEX_DATASET} …",
          flush=True)

    def fetch(p: str) -> None:
        hf_hub_download(INDEX_DATASET, p, repo_type="dataset", token=token,
                        local_dir=ROOT)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(fetch, wanted))
    n = sum(1 for f in DB.rglob("*") if f.is_file()) if DB.exists() else 0
    sz = (sum(f.stat().st_size for f in DB.rglob("*") if f.is_file())
          if DB.exists() else 0)
    print(f"index bootstrap done: {n} files, {sz / 1e9:.2f} GB", flush=True)


_bootstrap_index()
REPO_URL = "https://github.com/chinthave657/clausefinder"

_RETRIEVER = None
_ACRONYMS: dict[str, str] = {}
_INDEX_ERROR: str | None = None


class IndexNotReady(RuntimeError):
    """Raised when data/index is missing or incomplete — friendly banner,
    never a raw traceback, since a bigger ingest may be running elsewhere."""


def _retriever():
    """Lazy singleton — embedding model load is the slow part (once). A
    failed load is cached (not retried every request) but reported clearly."""
    global _RETRIEVER, _ACRONYMS, _INDEX_ERROR
    if _RETRIEVER is None and _INDEX_ERROR is None:
        try:
            from agent.tools.retrieval import Retriever, load_acronyms
            _RETRIEVER = Retriever(DB)
            _ACRONYMS = load_acronyms(PARSED)
        except Exception as e:  # missing data/index, empty table, bad path…
            _INDEX_ERROR = str(e)
    if _INDEX_ERROR is not None:
        raise IndexNotReady(_INDEX_ERROR)
    return _RETRIEVER


if os.environ.get("SPACE_ID"):  # HF Space: pre-load weights on CPU at boot
    try:
        _r = _retriever()
        _r._rerank_hits("warmup", [{"breadcrumb": "w", "text": "w", "id": "w",
                                    "parent_id": "w"}])
        print("models warm (CPU)", flush=True)
    except Exception as _e:
        print("warmup skipped:", _e, flush=True)


INDEX_BANNER = (
    "**The clause index is still building.** This demo needs a LanceDB "
    "index at `data/index/` (a larger ingest may be running elsewhere). "
    f"Check back soon, or browse the repo: [{REPO_URL}]({REPO_URL})"
)


# --------------------------------------------------------------- rate limit

DAILY_LIMIT = int(os.environ.get("DEMO_DAILY_LIMIT", "10"))
_QUOTA: dict[str, tuple[dt.date, int]] = {}
_QUOTA_LOCK = threading.Lock()

CAPACITY_BANNER = (
    f"**Daily query limit reached ({DAILY_LIMIT}/day) for this session.** "
    "Connect your own OpenRouter key in Settings above to keep going — it "
    f"bypasses this limit — or self-host: [{REPO_URL}]({REPO_URL})"
)


def _consume_quota(request: "gr.Request | None", bypass: bool) -> bool:
    """One counter per browser session (request.session_hash), reset daily.
    BYO-key sessions bypass the counter entirely (design §4)."""
    if bypass:
        return True
    sid = getattr(request, "session_hash", None) or "anon"
    today = dt.date.today()
    with _QUOTA_LOCK:
        day, count = _QUOTA.get(sid, (today, 0))
        if day != today:
            day, count = today, 0
        if count >= DAILY_LIMIT:
            _QUOTA[sid] = (day, count)
            return False
        _QUOTA[sid] = (day, count + 1)
        return True


# ------------------------------------------------------------------ BYO key

_ENV_LOCK = threading.Lock()


@contextlib.contextmanager
def _borrowed_key(api_key: str | None):
    """explain()/diff_releases() read OPENROUTER_API_KEY from the environment
    rather than taking a per-call key; borrow the env var for the duration
    of the call instead. Serializes
    concurrent BYO-key requests, which is fine at the demo's queue
    concurrency (2-3)."""
    if not api_key:
        yield
        return
    with _ENV_LOCK:
        prev = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = api_key
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = prev


def _releases() -> list[str]:
    try:
        tbl = _retriever().tbl
        # full single-column scan: a row-limit sample misses releases that
        # don't appear early in insertion order (Rel-18 sat after 147k Rel-17
        # rows and vanished from the dropdown)
        rels = sorted(set(tbl.to_arrow().select(["release"])["release"].to_pylist()))
    except Exception:
        rels = ["Rel-18"]
    return ["All releases"] + rels


def _rel(choice: str | None) -> str | None:
    return None if not choice or choice.startswith("All") else choice


# ---------------------------------------------------------------- rendering

def _sources_html(tagmap: dict[str, dict]) -> str:
    if not tagmap:
        return "<p><em>No sources.</em></p>"
    cards = []
    for tag, c in tagmap.items():
        link = (f'<a href="{html.escape(c.get("url", "#"))}" target="_blank" '
                f'rel="noopener">spec page ↗</a>')
        cards.append(
            f'<div class="source-card"><strong>{html.escape(tag)}</strong> '
            f'{html.escape(c["spec"])} §{html.escape(c["clause"])} '
            f'{html.escape(c.get("clause_title", ""))}<br>'
            f'<span class="meta">{html.escape(c["release"])} '
            f'V{html.escape(c.get("version", "?"))} · {link}</span></div>')
    return "\n".join(cards)


def _badges_html(report, answer: str) -> str:
    """Validator report → green (verbatim/valid) and amber (paraphrased /
    re-anchored) and red (failed) badges."""
    if not report.checks:
        return ('<span class="badge badge-red">no citations validated</span>'
                if not report.passed else "")
    out = []
    for c in report.checks:
        if c.ok and not c.reason:
            out.append(f'<span class="badge badge-green">{html.escape(c.tag)} '
                       'verbatim ✓</span>')
        elif c.ok:  # re-anchored quote: grounded, attribution corrected
            out.append(f'<span class="badge badge-amber">{html.escape(c.tag)} '
                       f'{html.escape(c.reason)}</span>')
        else:
            out.append(f'<span class="badge badge-red">{html.escape(c.tag)} '
                       f'{html.escape(c.reason or "failed")}</span>')
    n = answer.count("(quote omitted")
    if n:
        out.append(f'<span class="badge badge-amber">{n} quote(s) replaced — '
                   'paraphrased</span>')
    verdict = ('<span class="badge badge-green">all citations validated</span>'
               if report.passed else
               '<span class="badge badge-amber">review citations above</span>')
    return " ".join(out) + "<br>" + verdict


def _stream(history: list[dict], text: str):
    """Stream a finished answer into the last assistant message, chunked."""
    words = text.split(" ")
    acc = []
    for i, w in enumerate(words):
        acc.append(w)
        if i % 6 == 5 or i == len(words) - 1:
            yield history[:-1] + [{"role": "assistant", "content": " ".join(acc)}]


# --------------------------------------------------------------------- ask

@GPU(duration=25)  # device-hop + embed + rerank; cold model load happens at
                    # boot on CPU, not inside the window (was aborting at 15s)
def _gpu_retrieve(question: str, release: str | None):
    """Only this needs CUDA (query embed + rerank, ~1-2s). Runs inside the
    ZeroGPU window; LLM synthesis (network-bound) runs outside it so a slow
    API call can never abort the GPU task."""
    import torch

    from agent.tools.retrieval import enhance_query
    r = _retriever()
    if torch.cuda.is_available():
        r.to_device("cuda")
    return r.search(enhance_query(question, _ACRONYMS), release=release)


@GPU(duration=40)  # two releases x clause-set retrieval + device hop
def _gpu_diff_resolve(question: str, release_a: str, release_b: str):
    import torch

    from agent.diff import resolve_clause_sets
    r = _retriever()
    if torch.cuda.is_available():
        r.to_device("cuda")
    return resolve_clause_sets(question, (release_a, release_b), r, _ACRONYMS)


def _zerogpu_broken(e: Exception) -> bool:
    """Any failure inside a GPU-window function is CPU-retryable — those
    functions only do retrieval. Spaces strips exception details when
    re-raising from the worker (a broken allocation surfaces as
    gr.Error("RuntimeError") with no CUDA/quota text), so string-matching
    the cause is impossible; the only unretryable case is a missing index."""
    return not isinstance(e, IndexNotReady)


def _cpu_fallback(fn, *args):
    """Re-run a retrieval step on CPU with the reranker off (50 cross-encoder
    pairs on 2 vCPU would take minutes; fused order is the honest degraded
    mode). Answers stay cited; the abstain gate is inactive without rerank
    scores, which the weak-match caveat partially covers."""
    r = _retriever()
    prev = getattr(r, "rerank", None)
    try:
        if prev is not None:
            r.rerank = False
        return fn(*args)
    finally:
        if prev is not None:
            r.rerank = prev


def _ask_pipeline(question: str, release: str | None, api_key: str | None):
    from agent.answer import ask
    from agent.tools.retrieval import enhance_query
    try:
        hits = _gpu_retrieve(question, release)
    except Exception as e:
        if not _zerogpu_broken(e):
            raise
        print(f"ZeroGPU failed ({type(e).__name__}: {str(e)[:80]!r}) — CPU fallback", flush=True)
        hits = _cpu_fallback(
            lambda: _retriever().search(enhance_query(question, _ACRONYMS),
                                        release=release))
    return ask(question, _retriever(), release=release, acronyms=_ACRONYMS,
               api_key=api_key or None, hits=hits)


CONDENSE_SYSTEM = (
    "Rewrite the user's latest message as ONE standalone 3GPP question, "
    "resolving pronouns and references (e.g. 'explain more', 'what about "
    "Rel-18?') from the conversation. Preserve all technical identifiers "
    "exactly. If it is already standalone, return it unchanged. Return ONLY "
    "the question text.")


def _standalone_question(question: str, history: list[dict],
                         api_key: str | None) -> str:
    """Follow-up condensation: retrieval is stateless, so anaphoric
    follow-ups ('explain more') must become standalone questions first.
    One cheap Nano call; on any failure fall back to the raw question."""
    turns = [m for m in (history or []) if m.get("role") in ("user", "assistant")][-6:]
    if not turns:
        return question
    try:
        from agent.answer import NO_THINK, _client, _content
        client, model = _client(api_key)
        def _clip(s: str, n: int = 400) -> str:
            # word-boundary clip: mid-word truncation degrades the condenser
            return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"
        convo = "\n".join(f"{m['role']}: {_clip(str(m['content']))}" for m in turns)
        resp = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=120,
            messages=[{"role": "system", "content": CONDENSE_SYSTEM},
                      {"role": "user",
                       "content": f"Conversation:\n{convo}\n\nLatest message: {question}"}],
            extra_body=NO_THINK)
        out = _content(resp).strip().strip('"')
        return out if 5 < len(out) < 400 else question
    except Exception:
        return question


def run_ask(question: str, release_choice: str, history: list[dict],
            api_key: str, request: gr.Request):
    question = (question or "").strip()
    if not question:
        yield history, "", ""
        return
    raw_history = list(history or [])
    history = raw_history + [{"role": "user", "content": question}]
    if raw_history:
        question = _standalone_question(question, raw_history, api_key or None)
    if not _consume_quota(request, bypass=bool(api_key)):
        yield (history + [{"role": "assistant", "content": CAPACITY_BANNER}],
               "", "")
        return
    history = history + [{"role": "assistant", "content": "_Retrieving clauses…_"}]
    yield history, "", ""
    try:
        answer, report, tagmap = _ask_pipeline(question, _rel(release_choice), api_key)
    except IndexNotReady:
        yield history[:-1] + [{"role": "assistant", "content": INDEX_BANNER}], "", ""
        return
    except Exception as e:  # surface, never crash the queue
        yield (history[:-1] + [{"role": "assistant", "content": f"Error: {e}"}],
               "", "")
        return
    for h in _stream(history, answer):
        yield h, _sources_html(tagmap), ""
    yield (history[:-1] + [{"role": "assistant", "content": answer}],
           _sources_html(tagmap), _badges_html(report, answer))


# ----------------------------------------------------------------- explain

def run_explain(ref: str, release_choice: str, history: list[dict],
                api_key: str, request: gr.Request):
    ref = (ref or "").strip()
    if not ref:
        yield history, "", ""
        return
    history = (history or []) + [{"role": "user", "content": f"Explain {ref}"}]
    if not _consume_quota(request, bypass=bool(api_key)):
        yield (history + [{"role": "assistant", "content": CAPACITY_BANNER}],
               "", "")
        return
    history = history + [{"role": "assistant", "content": "_Fetching clause…_"}]
    yield history, "", ""
    try:
        from agent.explain import explain
        with _borrowed_key(api_key):
            text, report, tagmap = explain(ref, _retriever(),
                                           release=_rel(release_choice))
    except IndexNotReady:
        yield history[:-1] + [{"role": "assistant", "content": INDEX_BANNER}], "", ""
        return
    except Exception as e:
        yield (history[:-1] + [{"role": "assistant", "content": f"Error: {e}"}],
               "", "")
        return
    for h in _stream(history, text):
        yield h, _sources_html(tagmap), ""
    yield (history[:-1] + [{"role": "assistant", "content": text}],
           _sources_html(tagmap), _badges_html(report, text))


# -------------------------------------------------------------------- diff

def run_diff(question: str, rel_a: str, rel_b: str, history: list[dict],
             api_key: str, request: gr.Request):
    question = (question or "").strip()
    if not question:
        yield history, "", ""
        return
    history = (history or []) + [
        {"role": "user", "content": f"Diff {rel_a} vs {rel_b}: {question}"}]
    if not _consume_quota(request, bypass=bool(api_key)):
        yield (history + [{"role": "assistant", "content": CAPACITY_BANNER}],
               "", "")
        return
    history = history + [{"role": "assistant", "content": "_Resolving clause "
                          "sets on both releases (~1 min)…_"}]
    yield history, "", ""
    from agent import diff as diff_mod
    try:
        try:
            sides = _gpu_diff_resolve(question, rel_a, rel_b)   # GPU window: retrieval only
        except Exception as e:
            if not _zerogpu_broken(e):
                raise
            print(f"ZeroGPU failed ({type(e).__name__}: {str(e)[:80]!r}) — CPU fallback (diff)", flush=True)
            from agent.diff import resolve_clause_sets
            sides = _cpu_fallback(
                lambda: resolve_clause_sets(question, (rel_a, rel_b),
                                            _retriever(), _ACRONYMS))
        with _borrowed_key(api_key):                        # synthesis: network-bound, no GPU
            result = diff_mod.diff_releases(question, _retriever(), rel_a, rel_b,
                                            acronyms=_ACRONYMS, sides=sides)
        text, report, tagmap = result[0], result[1], result[2]
    except IndexNotReady:
        yield history[:-1] + [{"role": "assistant", "content": INDEX_BANNER}], "", ""
        return
    except Exception as e:
        yield (history[:-1] + [{"role": "assistant", "content": f"Error: {e}"}],
               "", "")
        return
    for h in _stream(history, text):
        yield h, _sources_html(tagmap), ""
    yield (history[:-1] + [{"role": "assistant", "content": text}],
           _sources_html(tagmap), _badges_html(report, text))


# ---------------------------------------------------------------------- UI

ASK_EXAMPLES = [
    "What triggers an RRC re-establishment procedure?",
    "What does the UE do when timer T310 expires?",
    "What happens when the UE receives an RRCSetup message?",
    "When does the UE send a MeasurementReport?",
    "What is conditional handover and when is it executed?",
]

DIFF_EXAMPLES = [
    "How did conditional handover change?",
    "What changed in RRC re-establishment?",
    "Did T310 timer handling change?",
    "What's new in MeasurementReport triggering conditions?",
]

EXPLAIN_EXAMPLES = [
    "TS 38.331 5.3.5.3",
    "TS 38.331 5.3.7",
    "TS 38.331 5.3.10",
    "TS 38.331 5.3.13",
]

FOOTER = ('<div class="cf-footer"><strong>Answers cite official 3GPP specs '
          '— verify against the referenced clause before use.</strong> '
          'ClauseFinder is a research assistant, not a normative source.</div>')


def _side_panel():
    """Shared right-hand column: sources + validator report."""
    with gr.Column(scale=1):
        gr.Markdown("#### Sources")
        sources = gr.HTML('<p><em>Sources appear here after a query.</em></p>')
        gr.Markdown("#### Citation validator")
        badges = gr.HTML("")
    return sources, badges


def build_demo() -> gr.Blocks:
    releases = _releases()
    rel_choices = [r for r in releases if r != "All releases"] or ["Rel-18"]
    default_rel = "Rel-18" if "Rel-18" in rel_choices else rel_choices[0]

    with gr.Blocks(title="ClauseFinder") as demo:
        gr.Markdown("# 3GPP ClauseFinder\nClause-cited 3GPP answers and release "
                    "diffs on the NVIDIA agentic stack.")

        api_key_state = gr.State("")
        with gr.Accordion("Settings — bring your own OpenRouter key (optional)",
                          open=False):
            gr.Markdown(
                f"This demo allows **{DAILY_LIMIT} free queries/day** per "
                "browser session. Paste an [OpenRouter](https://openrouter.ai/keys) "
                "API key to use your own quota instead — it is kept only in "
                "this browser session's memory, used solely for your queries, "
                "and never logged or persisted server-side.")
            api_key_box = gr.Textbox(
                label="OpenRouter API key", type="password",
                placeholder="sk-or-v1-…", show_label=True)
            api_key_box.change(lambda k: (k or "").strip(),
                               inputs=api_key_box, outputs=api_key_state)

        with gr.Tabs():
            # ------------------------------------------------------- Ask
            with gr.Tab("Ask"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ask_chat = gr.Chatbot(label="Answer", height=420)
                        ask_q = gr.Textbox(label="Question",
                                           placeholder="e.g. What triggers an "
                                                       "RRC re-establishment?")
                        with gr.Row():
                            ask_rel = gr.Dropdown(releases, value=default_rel,
                                                  label="Release", scale=1)
                            ask_btn = gr.Button("Ask", variant="primary", scale=1)
                        gr.Markdown("**Examples**")
                        ex_btns = [gr.Button(q, size="sm") for q in ASK_EXAMPLES]
                    ask_sources, ask_badges = _side_panel()
                for b, q in zip(ex_btns, ASK_EXAMPLES):
                    b.click(lambda q=q: q, outputs=ask_q)
                gr.on([ask_btn.click, ask_q.submit], run_ask,
                      inputs=[ask_q, ask_rel, ask_chat, api_key_state],
                      outputs=[ask_chat, ask_sources, ask_badges])

            # ------------------------------------------------------ Diff
            with gr.Tab("Diff"):
                with gr.Row():
                    with gr.Column(scale=2):
                        diff_chat = gr.Chatbot(label="Release diff", height=420)
                        diff_q = gr.Textbox(
                            label="What changed?",
                            placeholder="e.g. How did conditional handover "
                                        "change between releases?")
                        with gr.Row():
                            diff_a = gr.Dropdown(
                                rel_choices, value=default_rel,
                                label="Release A", scale=1)
                            diff_b = gr.Dropdown(
                                rel_choices, value=default_rel,
                                label="Release B", scale=1)
                            diff_btn = gr.Button("Diff", variant="primary",
                                                 scale=1)
                        gr.Markdown("_Diff runs Super-120B with reasoning ON "
                                    "over a deterministic clause diff — "
                                    "expect ~1 minute._")
                        gr.Markdown("**Examples**")
                        diff_ex_btns = [gr.Button(q, size="sm") for q in DIFF_EXAMPLES]
                    diff_sources, diff_badges = _side_panel()
                for b, q in zip(diff_ex_btns, DIFF_EXAMPLES):
                    b.click(lambda q=q: q, outputs=diff_q)
                gr.on([diff_btn.click, diff_q.submit], run_diff,
                      inputs=[diff_q, diff_a, diff_b, diff_chat, api_key_state],
                      outputs=[diff_chat, diff_sources, diff_badges])

            # --------------------------------------------------- Explain
            with gr.Tab("Explain"):
                with gr.Row():
                    with gr.Column(scale=2):
                        exp_chat = gr.Chatbot(label="Explanation", height=420)
                        exp_ref = gr.Textbox(
                            label="Clause reference",
                            placeholder="e.g. TS 38.331 5.3.5.3")
                        with gr.Row():
                            exp_rel = gr.Dropdown(releases, value=default_rel,
                                                  label="Release", scale=1)
                            exp_btn = gr.Button("Explain", variant="primary",
                                                scale=1)
                        gr.Markdown("Fetches the named clause directly by "
                                    "metadata — no vector search — and "
                                    "explains it in plain English.")
                        gr.Markdown("**Examples**")
                        exp_ex_btns = [gr.Button(q, size="sm") for q in EXPLAIN_EXAMPLES]
                    exp_sources, exp_badges = _side_panel()
                for b, q in zip(exp_ex_btns, EXPLAIN_EXAMPLES):
                    b.click(lambda q=q: q, outputs=exp_ref)
                gr.on([exp_btn.click, exp_ref.submit], run_explain,
                      inputs=[exp_ref, exp_rel, exp_chat, api_key_state],
                      outputs=[exp_chat, exp_sources, exp_badges])

        gr.HTML(FOOTER)
    return demo


demo = build_demo()
demo.queue(max_size=20, default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS,
                server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                server_port=int(os.environ.get("PORT", "7860")))
