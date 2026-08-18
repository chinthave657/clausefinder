"""ClauseFinder — Gradio app for HF Spaces (ZeroGPU) and local CPU.

Three tabs = router modes (design §2, rule-first: the UI tab IS the mode):
  Ask     — hybrid retrieval + Nano synthesis with citation validator
  Diff    — release diff (Super-120B synthesis; graceful stub until wired)
  Explain — one clause fetched by metadata (no vector search) + Nano explain

Run locally:  uv run python web/app.py
"""
from __future__ import annotations

import html
import sys
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

_RETRIEVER = None
_ACRONYMS: dict[str, str] = {}


def _retriever():
    """Lazy singleton — embedding model load is the slow part (once)."""
    global _RETRIEVER, _ACRONYMS
    if _RETRIEVER is None:
        from agent.tools.retrieval import Retriever, load_acronyms
        _RETRIEVER = Retriever(DB)
        _ACRONYMS = load_acronyms(PARSED)
    return _RETRIEVER


def _releases() -> list[str]:
    try:
        tbl = _retriever().tbl
        rels = sorted({r["release"] for r in
                       tbl.search().select(["release"]).limit(10000).to_list()})
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
    n = answer.count("(see cited clause — paraphrased)")
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

@GPU
def _ask_pipeline(question: str, release: str | None):
    from agent.answer import ask
    return ask(question, _retriever(), release=release, acronyms=_ACRONYMS)


def run_ask(question: str, release_choice: str, history: list[dict]):
    question = (question or "").strip()
    if not question:
        yield history, "", ""
        return
    history = (history or []) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "_Retrieving clauses…_"}]
    yield history, "", ""
    try:
        answer, report, tagmap = _ask_pipeline(question, _rel(release_choice))
    except Exception as e:  # surface, never crash the queue
        yield (history[:-1] + [{"role": "assistant", "content": f"Error: {e}"}],
               "", "")
        return
    for h in _stream(history, answer):
        yield h, _sources_html(tagmap), ""
    yield (history[:-1] + [{"role": "assistant", "content": answer}],
           _sources_html(tagmap), _badges_html(report, answer))


# ----------------------------------------------------------------- explain

def run_explain(ref: str, release_choice: str, history: list[dict]):
    ref = (ref or "").strip()
    if not ref:
        yield history, "", ""
        return
    history = (history or []) + [
        {"role": "user", "content": f"Explain {ref}"},
        {"role": "assistant", "content": "_Fetching clause…_"}]
    yield history, "", ""
    try:
        from agent.explain import explain
        text, report, tagmap = explain(ref, _retriever(),
                                       release=_rel(release_choice))
    except Exception as e:
        yield (history[:-1] + [{"role": "assistant", "content": f"Error: {e}"}],
               "", "")
        return
    for h in _stream(history, text):
        yield h, _sources_html(tagmap), ""
    yield (history[:-1] + [{"role": "assistant", "content": text}],
           _sources_html(tagmap), _badges_html(report, text))


# -------------------------------------------------------------------- diff

def run_diff(question: str, rel_a: str, rel_b: str, history: list[dict]):
    question = (question or "").strip()
    if not question:
        yield history, "", ""
        return
    history = (history or []) + [
        {"role": "user", "content": f"Diff {rel_a} vs {rel_b}: {question}"},
        {"role": "assistant", "content": "_Resolving clause sets on both "
                                         "releases (~1 min)…_"}]
    yield history, "", ""
    try:
        from agent import diff as diff_mod  # owned by the diff workstream
    except ImportError:
        msg = ("Diff mode needs a second release in the index (a larger ingest "
               "is in progress) and the `agent/diff.py` Super-120B synthesis "
               "agent. It is not wired up in this build yet — Ask and Explain "
               "are fully functional.")
        yield history[:-1] + [{"role": "assistant", "content": msg}], "", ""
        return
    try:
        fn = getattr(diff_mod, "diff", None) or getattr(diff_mod, "run")
        text, report, tagmap = fn(question, _retriever(), rel_a, rel_b)
    except Exception as e:
        yield (history[:-1] + [{"role": "assistant", "content": f"Error: {e}"}],
               "", "")
        return
    for h in _stream(history, text):
        yield h, _sources_html(tagmap), ""
    yield (history[:-1] + [{"role": "assistant", "content": text}],
           _sources_html(tagmap), _badges_html(report, text))


# ---------------------------------------------------------------------- UI

EXAMPLES = [
    "What triggers an RRC re-establishment procedure?",
    "What does the UE do when timer T310 expires?",
    "What happens when the UE receives an RRCSetup message?",
    "When does the UE send a MeasurementReport?",
    "What is conditional handover and when is it executed?",
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
    default_rel = "Rel-18" if "Rel-18" in releases else releases[0]

    with gr.Blocks(title="ClauseFinder") as demo:
        gr.Markdown("# ClauseFinder\nClause-cited 3GPP answers and release "
                    "diffs on the NVIDIA agentic stack.")

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
                        ex_btns = [gr.Button(q, size="sm") for q in EXAMPLES]
                    ask_sources, ask_badges = _side_panel()
                for b, q in zip(ex_btns, EXAMPLES):
                    b.click(lambda q=q: q, outputs=ask_q)
                gr.on([ask_btn.click, ask_q.submit], run_ask,
                      inputs=[ask_q, ask_rel, ask_chat],
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
                                [r for r in releases if r != "All releases"] or
                                ["Rel-17"], value=default_rel,
                                label="Release A", scale=1)
                            diff_b = gr.Dropdown(
                                [r for r in releases if r != "All releases"] or
                                ["Rel-18"], value=default_rel,
                                label="Release B", scale=1)
                            diff_btn = gr.Button("Diff", variant="primary",
                                                 scale=1)
                        gr.Markdown("_Diff runs Super-120B with reasoning ON "
                                    "over a deterministic clause diff — "
                                    "expect ~1 minute._")
                    diff_sources, diff_badges = _side_panel()
                gr.on([diff_btn.click, diff_q.submit], run_diff,
                      inputs=[diff_q, diff_a, diff_b, diff_chat],
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
                    exp_sources, exp_badges = _side_panel()
                gr.on([exp_btn.click, exp_ref.submit], run_explain,
                      inputs=[exp_ref, exp_rel, exp_chat],
                      outputs=[exp_chat, exp_sources, exp_badges])

        gr.HTML(FOOTER)
    return demo


demo = build_demo()
demo.queue(max_size=20, default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS)
