"""ClauseFinder CLI: ask | search | ingest."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

DB = Path("data/index")
PARSED = Path("data/parsed")


@app.command()
def search(query: str, release: str = typer.Option(None), top: int = 8):
    """Retrieval only — inspect what the index returns (no LLM)."""
    from agent.tools.retrieval import Retriever
    r = Retriever(DB)
    for hit in r.search(query, release=release, top=top):
        c = hit.chunk
        console.print(f"[bold]{hit.tag}[/] {c['spec']} §{c['clause']} "
                      f"[dim]{c['clause_title']}[/] ({c['release']}) score={hit.score:.4f}")
        console.print("   " + c["text"][:220].replace("\n", " ") + "…\n")


@app.command()
def ask(question: str, release: str = typer.Option(None)):
    """Clause-cited answer (LLM via NVIDIA_API_KEY or OPENROUTER_API_KEY)."""
    import json
    from agent.answer import ask as _ask
    from agent.tools.retrieval import Retriever, load_acronyms
    r = Retriever(DB)
    answer, report, tagmap = _ask(question, r, release=release,
                                  acronyms=load_acronyms(PARSED))
    console.print(Panel(answer, title="answer", border_style="green" if report.passed else "yellow"))
    for c in report.checks:
        mark = "[green]✓[/]" if c.ok else f"[red]✗ {c.reason}[/]"
        console.print(f"  {c.tag} {mark}")
    if tagmap:
        console.print("\n[dim]sources:[/]")
        for tag, c in tagmap.items():
            console.print(f"  [dim]{tag} {c['spec']} §{c['clause']} {c['release']} — {c['url']}[/]")


@app.command()
def ingest(corpus: Path = Path("data/corpus")):
    """Parse downloaded corpus into chunks + edges."""
    from ingest.chunker import parse_corpus
    parse_corpus(corpus, PARSED)


if __name__ == "__main__":
    app()
