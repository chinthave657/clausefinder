"""ClauseFinder CLI: ask | search | ingest."""
from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

app = typer.Typer(add_completion=False, no_args_is_help=True)

def _open_retriever():
    from agent.tools.retrieval import Retriever
    try:
        return Retriever(DB)
    except Exception as e:
        if "chunks" in str(e) or "not found" in str(e).lower() or not DB.exists():
            console.print("[red]No index at data/index[/] — build one first: "
                          "[bold]make ingest index[/] (see README quickstart)")
            raise typer.Exit(code=1)
        raise

console = Console()

DB = Path("data/index")
PARSED = Path("data/parsed")


@app.command()
def search(query: str, release: str = typer.Option(None), top: int = 8):
    """Retrieval only — inspect what the index returns (no LLM)."""
    r = _open_retriever()
    for hit in r.search(query, release=release, top=top):
        c = hit.chunk
        console.print(f"[bold]{hit.tag}[/] {c['spec']} §{c['clause']} "
                      f"[dim]{c['clause_title']}[/] ({c['release']}) score={hit.score:.4f}")
        console.print("   " + c["text"][:220].replace("\n", " ") + "…\n")


@app.command()
def ask(question: str, release: str = typer.Option(None)):
    """Clause-cited answer (LLM via NVIDIA_API_KEY or OPENROUTER_API_KEY)."""
    from agent.answer import ask as _ask
    from agent.tools.retrieval import load_acronyms
    r = _open_retriever()
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


@app.command()
def diff(question: str,
         release_a: str = typer.Option("Rel-17", "--from", help="old release"),
         release_b: str = typer.Option("Rel-18", "--to", help="new release"),
         show_diff: bool = typer.Option(True, help="print the deterministic diff")):
    """Release-to-release clause diff with cited synthesis (Super-120B).

    Gracefully reports when a requested release is absent from the index —
    the dev index carries Rel-18 only until the full corpus lands (design §6 P3)."""
    from agent.diff import diff_releases
    from agent.tools.retrieval import load_acronyms
    r = _open_retriever()
    avail = {row["release"] for row in r.tbl.to_arrow().select(["release"]).to_pylist()}
    missing = [rel for rel in (release_a, release_b) if rel not in avail]
    if missing:
        console.print(f"[yellow]Release(s) not in index: {', '.join(missing)}. "
                      f"Indexed releases: {', '.join(sorted(avail)) or '(none)'}[/]")
        raise typer.Exit(code=1)
    answer, report, tagmap, rendered = diff_releases(
        question, r, release_a, release_b, acronyms=load_acronyms(PARSED))
    if show_diff and rendered:
        console.print(Panel(rendered, title=f"deterministic diff {release_a} -> {release_b}",
                            border_style="cyan"))
    console.print(Panel(answer, title="diff summary",
                        border_style="green" if report.passed else "yellow"))
    for c in report.checks:
        mark = "[green]✓[/]" if c.ok else f"[red]✗ {c.reason}[/]"
        console.print(f"  {c.tag} {mark}")
    if tagmap:
        console.print("\n[dim]sources:[/]")
        for tag, c in tagmap.items():
            console.print(f"  [dim]{tag} {c['spec']} §{c['clause']} {c['release']} — {c['url']}[/]")


if __name__ == "__main__":
    app()
