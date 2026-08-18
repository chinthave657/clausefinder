"""Diff-mode CLI command. Kept out of cli/main.py; the orchestrator wires
`diff` into the typer app (`app.command()(diff)`), or run standalone:
    uv run python -m cli.diff_cmd "question" --from Rel-17 --to Rel-18
"""
from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

console = Console()

DB = Path("data/index")
PARSED = Path("data/parsed")


def diff(question: str,
         release_a: str = typer.Option("Rel-17", "--from", help="old release"),
         release_b: str = typer.Option("Rel-18", "--to", help="new release"),
         show_diff: bool = typer.Option(True, help="print the deterministic diff")):
    """Release-to-release clause diff with cited synthesis (Super-120B)."""
    from agent.diff import diff_releases
    from agent.tools.retrieval import Retriever, load_acronyms
    r = Retriever(DB)
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
    typer.run(diff)
