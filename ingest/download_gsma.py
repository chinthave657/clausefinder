"""Download spec markdown from the official GSMA/3GPP mirror on Hugging Face.

Layout: marked/Rel-XX/{NN}_series/{spec}/raw.md  (images alongside, skipped here).
"""
from __future__ import annotations

from pathlib import Path

import typer
from huggingface_hub import snapshot_download

DATASET = "GSMA/3GPP"
DEFAULT_SERIES = [23, 24, 29, 33, 38]
DEFAULT_RELEASES = ["Rel-17", "Rel-18"]

app = typer.Typer(add_completion=False)


@app.command()
def download(
    out: Path = typer.Option(Path("data/corpus"), help="Local corpus root"),
    series: list[int] = typer.Option(DEFAULT_SERIES, help="3GPP series numbers"),
    releases: list[str] = typer.Option(DEFAULT_RELEASES, help="Releases, e.g. Rel-18"),
    spec: list[str] = typer.Option(None, help="Restrict to specific specs, e.g. 38331"),
) -> None:
    patterns = []
    for rel in releases:
        for s in series:
            if spec:
                patterns += [f"marked/{rel}/{s}_series/{sp}/raw.md" for sp in spec]
            else:
                patterns.append(f"marked/{rel}/{s}_series/*/raw.md")
    path = snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=out,
    )
    n = sum(1 for _ in Path(path).rglob("raw.md"))
    typer.echo(f"downloaded {n} spec files under {path}")


if __name__ == "__main__":
    app()
