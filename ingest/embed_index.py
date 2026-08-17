"""Embed child chunks with OTel-Embedding-568M and build the LanceDB index.

No ANN index is created on purpose: at P0 corpus size, LanceDB brute-force
(flat) search is exact — the literature (Telco-RAG) measured HNSW losing
accuracy vs flat on 3GPP retrieval. FTS (tantivy BM25) rides the same table.
"""
from __future__ import annotations

import json
from pathlib import Path

import lancedb
import typer
from sentence_transformers import SentenceTransformer

MODEL_ID = "farbodtavakkoli/OTel-Embedding-568M"
EMBED_MAX_TOKENS = 1400  # OTel fine-tune length; inputs truncated by tokenizer

app = typer.Typer(add_completion=False)


def load_model() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    model.max_seq_length = EMBED_MAX_TOKENS
    return model


@app.command()
def build(
    parsed: Path = typer.Option(Path("data/parsed"), help="chunker output dir"),
    db_path: Path = typer.Option(Path("data/index"), help="LanceDB dir"),
    batch: int = typer.Option(32),
) -> None:
    rows = [json.loads(l) for l in (parsed / "chunks.jsonl").open()]
    typer.echo(f"embedding {len(rows)} chunks with {MODEL_ID} …")
    model = load_model()
    texts = [r["breadcrumb"] + "\n" + r["text"] for r in rows]
    vecs = model.encode(texts, batch_size=batch, show_progress_bar=True,
                        normalize_embeddings=True)
    for r, v in zip(rows, vecs):
        r["vector"] = v.tolist()

    db = lancedb.connect(db_path)
    if "chunks" in db.table_names():
        db.drop_table("chunks")
    tbl = db.create_table("chunks", rows)
    tbl.create_fts_index("text", replace=True)
    # scalar indexes for release/series prefilter
    tbl.create_scalar_index("release", replace=True)
    tbl.create_scalar_index("series", replace=True)

    edges = [json.loads(l) for l in (parsed / "edges.jsonl").open()]
    if edges:
        if "edges" in db.table_names():
            db.drop_table("edges")
        db.create_table("edges", edges)
    typer.echo(f"index built at {db_path}: {len(rows)} chunks, {len(edges)} edges")


if __name__ == "__main__":
    app()
