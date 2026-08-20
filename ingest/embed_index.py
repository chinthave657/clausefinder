"""Embed child chunks with OTel-Embedding-568M and build the LanceDB index.

Streaming + checkpointed: chunks are read lazily, embedded in small batches,
and appended to LanceDB immediately — peak RAM is one batch, not the corpus.
A checkpoint file records how many rows are committed; rerunning resumes
where the last run stopped (crash/OOM/restart safe).

No ANN index is created on purpose: at this corpus size, LanceDB brute-force
(flat) search is exact — the literature (Telco-RAG) measured HNSW losing
accuracy vs flat on 3GPP retrieval. FTS (tantivy BM25) rides the same table.
"""
from __future__ import annotations

import json
from itertools import islice
from pathlib import Path

import lancedb
import typer

MODEL_ID = "farbodtavakkoli/OTel-Embedding-568M"
EMBED_MAX_TOKENS = 1400  # OTel fine-tune length; inputs truncated by tokenizer

app = typer.Typer(add_completion=False)


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    kwargs = {}
    if device == "cuda":
        # fp16 halves VRAM (T4 14.5GB OOMs at fp32 batch 64 × 1400 tok);
        # retrieval quality unaffected. MPS/CPU stay fp32.
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True, device=device, **kwargs)
    model.max_seq_length = EMBED_MAX_TOKENS
    typer.echo(f"embedding on device: {device} ({'fp16' if device == 'cuda' else 'fp32'})")
    return model


def _commit(db, tbl, pending: list[dict], done: int, ckpt: Path) -> int:
    if tbl is None and "chunks" not in db.table_names():
        db.create_table("chunks", pending)
    else:
        db.open_table("chunks").add(pending)
    done += len(pending)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(str(done))
    return done


def _batches(path: Path, size: int, skip: int):
    with path.open() as f:
        it = (json.loads(l) for l in islice(f, skip, None))
        while chunk := list(islice(it, size)):
            yield chunk


@app.command()
def build(
    parsed: Path = typer.Option(Path("data/parsed"), help="chunker output dir"),
    db_path: Path = typer.Option(Path("data/index"), help="LanceDB dir"),
    batch: int = typer.Option(16, help="embed batch (peak-RAM knob)"),
    commit_every: int = typer.Option(512, help="rows per LanceDB append"),
    stop_after: int = typer.Option(0, help="exit 0 after N new rows (snapshot-upload loop)"),
    fresh: bool = typer.Option(False, help="drop table + checkpoint, start over"),
) -> None:
    # Checkpoint is scoped to the parsed dir: a delta run (different chunks
    # file, different ordering) must never resume against the full-run offset.
    ckpt = db_path / f".embed_checkpoint_{parsed.name}"
    total = sum(1 for _ in (parsed / "chunks.jsonl").open())
    if total == 0:
        raise SystemExit(f"{parsed}/chunks.jsonl is empty — run the chunker first")
    db = lancedb.connect(db_path)

    if fresh and "chunks" in db.table_names():
        db.drop_table("chunks")
        ckpt.unlink(missing_ok=True)
    done = int(ckpt.read_text()) if ckpt.exists() and "chunks" in db.table_names() else 0
    if done >= total:
        typer.echo(f"already complete ({done}/{total}); finalizing indexes only")
    else:
        typer.echo(f"embedding {total - done}/{total} chunks with {MODEL_ID} "
                   f"(resume at {done}) …")
        start_done = done
        model = load_model()
        tbl = db.open_table("chunks") if "chunks" in db.table_names() else None
        pending: list[dict] = []
        for rows in _batches(parsed / "chunks.jsonl", batch, done):
            texts = [r["breadcrumb"] + "\n" + r["text"] for r in rows]
            vecs = model.encode(texts, batch_size=batch,
                                normalize_embeddings=True)
            for r, v in zip(rows, vecs):
                r["vector"] = v.tolist()
            pending.extend(rows)
            if len(pending) >= commit_every:
                done = _commit(db, tbl := (tbl if tbl is not None else None), pending, done, ckpt)
                tbl = db.open_table("chunks")
                typer.echo(f"  committed {done}/{total}")
                pending = []
                if stop_after and done - start_done >= stop_after:
                    typer.echo(f"stop-after {stop_after} reached at {done}/{total}")
                    raise typer.Exit(0)
        if pending:
            done = _commit(db, tbl, pending, done, ckpt)

    tbl = db.open_table("chunks")
    typer.echo("building FTS + scalar indexes …")
    tbl.create_fts_index("text", replace=True)
    tbl.create_scalar_index("release", replace=True)
    tbl.create_scalar_index("series", replace=True)

    edges = [json.loads(l) for l in (parsed / "edges.jsonl").open()]
    if edges:
        if "edges" in db.table_names():
            db.drop_table("edges")
        db.create_table("edges", edges)
    typer.echo(f"index built at {db_path}: {tbl.count_rows()} chunks, {len(edges)} edges")


if __name__ == "__main__":
    app()
