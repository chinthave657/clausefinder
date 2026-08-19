"""OpenAI-compatible shim exposing ClauseFinder's ask() as a chat model.

Lets gsma-labs/satellite (or any Inspect-AI-style eval harness that speaks
the OpenAI chat-completions wire format) point at the Nano+ClauseFinder RAG
config as if it were just another model endpoint — no satellite-side code
changes, only a `--model openai/clausefinder-local` + `base_url` pointed
here. See eval/satellite/README.md for the full design rationale and the
run recipe; this file is the implementation only.

Deliberately NOT streaming, NOT stateful across requests, and NOT wired to
mode=diff/explain — satellite's benchmarks are single-turn Q&A, which is
exactly agent.answer.ask()'s shape. One request in, one non-streamed
completion out; validator report is smuggled into the response for local
inspection but plain OpenAI clients will ignore the extra field.

Run: uv run uvicorn eval.satellite.shim:app --port 8091
(never started automatically; see README "run the shim" step)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.tools.retrieval import Retriever, load_acronyms

DB = Path(os.environ.get("CLAUSEFINDER_DB", "data/index"))
PARSED = Path(os.environ.get("CLAUSEFINDER_PARSED", "data/parsed"))
SHIM_MODEL_NAME = "clausefinder-local"

app = FastAPI(title="ClauseFinder OpenAI-compatible shim")

# Loaded lazily on first request, not at import time: importing this module
# (e.g. for tests) must not touch data/index or make network calls.
_retriever: Retriever | None = None
_acronyms: dict | None = None


def _state() -> tuple[Retriever, dict]:
    global _retriever, _acronyms
    if _retriever is None:
        _retriever = Retriever(DB)
        _acronyms = load_acronyms(PARSED)
    return _retriever, _acronyms


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = SHIM_MODEL_NAME
    messages: list[ChatMessage]
    temperature: float | None = None   # accepted, ignored — ask() runs temp 0 (design §3)
    max_tokens: int | None = None      # accepted, ignored — answer agent sets its own cap


def _last_user_message(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    raise HTTPException(422, "no user message in request")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": SHIM_MODEL_NAME}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": SHIM_MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
    from agent.answer import ask  # deferred: avoids importing the LLM client at module load

    question = _last_user_message(req.messages)
    retriever, acronyms = _state()
    answer, report, tagmap = ask(question, retriever, acronyms=acronyms)

    return {
        "id": f"clausefinder-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SHIM_MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        # Non-standard extension field — OpenAI-strict clients ignore unknown
        # top-level keys; useful for local debugging of satellite runs.
        "clausefinder_validator": {
            "passed": report.passed,
            "n_citations": len(tagmap),
        },
    }
