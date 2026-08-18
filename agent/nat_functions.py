"""NeMo Agent Toolkit (nvidia-nat) function registrations.

Thin orchestration only (design §"NAT"): logic stays in the plain-Python
retrieval/answer/rails modules; NAT wraps them as workflow functions so
``nat run --config_file agent/configs/ask.yml --input "..."`` works.
Discovered via the ``nat.plugins`` entry point declared in pyproject.toml.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class ClauseFinderAskConfig(FunctionBaseConfig, name="clausefinder_ask"):
    """Clause-cited 3GPP answer over the local LanceDB index, wrapped in
    guardrails (topic/injection input rails, config-softening +
    citation-required output rails)."""
    db_path: str = Field(default="data/index", description="LanceDB index dir")
    parsed_dir: str = Field(default="data/parsed", description="chunker output dir (acronyms.json)")
    release: str | None = Field(default=None, description="soft release filter, e.g. Rel-18")
    guardrails: bool = Field(default=True, description="apply NeMo Guardrails input/output rails")


class ClauseFinderSearchConfig(FunctionBaseConfig, name="clausefinder_search"):
    """Retrieval only — inspect what the index returns (no LLM)."""
    db_path: str = Field(default="data/index", description="LanceDB index dir")
    release: str | None = Field(default=None, description="soft release filter")
    top: int = Field(default=8, description="results after parent expansion")


def _build_retriever(db_path: str):
    from agent.tools.retrieval import Retriever
    return Retriever(Path(db_path))


@register_function(config_type=ClauseFinderAskConfig)
async def clausefinder_ask(config: ClauseFinderAskConfig, _builder: Builder):
    from agent.answer import ask
    from agent.rails_runtime import wrap_ask
    from agent.tools.retrieval import load_acronyms

    retriever = _build_retriever(config.db_path)
    acronyms = load_acronyms(Path(config.parsed_dir))

    def _sync(question: str) -> str:
        if config.guardrails:
            answer, report, tagmap = wrap_ask(question, retriever,
                                              release=config.release, acronyms=acronyms)
        else:
            answer, report, tagmap = ask(question, retriever,
                                         release=config.release, acronyms=acronyms)
        lines = [answer]
        if tagmap:
            lines.append("\nSources:")
            lines += [f"  {t} {c['spec']} §{c['clause']} {c['clause_title']} ({c['release']})"
                      for t, c in tagmap.items()]
            lines.append(f"Citation validation: {'passed' if report.passed else 'FAILED'}")
        return "\n".join(lines)

    async def _respond(question: str) -> str:
        return await asyncio.to_thread(_sync, question)

    yield FunctionInfo.from_fn(
        _respond,
        description="Answer a telecom-standards question with clause-cited, "
                    "guardrailed output grounded in the local 3GPP index.")


@register_function(config_type=ClauseFinderSearchConfig)
async def clausefinder_search(config: ClauseFinderSearchConfig, _builder: Builder):
    retriever = _build_retriever(config.db_path)

    def _sync(query: str) -> str:
        hits = retriever.search(query, release=config.release, top=config.top)
        return "\n".join(
            f"{h.tag} {h.chunk['spec']} §{h.chunk['clause']} {h.chunk['clause_title']} "
            f"({h.chunk['release']}) score={h.score:.4f}"
            for h in hits) or "no hits"

    async def _respond(query: str) -> str:
        return await asyncio.to_thread(_sync, query)

    yield FunctionInfo.from_fn(
        _respond,
        description="Hybrid (vector+FTS, weighted RRF) retrieval over the local "
                    "3GPP clause index; returns ranked clause hits, no LLM.")
