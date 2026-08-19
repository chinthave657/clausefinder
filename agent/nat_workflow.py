"""NAT workflow layer: diff + router-dispatched functions, and an in-process
Python-API runner (design §3.4 -- "the ask flow as a NAT workflow runnable
via Python API in-process").

``agent/nat_functions.py`` already registers ``clausefinder_ask`` /
``clausefinder_search`` (ask-mode + bare retrieval). This module adds the two
pieces that were still missing:

  - ``clausefinder_diff``  -- diff-mode wrapped as a NAT function, mirroring
    ``agent/configs/diff.yml``.
  - ``clausefinder_route`` -- the unified-chat entry point: runs
    ``agent.router.route()`` and dispatches to ask / diff / explain. UI tabs
    call the mode-specific function directly and skip this (design: "UI tab
    is the mode"); this is only for a single free-text chat surface.

Registration is via the same ``@register_function`` decorator NAT uses
everywhere -- it populates the process-wide ``GlobalTypeRegistry`` as a
module-level side effect of importing this file, which is what makes the
in-process ``run()``/``arun()`` helpers below work without a ``nat.plugins``
entry point declared in pyproject.toml: importing ``agent.nat_workflow``
*is* the plugin discovery step for this repo's own functions. (Third-party
NAT functions installed as separate packages still need their own entry
points; ours doesn't need to leave this repo.)
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

# Re-registers clausefinder_ask / clausefinder_search as a side effect.
import agent.nat_functions  # noqa: F401,E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def _format(answer: str, report, tagmap: dict[str, dict]) -> str:
    lines = [answer]
    if tagmap:
        lines.append("\nSources:")
        lines += [f"  {t} {c['spec']} §{c['clause']} {c['clause_title']} ({c['release']})"
                  for t, c in tagmap.items()]
        lines.append(f"Citation validation: {'passed' if report.passed else 'FAILED'}")
    return "\n".join(lines)


def _extract_releases(text: str, fallback_a: str, fallback_b: str) -> tuple[str, str]:
    """Pull the first two distinct 'Rel-NN' tokens out of the query; fall
    back to config defaults when the query doesn't name two releases (e.g.
    the router dispatched here on the LLM tier, not the regex tier)."""
    from agent.router import RELEASE_RE

    seen: list[str] = []
    for n in RELEASE_RE.findall(text):
        tag = f"Rel-{n}"
        if tag not in seen:
            seen.append(tag)
    if len(seen) >= 2:
        return seen[0], seen[1]
    return fallback_a, fallback_b


class ClauseFinderDiffConfig(FunctionBaseConfig, name="clausefinder_diff"):
    """Release-to-release clause diff wrapped as a NAT function (design §2,
    diff agent). Release tokens are parsed from the query when present
    (``_extract_releases``); ``release_a``/``release_b`` are the fallback."""
    db_path: str = Field(default="data/index", description="LanceDB index dir")
    parsed_dir: str = Field(default="data/parsed", description="chunker output dir (acronyms.json)")
    release_a: str = Field(default="Rel-17", description="fallback old release")
    release_b: str = Field(default="Rel-18", description="fallback new release")


@register_function(config_type=ClauseFinderDiffConfig)
async def clausefinder_diff(config: ClauseFinderDiffConfig, _builder: Builder):
    from agent.diff import diff_releases
    from agent.tools.retrieval import Retriever, load_acronyms

    retriever = Retriever(Path(config.db_path))
    acronyms = load_acronyms(Path(config.parsed_dir))

    def _sync(question: str) -> str:
        release_a, release_b = _extract_releases(question, config.release_a, config.release_b)
        answer, report, tagmap, _rendered = diff_releases(
            question, retriever, release_a, release_b, acronyms=acronyms)
        return _format(answer, report, tagmap)

    async def _respond(question: str) -> str:
        return await asyncio.to_thread(_sync, question)

    yield FunctionInfo.from_fn(
        _respond,
        description="Release-to-release clause diff (Rel-A vs Rel-B) with cited synthesis "
                    "over the local 3GPP index; release tokens are parsed from the query "
                    "when present, else config defaults.")


class ClauseFinderRouteConfig(FunctionBaseConfig, name="clausefinder_route"):
    """Unified-chat entry point (design §3, ROUTER): rule-first regex tier,
    Nano structured-output fallback only on ambiguity, dispatches to
    ask/diff/explain. UI-tab traffic should call the mode-specific function
    directly instead of this one."""
    db_path: str = Field(default="data/index", description="LanceDB index dir")
    parsed_dir: str = Field(default="data/parsed", description="chunker output dir (acronyms.json)")
    release_a: str = Field(default="Rel-17", description="diff fallback old release")
    release_b: str = Field(default="Rel-18", description="diff fallback new release")
    guardrails: bool = Field(default=True, description="apply NeMo Guardrails on the ask path")


@register_function(config_type=ClauseFinderRouteConfig)
async def clausefinder_route(config: ClauseFinderRouteConfig, _builder: Builder):
    from agent.answer import ask
    from agent.diff import diff_releases
    from agent.explain import explain
    from agent.router import Mode, route
    from agent.tools.retrieval import Retriever, load_acronyms

    retriever = Retriever(Path(config.db_path))
    acronyms = load_acronyms(Path(config.parsed_dir))

    def _sync(question: str) -> str:
        mode = route(question)
        if mode is Mode.diff:
            release_a, release_b = _extract_releases(question, config.release_a, config.release_b)
            answer, report, tagmap, _rendered = diff_releases(
                question, retriever, release_a, release_b, acronyms=acronyms)
        elif mode is Mode.explain:
            answer, report, tagmap = explain(question, retriever)
        elif config.guardrails:
            from agent.rails_runtime import wrap_ask
            answer, report, tagmap = wrap_ask(question, retriever, acronyms=acronyms)
        else:
            answer, report, tagmap = ask(question, retriever, acronyms=acronyms)
        return _format(answer, report, tagmap)

    async def _respond(question: str) -> str:
        return await asyncio.to_thread(_sync, question)

    yield FunctionInfo.from_fn(
        _respond,
        description="Unified-chat entry point: rule-first router (regex tier, LLM "
                    "fallback on ambiguity) picks ask/diff/explain and dispatches to "
                    "the matching pipeline.")


# --------------------------------------------------------------------------
# In-process Python-API runner. No `nat run` CLI, no nat.plugins entry point:
# importing this module registers the function types above directly into the
# GlobalTypeRegistry, so nat.runtime.loader.load_config() can validate a YAML
# config that references them right away.
# --------------------------------------------------------------------------

def load_config(config_file: str | Path):
    from nat.runtime.loader import load_config as _load_config
    return _load_config(config_file)


async def arun_workflow(config_file: str | Path, message: str) -> str:
    """Build + run a NAT workflow config in-process; returns the string result."""
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.runtime.session import SessionManager

    config = load_config(config_file)
    async with WorkflowBuilder.from_config(config=config) as builder:
        session = await SessionManager.create(config=config, shared_builder=builder)
        try:
            async with session.run(message) as runner:
                return await runner.result(str)
        finally:
            await session.shutdown()


def run_workflow(config_file: str | Path, message: str) -> str:
    """Sync wrapper around arun_workflow -- CLI / smoke-test convenience."""
    return asyncio.run(arun_workflow(config_file, message))


def run_ask(question: str) -> str:
    return run_workflow(CONFIGS_DIR / "ask.yml", question)


def run_diff(question: str) -> str:
    return run_workflow(CONFIGS_DIR / "diff.yml", question)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What does TS 38.331 say about RRC reconfiguration?"
    print(run_ask(q))
