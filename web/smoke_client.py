"""Smoke test: drive the running Gradio app with gradio_client.

Usage: uv run python web/app.py   (in one shell)
       uv run python web/smoke_client.py [url]
Asserts an Ask query returns a [Ck]-cited answer and sources, and an
Explain query resolves a clause by metadata.
"""
from __future__ import annotations

import re
import sys

from gradio_client import Client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"


def last_assistant(history) -> str:
    msgs = [m for m in history if m.get("role") == "assistant"]
    assert msgs, f"no assistant message in {history!r}"
    content = msgs[-1]["content"]
    return content if isinstance(content, str) else str(content)


def main() -> None:
    client = Client(URL)

    print("== Ask ==")
    history, sources, badges = client.predict(
        "What does the UE do when timer T310 expires?", "Rel-18", [],
        api_name="/run_ask")
    answer = last_assistant(history)
    print(answer[:400], "…\n")
    assert re.search(r"\[C\d+\]", answer), f"no [Ck] citation in answer: {answer[:200]}"
    assert "TS 38.331" in sources, f"sources panel missing spec: {sources[:200]}"
    assert "badge" in badges, f"no validator badges: {badges[:200]}"
    print("ask: cited answer + sources + badges OK")

    print("\n== Explain ==")
    history, sources, badges = client.predict(
        "TS 38.331 5.3.5.3", "Rel-18", [], api_name="/run_explain")
    text = last_assistant(history)
    print(text[:400], "…\n")
    assert re.search(r"\[C\d+\]", text), f"no [Ck] citation in explanation: {text[:200]}"
    assert "5.3.5" in sources, f"sources panel missing clause: {sources[:200]}"
    print("explain: cited explanation + sources OK")

    print("\n== Diff (graceful stub until agent/diff.py lands) ==")
    history, _, _ = client.predict(
        "How did conditional handover change?", "Rel-18", "Rel-18", [],
        api_name="/run_diff")
    print(last_assistant(history)[:200])

    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
