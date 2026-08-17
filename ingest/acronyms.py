"""Build the acronym/definition table from each spec's own §3 (Definitions and
Abbreviations). Applied at query time as APPEND-not-substitute enhancement —
the single biggest measured lever in Telco-RAG (+6pp on lexicon questions).
TR 21.905 (3GPP Vocabulary) merges in when present in the corpus.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# GSMA markdown renders §3.2 as tables: "| 5GC | 5G Core Network |"
ROW_RE = re.compile(r"^\|\s*([A-Z][A-Za-z0-9\-]{1,12})\s*\|\s*([^|]{3,90})\|")


def extract_from_chunks(parsed_dir: Path) -> dict[str, str]:
    acronyms: dict[str, str] = {}
    for line in (parsed_dir / "chunks.jsonl").open():
        c = json.loads(line)
        if "abbreviation" not in c["clause_title"].lower():
            continue
        for ln in c["text"].splitlines():
            m = ROW_RE.match(ln.strip())
            if m and m.group(1) not in acronyms and "---" not in m.group(2):
                acronyms[m.group(1)] = m.group(2).strip()
    return acronyms


if __name__ == "__main__":
    parsed = Path("data/parsed")
    table = extract_from_chunks(parsed)
    (parsed / "acronyms.json").write_text(json.dumps(table, indent=0, sort_keys=True))
    print(f"{len(table)} acronyms extracted")
