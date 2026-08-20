# Legal posture

This document states, plainly, where ClauseFinder's content comes from, how
it's used, and what is and isn't distributed in this repository. It's meant
to be verifiable, not aspirational — if anything here stops matching what
the code actually does, treat that as a bug and open an issue.

## Corpus provenance

ClauseFinder ingests 3GPP technical specifications exclusively via the
**official GSMA/3GPP markdown mirror** published on Hugging Face as part of
the GSMA Open Telco AI ecosystem (`GSMA/3GPP` dataset;
see `ingest/download_gsma.py`). No specification text is scraped from
3gpp.org, ETSI, or any other source — the GSMA mirror is the single
ingestion point, and it is GSMA's own redistribution of the specs for
exactly this kind of AI-tooling use.

3GPP specifications (TS = Technical Specification, TR = Technical Report)
are copyright the 3GPP Organizational Partners (ARIB, ATIS, CCSA, ETSI,
TSDSI, TTA, TTC). GSMA's mirror does not change that copyright; it changes
the format (PDF → markdown) and the access path.

## How specs are cited

Every chunk retrieved and shown to a user carries, at minimum:

- the spec number and type (e.g. `TS 38.331`),
- the release (e.g. `Rel-18`) and version (e.g. `V18.0.0`),
- the clause number and title,
- a deep link to the spec's official page on 3GPP's own DynaReport system
  (`https://www.3gpp.org/DynaReport/{spec}.htm`, built in
  `ingest/chunker.py`'s `_spec_url`).

Answers cite short, verbatim, attributed excerpts (the `[Ck]` tag +
quote-in-double-quotes convention enforced by
`agent/validators/citations.py`) — never a reproduction of a full clause,
section, or document. The citation validator's job is precisely to keep
quotes short, exact, and traceable back to a specific clause; see
`docs/architecture.md` for how that's enforced mechanically.

## No corpus redistribution in this repository

**This repo contains no spec documents or substantial spec text; eval-set rationales include only short (<15-word) attributed fragments used to justify gold clause labels, and test fixtures are synthetic.** `data/` (the downloaded corpus, the
parsed chunks, and the built LanceDB index) is git-ignored
(see `.gitignore`) and is never committed. What ships in the repo is:

- the **pipeline** that downloads GSMA's public mirror, chunks it, and
  builds an index (`ingest/`) — code, not content;
- **short quoted excerpts** that appear transiently in answers at query
  time, generated per-user, per-question, never stored as a static asset in
  the repo.

## The gated index dataset

The built LanceDB index (embeddings + chunked spec text + breadcrumbs) is a
derived artifact of the GSMA corpus, and it *is* spec text, chunked and
embedded — so it is **not** published as an open, anonymous-download asset.
When a pre-built index is published for the hosted demo (HF ZeroGPU Space)
to download at boot, it goes to a **gated Hugging Face dataset**, not a
public one: access is behind Hugging Face's gating flow, and the dataset
card carries the same GSMA/3GPP provenance statement as this document —
source dataset, license posture, and a pointer back to the official specs.
This mirrors GSMA's own distribution model (gated/tracked, not anonymous
open download) rather than loosening it.

## What ClauseFinder is not

ClauseFinder output is **not** an authoritative copy of any 3GPP
specification, and the app does not present it as one. Answers are a
retrieval-and-synthesis layer over the official text, with citations meant
to make every claim checkable — but checking means going to the cited
clause via the deep link and confirming, not trusting the synthesis
unread. This posture is stated in the app itself and in `NOTICE`.

## Takedown / provenance concerns

If you are a rights holder (3GPP Organizational Partner, GSMA, or anyone
with a good-faith concern about how spec content is sourced, cited, or
displayed here), contact **chinthave@gmail.com** with:

- the specific spec(s)/clause(s) or feature of the app in question,
- what you'd like changed (removal, correction, attribution fix), and
- how to verify the request.

Requests are handled promptly; this is a personal open-source project, not
a company, so response time is best-effort but genuine.
