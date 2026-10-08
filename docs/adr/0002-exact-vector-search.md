# ADR 0002: Exact vector search, no ANN index

Status: accepted (2026-10-08)

## Context
pgvector supports HNSW and IVFFlat indexes. With a `WHERE chunker = ...` filter, an ANN index
scans approximate neighbours first and filters afterwards, which can return fewer than k rows
unless iterative scans are tuned.

## Decision
No ANN index. The benchmark corpus is small (chunk counts per strategy are recorded in each
run's `environment.json`), so an exact scan is cheap, and it removes approximation as a variable
when comparing retrieval configurations. Measured retrieval latency is in `docs/results/`.

## Consequences
- Retrieval latency numbers are for exact search on a small corpus and do not predict
  behaviour at scale.
- At tens of thousands of chunks an HNSW index (with `hnsw.iterative_scan` for filtered
  queries) would be needed, and recall against exact search should then be measured.
