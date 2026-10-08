# ADR 0001: PostgreSQL with pgvector as the only datastore

Status: accepted (2026-10-08)

## Context
The service stores documents, page metadata and chunks, and needs dense similarity search,
keyword search, and the ability to compare chunking strategies on the same parse.

## Decision
One PostgreSQL 16 database with the pgvector extension. Chunks for every chunking strategy live
in one `chunks` table keyed by `chunker`. Dense search uses pgvector cosine distance; lexical
search uses a generated `tsvector` column with a GIN index. Hybrid fusion happens in Python.

## Alternatives considered
- Chroma or FAISS next to Postgres: a second system to run and keep consistent, and no
  full-text search or joins in the same query.
- Pinecone or another hosted vector DB: paid, remote, and harder to reproduce from a clean checkout.
- LangChain's `PGVector` store: uses its own schema (`langchain_pg_embedding`) with metadata in
  JSONB, which makes the per-chunker comparison and the full-text column awkward.

## Consequences
- Everything needed for the benchmark starts with `docker compose up`.
- Postgres full-text ranking (`ts_rank_cd`) is not BM25; results labelled "lexical" are Postgres
  ranking, not a BM25 implementation.
