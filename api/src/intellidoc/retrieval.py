"""Dense, lexical and hybrid retrieval over the chunks table.

Every result carries `similarity`: the cosine similarity between the query and chunk
embeddings, computed the same way whichever mode found the chunk. The answer workflow uses
it as a retrieval-confidence floor, so the floor means the same thing across modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

from psycopg import Connection
from psycopg.rows import dict_row

from intellidoc.embeddings import Embedder, Vector

Mode = Literal["dense", "hybrid", "hybrid_rerank"]
RRF_K = 60


@dataclass(frozen=True)
class RetrievalConfig:
    chunker: str = "structure"
    mode: Mode = "hybrid"
    k: int = 5
    candidates: int = 20

    @property
    def name(self) -> str:
        return f"{self.chunker}-{self.mode}"


@dataclass
class Retrieved:
    chunk_id: int
    doc_id: str
    title: str
    filename: str
    pages: list[int]
    kind: str
    section: str
    text: str
    metadata: dict[str, Any]
    similarity: float
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)


_SELECT = """
SELECT c.id AS chunk_id, c.doc_id, d.title, d.filename, c.pages, c.kind, c.section, c.text,
       c.metadata, 1 - (c.embedding <=> %(q)s) AS similarity
FROM chunks c JOIN documents d ON d.id = c.doc_id
"""

_DENSE = (
    _SELECT
    + """
WHERE c.chunker = %(chunker)s
  AND (%(docs)s::text[] IS NULL OR c.doc_id = ANY(%(docs)s::text[]))
ORDER BY c.embedding <=> %(q)s
LIMIT %(limit)s
"""
)

# OR together the query's lexemes (plainto_tsquery would AND them, which is too strict for
# questions). The lexemes come from to_tsvector and are already normalised, so they are cast
# straight to tsquery rather than passed through to_tsquery's dictionary a second time.
_LEXICAL = (
    _SELECT
    + """,
LATERAL (
    SELECT array_to_string(
        ARRAY(SELECT quote_literal(l) FROM unnest(tsvector_to_array(
            to_tsvector('english', %(text)s))) AS l), ' | ') AS terms
) t
WHERE c.chunker = %(chunker)s AND t.terms <> ''
  AND (%(docs)s::text[] IS NULL OR c.doc_id = ANY(%(docs)s::text[]))
  AND c.tsv @@ t.terms::tsquery
ORDER BY ts_rank_cd(c.tsv, t.terms::tsquery) DESC, c.id
LIMIT %(limit)s
"""
)


def _rows(conn: Connection, sql: str, params: dict[str, Any]) -> list[Retrieved]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return [Retrieved(**row) for row in cur.fetchall()]


def dense(
    conn: Connection,
    query_vector: Vector,
    chunker: str,
    limit: int,
    doc_ids: list[str] | None = None,
) -> list[Retrieved]:
    params = {"q": query_vector, "chunker": chunker, "limit": limit, "docs": doc_ids}
    return _rows(conn, _DENSE, params)


def lexical(
    conn: Connection,
    text: str,
    query_vector: Vector,
    chunker: str,
    limit: int,
    doc_ids: list[str] | None = None,
) -> list[Retrieved]:
    params = {"q": query_vector, "text": text, "chunker": chunker, "limit": limit, "docs": doc_ids}
    return _rows(conn, _LEXICAL, params)


def reciprocal_rank_fusion(ranked: dict[str, list[Retrieved]], k: int = RRF_K) -> list[Retrieved]:
    """Combine ranked lists: score = sum over lists of 1 / (k + rank)."""
    merged: dict[int, Retrieved] = {}
    for source, results in ranked.items():
        for rank, item in enumerate(results, start=1):
            entry = merged.setdefault(item.chunk_id, item)
            entry.ranks[source] = rank
            entry.score += 1.0 / (k + rank)
    return sorted(merged.values(), key=lambda r: (-r.score, r.chunk_id))


class Reranker:
    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        self.name = model

    @cached_property
    def _model(self):  # type: ignore[no-untyped-def]
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(self.name)

    def rerank(self, query: str, results: list[Retrieved]) -> list[Retrieved]:
        scores = list(self._model.rerank(query, [r.text for r in results]))
        for result, score in zip(results, scores, strict=True):
            result.score = float(score)
        return sorted(results, key=lambda r: (-r.score, r.chunk_id))


class Retriever:
    def __init__(self, embedder: Embedder, reranker: Reranker | None = None) -> None:
        self.embedder = embedder
        self.reranker = reranker

    def search(
        self,
        conn: Connection,
        query: str,
        config: RetrievalConfig,
        doc_ids: list[str] | None = None,
    ) -> list[Retrieved]:
        """Search all documents, or only `doc_ids` when given."""
        vector = self.embedder.embed_query(query)
        if config.mode == "dense":
            results = dense(conn, vector, config.chunker, config.k, doc_ids)
            for r in results:
                r.score = r.similarity
            return results

        fused = reciprocal_rank_fusion(
            {
                "dense": dense(conn, vector, config.chunker, config.candidates, doc_ids),
                "lexical": lexical(conn, query, vector, config.chunker, config.candidates, doc_ids),
            }
        )
        if config.mode == "hybrid_rerank":
            if self.reranker is None:
                raise ValueError("hybrid_rerank requires a reranker")
            fused = self.reranker.rerank(query, fused[: config.candidates])
        return fused[: config.k]
