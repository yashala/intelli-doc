"""Parse a PDF once, then store chunks for each requested chunking strategy."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from psycopg import Connection
from psycopg.types.json import Jsonb

from intellidoc.embeddings import Embedder
from intellidoc.ingest.chunking import fixed_chunks, search_text, structure_chunks
from intellidoc.ingest.parsing import parse_pdf
from intellidoc.models import Chunk, ParsedDocument

log = logging.getLogger(__name__)

CHUNKERS: dict[str, Callable[[str, ParsedDocument], list[Chunk]]] = {
    "fixed": fixed_chunks,
    "structure": structure_chunks,
}


@dataclass
class IngestResult:
    doc_id: str
    title: str
    pages: int
    ocr_pages: int
    chunks: dict[str, int]
    timings_ms: dict[str, float] = field(default_factory=dict)


def document_id(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def ingest(
    conn: Connection,
    path: Path,
    embedder: Embedder,
    chunkers: list[str] | None = None,
    filename: str | None = None,
) -> IngestResult:
    chunkers = chunkers or list(CHUNKERS)
    doc_id = document_id(path)
    timings: dict[str, float] = {}

    start = time.perf_counter()
    parsed = parse_pdf(path)
    timings["parse"] = (time.perf_counter() - start) * 1000

    with conn.transaction():
        conn.execute(
            """INSERT INTO documents (id, filename, title, page_count) VALUES (%s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title""",
            (doc_id, filename or path.name, parsed.title, len(parsed.pages)),
        )
        conn.execute("DELETE FROM pages WHERE doc_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO pages VALUES (%s, %s, %s, %s, %s)",
                [
                    (doc_id, p.page, p.method, p.native_chars, p.ocr_confidence)
                    for p in parsed.pages
                ],
            )

    counts = {}
    for name in chunkers:
        chunks = CHUNKERS[name](doc_id, parsed)
        texts = [search_text(c) for c in chunks]
        start = time.perf_counter()
        vectors = embedder.embed_passages(texts)
        timings[f"embed_{name}"] = (time.perf_counter() - start) * 1000
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s AND chunker = %s", (doc_id, name))
            cur.executemany(
                """INSERT INTO chunks (doc_id, chunker, ordinal, pages, kind, section, text,
                                       search_text, metadata, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (doc_id, name, i, c.pages, c.kind, c.section, c.text, t, Jsonb(c.metadata), v)
                    for i, (c, t, v) in enumerate(zip(chunks, texts, vectors, strict=True))
                ],
            )
        counts[name] = len(chunks)

    ocr_pages = sum(1 for p in parsed.pages if p.method == "ocr")
    log.info(
        "ingested document",
        extra={
            "doc_id": doc_id,
            "pages": len(parsed.pages),
            "ocr_pages": ocr_pages,
            "chunks": counts,
            "timings_ms": timings,
        },
    )
    return IngestResult(doc_id, parsed.title, len(parsed.pages), ocr_pages, counts, timings)
