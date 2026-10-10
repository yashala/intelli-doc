"""Chunking strategies compared in the benchmark.

`fixed` reproduces the original app's 1000/200 character window, but keeps page provenance
so citations can be scored. `structure` follows the parsed layout: a chunk never mixes
sections or pages, tables and figure text stay whole, and each chunk carries a title/section header
that is prepended when embedding and indexing.
"""

from __future__ import annotations

from collections.abc import Iterable

from intellidoc.ingest import injection
from intellidoc.models import Chunk, Element, ParsedDocument


def fixed_chunks(
    doc_id: str, parsed: ParsedDocument, size: int = 1000, overlap: int = 200
) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = ""
    spans: list[tuple[int, int, int]] = []  # (start, end, page)
    for element in parsed.elements:
        start = len(text)
        text += element.text + "\n"
        spans.append((start, len(text), element.page))

    chunks = []
    for start in range(0, max(len(text) - overlap, 1), size - overlap):
        end = min(start + size, len(text))
        pages = sorted({page for s, e, page in spans if s < end and e > start})
        body = text[start:end].strip()
        if body:
            chunks.append(_finish(Chunk(doc_id, pages, "text", body), parsed, [], ""))
    return chunks


def structure_chunks(doc_id: str, parsed: ParsedDocument, max_chars: int = 1200) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""
    buffer: list[Element] = []

    def flush() -> None:
        if buffer:
            body = "\n".join(e.text for e in buffer)
            pages = sorted({e.page for e in buffer})
            chunks.append(_finish(Chunk(doc_id, pages, "text", body), parsed, buffer, section))
            buffer.clear()

    for element in parsed.elements:
        if element.kind == "heading":
            flush()
            section = element.text
        elif element.kind in ("table", "image_text"):
            flush()
            for piece in (
                _split_table(element.text, max_chars) if element.kind == "table" else [element.text]
            ):
                chunk = Chunk(doc_id, [element.page], element.kind, piece)
                chunks.append(_finish(chunk, parsed, [element], section))
        else:
            too_long = sum(len(e.text) for e in buffer) + len(element.text) > max_chars
            if buffer and (too_long or buffer[-1].page != element.page):
                flush()
            buffer.append(element)
    flush()
    return chunks


def _split_table(markdown: str, max_chars: int) -> Iterable[str]:
    """Split a large Markdown table by rows, repeating the header in each piece."""
    lines = markdown.split("\n")
    header, rows = lines[:2], lines[2:]
    piece: list[str] = []
    for row in rows:
        if piece and len("\n".join(header + piece + [row])) > max_chars:
            yield "\n".join(header + piece)
            piece = []
        piece.append(row)
    yield "\n".join(header + piece)


def _finish(chunk: Chunk, parsed: ParsedDocument, sources: list[Element], section: str) -> Chunk:
    chunk.section = section
    methods = sorted({e.method for e in sources}) or sorted(
        {p.method for p in parsed.pages if p.page in chunk.pages}
    )
    confidences = [e.ocr_confidence for e in sources if e.ocr_confidence is not None]
    flags = injection.scan(chunk.text)
    chunk.metadata = {
        "title": parsed.title,
        "section": section,
        "methods": methods,
        "min_ocr_confidence": round(min(confidences), 1) if confidences else None,
        "injection_flags": flags,
    }
    return chunk


def search_text(chunk: Chunk) -> str:
    """Text used for embeddings and full-text search: a context header plus the chunk body."""
    title = str(chunk.metadata.get("title", ""))
    header = " > ".join(part for part in (title, chunk.section) if part)
    return f"{header}\n{chunk.text}" if header else chunk.text
