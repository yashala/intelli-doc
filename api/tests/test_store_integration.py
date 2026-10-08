from pathlib import Path

import psycopg
import pytest
from reportlab.platypus import PageBreak

from intellidoc.embeddings import FastEmbedder
from intellidoc.ingest.service import ingest
from intellidoc.retrieval import RetrievalConfig, Retriever, lexical
from tests import pdf_factory as f

pytestmark = pytest.mark.integration


@pytest.fixture
def memo(tmp_path: Path) -> Path:
    return f.build(
        tmp_path / "memo.pdf",
        [
            f.para("Clinic Maintenance Memo", "Title"),
            f.para("Sterilization", "Heading2"),
            f.para("Autoclave unit AC-2 failed its weekly spore test on 2025-08-14."),
            PageBreak(),
            f.para("Imaging", "Heading2"),
            f.para("The ultrasound probe L12 was replaced by Medira Systems for 7,450 dollars."),
            PageBreak(),
            f.para("Audit", "Heading2"),
            f.para("The next equipment audit is scheduled for February 2026."),
        ],
    )


def test_reingesting_a_document_replaces_its_chunks(
    conn: psycopg.Connection, embedder: FastEmbedder, memo: Path
) -> None:
    first = ingest(conn, memo, embedder)
    second = ingest(conn, memo, embedder)
    assert first.doc_id == second.doc_id
    rows = conn.execute(
        "SELECT chunker, count(*) FROM chunks GROUP BY chunker ORDER BY chunker"
    ).fetchall()
    assert rows == [("fixed", first.chunks["fixed"]), ("structure", first.chunks["structure"])]


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_retrieval_returns_the_page_containing_the_fact(
    conn: psycopg.Connection, embedder: FastEmbedder, memo: Path, mode: str
) -> None:
    ingest(conn, memo, embedder)
    config = RetrievalConfig(chunker="structure", mode=mode, k=3)  # type: ignore[arg-type]
    results = Retriever(embedder).search(conn, "Who replaced the ultrasound probe?", config)
    assert results[0].pages == [2]
    assert "Medira" in results[0].text
    assert 0 < results[0].similarity <= 1


def test_lexical_search_matches_hyphenated_identifiers(
    conn: psycopg.Connection, embedder: FastEmbedder, memo: Path
) -> None:
    ingest(conn, memo, embedder)
    query = "What happened to AC-2?"
    results = lexical(conn, query, embedder.embed_query(query), "structure", 5)
    assert results and "AC-2" in results[0].text


def test_lexical_search_with_only_stopwords_returns_nothing(
    conn: psycopg.Connection, embedder: FastEmbedder, memo: Path
) -> None:
    ingest(conn, memo, embedder)
    assert lexical(conn, "what is the", embedder.embed_query("x"), "structure", 5) == []
