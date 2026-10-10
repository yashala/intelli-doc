from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from reportlab.platypus import PageBreak

from intellidoc.db import apply_schema
from intellidoc.settings import get_settings
from tests import pdf_factory as f

pytestmark = pytest.mark.integration


@pytest.fixture
def client(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_MODE", "offline")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    with psycopg.connect(database_url) as conn:
        apply_schema(conn)  # a fresh database has no tables until something creates them
        conn.execute("TRUNCATE documents CASCADE")
    from intellidoc.app import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture
def memo_pdf(tmp_path: Path) -> bytes:
    path = f.build(
        tmp_path / "memo.pdf",
        [
            f.para("Clinic Maintenance Memo", "Title"),
            f.para("Sterilization", "Heading2"),
            f.para("Autoclave unit AC-2 failed its weekly spore test on 2025-08-14."),
            PageBreak(),
            f.para("Imaging", "Heading2"),
            f.para("The ultrasound probe L12 was replaced by Medira Systems for 7,450 dollars."),
        ],
    )
    return path.read_bytes()


def test_health_reports_database_and_mode(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "database": True,
        "documents": 0,
        "llm_mode": "offline",
        "answer_model": "offline-extractive",
    }


def test_upload_rejects_non_pdf(client: TestClient) -> None:
    response = client.post("/documents", files={"file": ("notes.pdf", b"hello", "application/pdf")})
    assert response.status_code == 415


def test_upload_rejects_files_over_the_limit(client: TestClient) -> None:
    big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)
    response = client.post("/documents", files={"file": ("big.pdf", big, "application/pdf")})
    assert response.status_code == 413


def test_upload_then_ask_returns_answer_cited_to_the_right_page(
    client: TestClient, memo_pdf: bytes
) -> None:
    upload = client.post("/documents", files={"file": ("memo.pdf", memo_pdf, "application/pdf")})
    assert upload.status_code == 201
    doc_id = upload.json()["doc_id"]
    assert upload.json()["pages"] == 2

    answer = client.post("/ask", json={"question": "Who replaced the ultrasound probe L12?"})
    assert answer.status_code == 200
    body = answer.json()
    assert body["refused"] is False
    assert "Medira" in body["answer"]
    assert [(c["doc_id"], c["pages"]) for c in body["citations"]] == [(doc_id, [2])]

    image = client.get(f"/documents/{doc_id}/pages/2.png")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert client.get(f"/documents/{doc_id}/pages/9.png").status_code == 404


def test_question_not_covered_by_documents_is_refused(client: TestClient, memo_pdf: bytes) -> None:
    client.post("/documents", files={"file": ("memo.pdf", memo_pdf, "application/pdf")})
    body = client.post("/ask", json={"question": "What is the capital of Australia?"}).json()
    assert body["refused"] is True
    assert body["refusal_reason"] == "insufficient_evidence"
    assert body["citations"] == []


def test_ask_validates_input(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "mode": "magic"}).status_code == 422


def test_ask_can_be_limited_to_selected_documents(client: TestClient, memo_pdf: bytes) -> None:
    doc_id = client.post(
        "/documents", files={"file": ("memo.pdf", memo_pdf, "application/pdf")}
    ).json()["doc_id"]
    body = client.post(
        "/ask", json={"question": "Who replaced the ultrasound probe L12?", "doc_ids": ["nope"]}
    ).json()
    assert body["refused"] is True
    body = client.post(
        "/ask", json={"question": "Who replaced the ultrasound probe L12?", "doc_ids": [doc_id]}
    ).json()
    assert body["refused"] is False
