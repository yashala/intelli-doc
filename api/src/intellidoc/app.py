from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

import pypdfium2 as pdfium
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from intellidoc import logging as json_logging
from intellidoc.answer.graph import AnswerWorkflow
from intellidoc.answer.models import AnswerModel, ChatAnswerModel, ExtractiveModel, make_chat
from intellidoc.db import create_pool
from intellidoc.embeddings import FastEmbedder
from intellidoc.ingest.service import ingest
from intellidoc.retrieval import Mode, Reranker, RetrievalConfig, Retriever
from intellidoc.settings import Settings, get_settings

log = logging.getLogger("intellidoc.api")


class Services:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pool: ConnectionPool = create_pool(settings.database_url)
        self.embedder = FastEmbedder(settings.embedding_model)
        self.retriever = Retriever(self.embedder, Reranker(settings.reranker_model))
        self.model = build_model(settings)
        self.workflows: dict[RetrievalConfig, AnswerWorkflow] = {}

    def workflow(self, config: RetrievalConfig) -> AnswerWorkflow:
        if config not in self.workflows:
            self.workflows[config] = AnswerWorkflow(
                self.pool, self.retriever, self.model, config, self.settings.min_similarity
            )
        return self.workflows[config]


def build_model(settings: Settings) -> AnswerModel:
    if settings.llm_mode == "offline":
        return ExtractiveModel()
    chat = make_chat(settings.llm_mode, settings.answer_model, settings.api_key(settings.llm_mode))
    return ChatAnswerModel(chat, settings.answer_model)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    json_logging.configure(settings.log_level)
    (settings.data_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "pages").mkdir(parents=True, exist_ok=True)
    app.state.services = Services(settings)
    log.info("started", extra={"llm_mode": settings.llm_mode})
    yield
    app.state.services.pool.close()


app = FastAPI(title="IntelliDoc", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def services(request: Request) -> Services:
    svc: Services = request.app.state.services
    return svc


Svc = Annotated[Services, Depends(services)]


@app.middleware("http")
async def access_log(request: Request, call_next: Any) -> Any:
    start = time.perf_counter()
    response = await call_next(request)
    log.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "ms": round((time.perf_counter() - start) * 1000, 1),
        },
    )
    return response


@app.get("/health")
def health(svc: Svc) -> dict[str, Any]:
    try:
        with svc.pool.connection() as conn:
            row = conn.execute("SELECT count(*) FROM documents").fetchone()
        db_ok, documents = True, int(row[0]) if row else 0
    except Exception:
        log.exception("health check failed")
        db_ok, documents = False, None
    return {
        "status": "ok" if db_ok else "degraded",
        "database": db_ok,
        "documents": documents,
        "llm_mode": svc.settings.llm_mode,
        "answer_model": svc.model.name,
    }


@app.get("/documents")
def list_documents(svc: Svc) -> list[dict[str, Any]]:
    with svc.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT d.id, d.filename, d.title, d.page_count, d.created_at,
                      count(p.*) FILTER (WHERE p.method = 'ocr') AS ocr_pages
               FROM documents d LEFT JOIN pages p ON p.doc_id = d.id
               GROUP BY d.id ORDER BY d.created_at DESC"""
        )
        return cur.fetchall()


@app.post("/documents", status_code=201)
def upload_document(file: UploadFile, svc: Svc) -> dict[str, Any]:
    limit = svc.settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File exceeds {svc.settings.max_upload_mb} MB")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(415, "Only PDF files are supported")
    digest = hashlib.sha256(data).hexdigest()[:16]
    path = svc.settings.data_dir / "pdfs" / f"{digest}.pdf"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
    Path(tmp.name).replace(path)
    try:
        with svc.pool.connection() as conn:
            result = ingest(conn, path, svc.embedder, filename=file.filename or path.name)
    except Exception as exc:
        log.exception("ingest failed", extra={"doc_id": digest})
        raise HTTPException(422, "The PDF could not be parsed") from exc
    return asdict(result)


@app.get("/documents/{doc_id}/pages/{page}.png")
def page_image(doc_id: str, page: int, svc: Svc) -> FileResponse:
    if not doc_id.isalnum():
        raise HTTPException(404)
    pdf_path = svc.settings.data_dir / "pdfs" / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "Document not found")
    out = svc.settings.data_dir / "pages" / f"{doc_id}-{page}.png"
    if not out.exists():
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            if not 1 <= page <= len(pdf):
                raise HTTPException(404, "Page not found")
            pdf[page - 1].render(scale=1.5).to_pil().save(out)
        finally:
            pdf.close()
    return FileResponse(out, media_type="image/png")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    chunker: Literal["fixed", "structure"] | None = None
    mode: Mode | None = None


class CitationOut(BaseModel):
    label: str
    doc_id: str
    title: str
    filename: str
    pages: list[int]
    text: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[CitationOut]
    config: str
    model: str
    prompt_version: str
    timings_ms: dict[str, float]
    usage: dict[str, int]


@app.post("/ask")
def ask(body: AskRequest, svc: Svc) -> AskResponse:
    config = RetrievalConfig(
        chunker=body.chunker or svc.settings.default_chunker,
        mode=body.mode or svc.settings.default_mode,
    )
    result = svc.workflow(config).run(body.question.strip())
    log.info(
        "answered",
        extra={
            "config": result.config,
            "refused": result.refused,
            "refusal_reason": result.refusal_reason,
            "citations": len(result.citations),
            "flagged_sources": result.flagged_sources,
            "timings_ms": result.timings_ms,
        },
    )
    return AskResponse(
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        citations=[
            CitationOut(
                label=c.label,
                doc_id=c.doc_id,
                title=c.title,
                filename=c.filename,
                pages=c.pages,
                text=c.text,
            )
            for c in result.citations
        ],
        config=result.config,
        model=result.model,
        prompt_version=result.prompt_version,
        timings_ms={k: round(v, 1) for k, v in result.timings_ms.items()},
        usage=asdict(result.usage),
    )


@app.get("/results")
def results(svc: Svc) -> list[dict[str, Any]]:
    """Summaries of the latest benchmark run per mode, read from committed artifacts."""
    runs_dir = svc.settings.artifacts_dir / "runs"
    latest: dict[str, Path] = {}
    for run in sorted(runs_dir.glob("*")) if runs_dir.exists() else []:
        if (run / "summary.json").exists():
            latest[run.name.rsplit("-", 1)[-1]] = run
    out = []
    for mode, run in sorted(latest.items()):
        env = json.loads((run / "environment.json").read_text())
        out.append(
            {
                "run_id": run.name,
                "mode": mode,
                "answer_model": env.get("answer_model"),
                "cpu": env.get("cpu"),
                "memory_gb": env.get("memory_gb"),
                "summary": json.loads((run / "summary.json").read_text()),
            }
        )
    return out
