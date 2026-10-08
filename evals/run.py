"""Run the benchmark: ingest the corpus, answer every question under each configuration, score.

    uv run --project api python evals/run.py --llm offline
    uv run --project api python evals/run.py --llm gemini --judge --split test

Writes artifacts/runs/<run_id>/{environment.json, records.jsonl, summary.json}.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import psycopg
import yaml
from pgvector.psycopg import register_vector

from intellidoc.answer.graph import AnswerWorkflow
from intellidoc.answer.models import (
    PROMPT_VERSION,
    AnswerModel,
    ExtractiveModel,
    GeminiModel,
    Usage,
)
from intellidoc.db import apply_schema, create_pool
from intellidoc.embeddings import FastEmbedder
from intellidoc.evaluation.judge import JUDGE_VERSION, GeminiJudge
from intellidoc.evaluation.metrics import (
    ChunkRef,
    Question,
    by_category,
    contains_all,
    percentile,
    score,
    summarise,
)
from intellidoc.ingest.service import document_id, ingest
from intellidoc.retrieval import Reranker, RetrievalConfig, Retriever
from intellidoc.settings import get_settings

ROOT = Path(__file__).resolve().parent.parent
CORPUS = [ROOT / "evals/corpus/pdfs", ROOT / "evals/corpus/real"]
QUESTION_FILES = [ROOT / "evals/questions.jsonl", ROOT / "evals/questions_real.jsonl"]


def load_questions(split: str) -> list[Question]:
    questions = [Question(**json.loads(line)) for path in QUESTION_FILES for line in path.open()]
    return [q for q in questions if split == "all" or q.split == split]


def load_configs(names: list[str] | None) -> list[dict[str, Any]]:
    configs = [yaml.safe_load(p.read_text()) for p in sorted((ROOT / "experiments").glob("*.yaml"))]
    return [c for c in configs if not names or c["name"] in names]


def corpus_documents() -> dict[str, Path]:
    return {p.stem: p for d in CORPUS for p in sorted(d.glob("*.pdf"))}


def ensure_ingested(url: str, docs: dict[str, Path], force: bool) -> dict[str, Any]:
    """Ingest documents that are not already stored; return per-document ingest timings."""
    embedder = FastEmbedder()
    timings = {}
    with psycopg.connect(url) as conn:
        apply_schema(conn)
        register_vector(conn)
        for key, path in docs.items():
            doc_id = document_id(path)
            stored = conn.execute(
                "SELECT count(DISTINCT chunker) FROM chunks WHERE doc_id = %s", (doc_id,)
            ).fetchone()
            if stored and stored[0] == 2 and not force:
                continue
            result = ingest(conn, path, embedder)
            timings[key] = {
                "pages": result.pages,
                "ocr_pages": result.ocr_pages,
                "chunks": result.chunks,
                "timings_ms": result.timings_ms,
            }
            print(f"ingested {key}: {result.pages} pages, {result.chunks}")
    return timings


def parse_ok(conn: psycopg.Connection, q: Question, doc_id: str, chunker: str) -> bool:
    """Whether any stored chunk on an expected page contains all evidence strings."""
    rows = conn.execute(
        "SELECT text FROM chunks WHERE doc_id = %s AND chunker = %s AND pages && %s::int[]",
        (doc_id, chunker, q.expected_pages),
    ).fetchall()
    return any(contains_all(row[0], q.evidence) for row in rows)


def environment(args: argparse.Namespace, model: str) -> dict[str, Any]:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
        except OSError:
            return ""

    packages = [
        "langgraph",
        "langchain-core",
        "langchain-google-genai",
        "fastembed",
        "pdfplumber",
        "pypdfium2",
        "pytesseract",
        "psycopg",
        "pgvector",
        "onnxruntime",
    ]
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(run(["git", "status", "--porcelain"])),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
        "python": platform.python_version(),
        "tesseract": run(["tesseract", "--version"]).splitlines()[0]
        if run(["tesseract", "--version"])
        else "",
        "packages": {p: _version(p) for p in packages},
        "llm": args.llm,
        "answer_model": model,
        "judge_model": get_settings().judge_model if args.judge else None,
        "prompt_version": PROMPT_VERSION,
        "judge_version": JUDGE_VERSION if args.judge else None,
        "embedding_model": get_settings().embedding_model,
        "reranker_model": get_settings().reranker_model,
        "memory_at_start": memory_state(),
        "split": args.split,
        "repeats": args.repeat,
        "corpus_manifest": json.loads((ROOT / "evals/corpus/manifest.json").read_text()),
        "real_sources": json.loads((ROOT / "evals/corpus/real/sources.json").read_text()),
    }


def memory_state() -> dict[str, Any]:
    """Swap use and free memory, so latency measured under memory pressure is identifiable."""
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
    containers = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}}"],
        capture_output=True,
        text=True,
    )
    return {
        "swap": swap.stdout.strip() or None,
        "docker_containers": containers.stdout.strip().splitlines(),
    }


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def build_model(llm: str) -> AnswerModel:
    settings = get_settings()
    if llm == "offline":
        return ExtractiveModel()
    if settings.google_api_key is None:
        raise SystemExit("GOOGLE_API_KEY is required for --llm gemini")
    return GeminiModel(settings.answer_model, settings.google_api_key.get_secret_value())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["offline", "gemini"], default="offline")
    parser.add_argument("--configs", nargs="*", help="experiment names (default: all)")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--repeat", type=int, default=1, help="runs per question for latency")
    parser.add_argument("--judge", action="store_true", help="LLM groundedness judge (online)")
    parser.add_argument("--limit", type=int, help="first N questions only (smoke runs)")
    parser.add_argument("--reingest", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "artifacts/runs"))
    args = parser.parse_args()

    settings = get_settings()
    url = os.environ.get("EVAL_DATABASE_URL", settings.database_url)
    docs = corpus_documents()
    ingest_timings = ensure_ingested(url, docs, args.reingest)
    doc_keys = {document_id(path): key for key, path in docs.items()}
    key_to_id = {key: doc_id for doc_id, key in doc_keys.items()}

    questions = load_questions(args.split)[: args.limit]
    model = build_model(args.llm)
    judge = None
    if args.judge:
        if settings.google_api_key is None:
            raise SystemExit("--judge needs GOOGLE_API_KEY")
        judge = GeminiJudge(settings.judge_model, settings.google_api_key.get_secret_value())

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{args.llm}"
    out = Path(args.out) / run_id
    out.mkdir(parents=True)
    env = environment(args, model.name)
    env["ingest"] = ingest_timings
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")

    pool = create_pool(url)
    retriever = Retriever(FastEmbedder(), Reranker(settings.reranker_model))
    summaries: dict[str, Any] = {}
    with (out / "records.jsonl").open("w") as records, pool.connection() as conn:
        for cfg in load_configs(args.configs):
            config = RetrievalConfig(cfg["chunker"], cfg["mode"], cfg["k"], cfg["candidates"])
            workflow = AnswerWorkflow(pool, retriever, model, config, cfg["min_similarity"])
            scores, latencies, usage = [], {"total": []}, Usage()  # type: ignore[var-annotated]
            judged: list[bool] = []
            judge_usage = Usage()
            for q in questions:
                for attempt in range(args.repeat):
                    start = time.perf_counter()
                    result = workflow.run(q.question, tags=[run_id])
                    latencies["total"].append((time.perf_counter() - start) * 1000)
                    for stage, ms in result.timings_ms.items():
                        latencies.setdefault(stage.rsplit("_", 1)[0], []).append(ms)
                    if attempt:
                        continue
                    usage.add(result.usage)
                    ok = (
                        parse_ok(conn, q, key_to_id[q.expected_doc], cfg["chunker"])
                        if q.answerable and q.expected_doc
                        else None
                    )
                    s = score(
                        q,
                        [ChunkRef(doc_keys[r.doc_id], r.pages, r.text) for r in result.retrieved],
                        [ChunkRef(doc_keys[c.doc_id], c.pages, c.text) for c in result.citations],
                        result.answer,
                        result.refused,
                        ok,
                    )
                    verdict = None
                    if judge and not result.refused:
                        v, used = judge.judge(
                            q.question, result.answer, [c.text for c in result.citations]
                        )
                        judge_usage.add(used)
                        verdict = v.model_dump() if v else None
                        if v and q.answerable:
                            judged.append(v.grounded)
                    scores.append(s)
                    records.write(
                        json.dumps(
                            {
                                "config": cfg["name"],
                                "question": asdict(q),
                                "score": asdict(s),
                                "answer": result.answer,
                                "refusal_reason": result.refusal_reason,
                                "queries": result.queries,
                                "retrieved": [
                                    {
                                        "doc": doc_keys[r.doc_id],
                                        "pages": r.pages,
                                        "chunk_id": r.chunk_id,
                                        "score": round(r.score, 4),
                                        "similarity": round(r.similarity, 4),
                                    }
                                    for r in result.retrieved
                                ],
                                "citations": [
                                    {
                                        "doc": doc_keys[c.doc_id],
                                        "pages": c.pages,
                                        "chunk_id": c.chunk_id,
                                    }
                                    for c in result.citations
                                ],
                                "flagged_sources": result.flagged_sources,
                                "timings_ms": result.timings_ms,
                                "usage": asdict(result.usage),
                                "judge": verdict,
                            }
                        )
                        + "\n"
                    )
            summary = summarise(scores)
            summary["judge_grounded_rate"] = round(sum(judged) / len(judged), 4) if judged else None
            summary["latency_ms"] = {
                stage: {"p50": percentile(v, 50), "p95": percentile(v, 95), "n": len(v)}
                for stage, v in latencies.items()
            }
            summary["usage"] = asdict(usage)
            summary["judge_usage"] = asdict(judge_usage)
            summary["by_category"] = by_category(scores)
            summary["config"] = cfg
            summaries[cfg["name"]] = summary
            print(
                f"{cfg['name']}: evidence_hit={summary['evidence_hit_rate']} "
                f"answer_acc={summary['answer_accuracy']} "
                f"cit_prec={summary['citation_precision']} "
                f"refusal={summary['correct_refusal_rate']} "
                f"p50={summary['latency_ms']['total']['p50']}ms"
            )
    pool.close()
    env["memory_at_end"] = memory_state()
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
