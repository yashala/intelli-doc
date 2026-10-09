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
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from intellidoc.answer.graph import AnswerWorkflow
from intellidoc.answer.models import (
    PROMPT_VERSION,
    AnswerModel,
    ChatAnswerModel,
    ExtractiveModel,
    ModelOutput,
    Source,
    Usage,
    make_chat,
)
from intellidoc.db import apply_schema, create_pool
from intellidoc.embeddings import FastEmbedder
from intellidoc.evaluation.judge import JUDGE_VERSION, ChatJudge
from intellidoc.evaluation.metrics import (
    ChunkRef,
    Question,
    Score,
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
        "langchain-openai",
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
    chat = make_chat(llm, settings.answer_model, settings.api_key(llm))  # type: ignore[arg-type]
    return ChatAnswerModel(chat, settings.answer_model)


class Throttle:
    """Space model calls to stay under a requests-per-minute limit (free API tiers)."""

    def __init__(self, rpm: float | None) -> None:
        self.interval = 60.0 / rpm if rpm else 0.0
        self.last = 0.0

    def wait(self) -> float:
        """Sleep if needed; return the milliseconds slept."""
        delay = self.last + self.interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()
        return max(delay, 0.0) * 1000


class ThrottledModel:
    """Applies the throttle and records each wait, so it can be removed from stage latency.

    The wait happens inside the workflow's generate/rewrite nodes; without this correction the
    recorded generate latency would be mostly deliberate sleeping.
    """

    def __init__(self, inner: AnswerModel, throttle: Throttle) -> None:
        self.inner, self.throttle, self.name = inner, throttle, inner.name
        self.waits: list[tuple[str, float]] = []

    def answer(self, question: str, sources: list[Source]) -> ModelOutput:
        self.waits.append(("generate", self.throttle.wait()))
        return self.inner.answer(question, sources)

    def rewrite(self, question: str) -> tuple[str, Usage]:
        self.waits.append(("rewrite", self.throttle.wait()))
        return self.inner.rewrite(question)


def remove_waits(timings: dict[str, float], waits: list[tuple[str, float]]) -> dict[str, float]:
    """Subtract throttle waits from the stage they occurred in (n-th wait -> stage_n)."""
    adjusted = dict(timings)
    seen: dict[str, int] = {}
    for stage, ms in waits:
        index = seen.get(stage, 0)
        seen[stage] = index + 1
        key = f"{stage}_{index}"
        if key in adjusted:
            adjusted[key] = max(adjusted[key] - ms, 0.0)
    return adjusted


def load_prices() -> dict[str, dict[str, float]]:
    prices: dict[str, dict[str, float]] = yaml.safe_load((ROOT / "evals/pricing.yaml").read_text())
    return prices


def cost_usd(usage: dict[str, int], model: str, prices: dict[str, dict[str, float]]) -> float:
    if model not in prices:
        raise SystemExit(f"no price for {model} in evals/pricing.yaml; refusing to run unbudgeted")
    price = prices[model]
    return (usage["input_tokens"] * price["input"] + usage["output_tokens"] * price["output"]) / 1e6


def is_quota_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "ResourceExhausted" in text


def summarise_records(
    records: list[dict[str, Any]], configs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build per-config summaries from records.jsonl, so resumed runs summarise identically."""
    summaries: dict[str, Any] = {}
    for cfg in configs:
        rows = [r for r in records if r["config"] == cfg["name"]]
        if not rows:
            continue
        scores = [Score(**r["score"]) for r in rows]
        latencies: dict[str, list[float]] = {"total": [ms for r in rows for ms in r["total_ms"]]}
        for r in rows:
            for stage, ms in r["timings_ms"].items():
                latencies.setdefault(stage.rsplit("_", 1)[0], []).append(ms)
        usage, judge_usage = Usage(), Usage()
        for r in rows:
            usage.add(Usage(**r["usage"]))
            judge_usage.add(Usage(**r["judge_usage"]))
        judged = [
            r["judge"]["grounded"] for r in rows if r["judge"] and r["question"]["answerable"]
        ]
        summary = summarise(scores)
        summary["judge_grounded_rate"] = round(sum(judged) / len(judged), 4) if judged else None
        summary["latency_ms"] = {
            stage: {"p50": percentile(v, 50), "p95": percentile(v, 95), "n": len(v)}
            for stage, v in latencies.items()
        }
        summary["usage"] = asdict(usage)
        summary["judge_usage"] = asdict(judge_usage)
        summary["cost_usd"] = round(sum(r.get("cost_usd", 0.0) for r in rows), 4)
        summary["by_category"] = by_category(scores)
        summary["config"] = cfg
        summaries[cfg["name"]] = summary
    return summaries


def main() -> None:
    # LangSmith reads LANGSMITH_* from the process environment, not from pydantic settings.
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["offline", "gemini", "openai"], default="offline")
    parser.add_argument("--configs", nargs="*", help="experiment names (default: all)")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--repeat", type=int, default=1, help="runs per question for latency")
    parser.add_argument("--judge", action="store_true", help="LLM groundedness judge (online)")
    parser.add_argument("--limit", type=int, help="first N questions only (smoke runs)")
    parser.add_argument("--rpm", type=float, help="max model requests per minute")
    parser.add_argument("--resume", help="continue an interrupted run directory")
    parser.add_argument(
        "--max-cost-usd", type=float, help="stop before estimated spend exceeds this (all sessions)"
    )
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
    configs = load_configs(args.configs)
    throttle = Throttle(args.rpm)
    model: AnswerModel = build_model(args.llm)
    if args.rpm:
        model = ThrottledModel(model, throttle)
    judge = None
    if args.judge:
        if args.llm == "offline":
            raise SystemExit("--judge needs an online provider")
        chat = make_chat(args.llm, settings.judge_model, settings.api_key(args.llm))
        judge = ChatJudge(chat, settings.judge_model)

    if args.resume:
        out = Path(args.resume)
        env = json.loads((out / "environment.json").read_text())
        env.setdefault("resumed_at", []).append(datetime.now(UTC).isoformat(timespec="seconds"))
    else:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{args.llm}"
        out = Path(args.out) / run_id
        out.mkdir(parents=True)
        env = environment(args, model.name)
        env["ingest"] = ingest_timings
    env["rpm_limit"] = args.rpm
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    records_path = out / "records.jsonl"
    existing = [json.loads(line) for line in records_path.open()] if records_path.exists() else []
    done = {(r["config"], r["question"]["id"]) for r in existing}
    prices = load_prices()
    for name in {model.name, judge.name if judge else model.name}:
        cost_usd({"input_tokens": 0, "output_tokens": 0}, name, prices)  # fail fast if unpriced
    spent = sum(r.get("cost_usd", 0.0) for r in existing)
    per_question: list[float] = [r["cost_usd"] for r in existing if "cost_usd" in r]
    stopped_on_budget = False

    pool = create_pool(url)
    retriever = Retriever(FastEmbedder(), Reranker(settings.reranker_model))
    stopped_on_quota = False
    with records_path.open("a") as records, pool.connection() as conn:
        for cfg in configs:
            if stopped_on_quota or stopped_on_budget:
                break
            config = RetrievalConfig(cfg["chunker"], cfg["mode"], cfg["k"], cfg["candidates"])
            workflow = AnswerWorkflow(pool, retriever, model, config, cfg["min_similarity"])
            for q in questions:
                if (cfg["name"], q.id) in done:
                    continue
                # Leave headroom of the most expensive question so far (x2) before each call.
                headroom = 2 * max(per_question, default=0.01)
                if args.max_cost_usd is not None and spent + headroom > args.max_cost_usd:
                    print(f"budget stop: spent ${spent:.4f} of ${args.max_cost_usd}")
                    stopped_on_budget = True
                    break
                try:
                    record = run_question(
                        workflow,
                        judge,
                        throttle,
                        conn,
                        q,
                        cfg,
                        args.repeat,
                        out.name,
                        doc_keys,
                        key_to_id,
                    )
                except Exception as exc:
                    if not is_quota_error(exc):
                        raise
                    print(f"quota exhausted at {cfg['name']} {q.id}: {type(exc).__name__}")
                    stopped_on_quota = True
                    break
                record["cost_usd"] = round(
                    cost_usd(record["usage"], model.name, prices)
                    + cost_usd(record["judge_usage"], judge.name if judge else model.name, prices),
                    6,
                )
                spent += record["cost_usd"]
                per_question.append(record["cost_usd"])
                records.write(json.dumps(record) + "\n")
                records.flush()
            print(f"{cfg['name']}: done")
    pool.close()

    all_records = [json.loads(line) for line in records_path.open()]
    summaries = summarise_records(all_records, configs)
    expected = len(configs) * len(questions)
    env["complete"] = len(all_records) >= expected
    env["records"] = f"{len(all_records)}/{expected}"
    env["cost_usd_total"] = round(sum(r.get("cost_usd", 0.0) for r in all_records), 4)
    env["pricing"] = prices
    env["memory_at_end"] = memory_state()
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    for name, sm in summaries.items():
        print(
            f"{name}: evidence_hit={sm['evidence_hit_rate']} answer_acc={sm['answer_accuracy']} "
            f"cit_prec={sm['citation_precision']} judge={sm['judge_grounded_rate']}"
        )
    print(f"wrote {out} ({env['records']} records, estimated ${env['cost_usd_total']})")
    if stopped_on_quota:
        print(f"resume with: evals/run.py --llm {args.llm} --judge --rpm {args.rpm} --resume {out}")


def run_question(
    workflow: AnswerWorkflow,
    judge: ChatJudge | None,
    throttle: Throttle,
    conn: psycopg.Connection,
    q: Question,
    cfg: dict[str, Any],
    repeat: int,
    run_id: str,
    doc_keys: dict[str, str],
    key_to_id: dict[str, str],
) -> dict[str, Any]:
    totals, waited = [], 0.0
    throttled = workflow.model if isinstance(workflow.model, ThrottledModel) else None
    for _ in range(repeat):
        if throttled:
            throttled.waits.clear()
        start = time.perf_counter()
        result = workflow.run(q.question, tags=[run_id])
        waits = throttled.waits[:] if throttled else []
        waited = sum(ms for _, ms in waits)
        totals.append((time.perf_counter() - start) * 1000 - waited)
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
    verdict, judge_usage = None, Usage()
    if judge and not result.refused:
        throttle.wait()
        v, judge_usage = judge.judge(
            q.question, result.answer, [c.text for c in result.citations], [run_id, cfg["name"]]
        )
        verdict = v.model_dump() if v else None
    return {
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
            {"doc": doc_keys[c.doc_id], "pages": c.pages, "chunk_id": c.chunk_id}
            for c in result.citations
        ],
        "flagged_sources": result.flagged_sources,
        "timings_ms": remove_waits(result.timings_ms, waits),
        "total_ms": totals,
        "throttle_wait_ms": round(waited, 1),
        "usage": asdict(result.usage),
        "judge": verdict,
        "judge_usage": asdict(judge_usage),
    }


if __name__ == "__main__":
    main()
