"""Answer workflow as a LangGraph state machine.

    retrieve -> gate --low similarity--> rewrite (once) -> retrieve
                     --ok--> generate -> validate --model says insufficient--> rewrite (once)
                                                  --no valid citation------> refuse
                                                  --ok--> answer
    Any branch that would need a second rewrite ends in refuse.

Citations are resolved in code from source labels to chunk ids and pages. The model never
supplies page numbers, so a document cannot forge a citation by writing one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from psycopg_pool import ConnectionPool

from intellidoc.answer.models import PROMPT_VERSION, AnswerDraft, AnswerModel, Source, Usage
from intellidoc.retrieval import RetrievalConfig, Retrieved, Retriever

RefusalReason = Literal[
    "low_retrieval_similarity", "insufficient_evidence", "no_valid_citation", "no_documents"
]


@dataclass
class Citation:
    label: str
    chunk_id: int
    doc_id: str
    title: str
    filename: str
    pages: list[int]
    text: str


@dataclass
class AnswerResult:
    question: str
    answer: str
    refused: bool
    refusal_reason: RefusalReason | None
    citations: list[Citation]
    retrieved: list[Retrieved]
    queries: list[str]
    timings_ms: dict[str, float]
    usage: Usage
    model: str
    prompt_version: str = PROMPT_VERSION
    config: str = ""
    flagged_sources: list[str] = field(default_factory=list)


class State(TypedDict, total=False):
    question: str
    query: str
    queries: list[str]
    attempt: int
    retrieved: list[Retrieved]
    draft: AnswerDraft
    refusal_reason: RefusalReason | None
    citations: list[Citation]
    timings_ms: dict[str, float]
    usage: Usage


REFUSAL_MESSAGE = "The indexed documents do not contain enough evidence to answer this."


class AnswerWorkflow:
    def __init__(
        self,
        pool: ConnectionPool,
        retriever: Retriever,
        model: AnswerModel,
        config: RetrievalConfig,
        min_similarity: float = 0.0,
        max_rewrites: int = 1,
    ) -> None:
        self.pool = pool
        self.retriever = retriever
        self.model = model
        self.config = config
        self.min_similarity = min_similarity
        self.max_rewrites = max_rewrites
        self.graph = self._build()

    def _build(self) -> Any:
        g = StateGraph(State)
        g.add_node("retrieve", self._retrieve)
        g.add_node("rewrite", self._rewrite)
        g.add_node("generate", self._generate)
        g.add_node("refuse", self._refuse)
        g.add_edge(START, "retrieve")
        g.add_conditional_edges("retrieve", self._after_retrieve, ["generate", "rewrite", "refuse"])
        g.add_edge("rewrite", "retrieve")
        g.add_conditional_edges("generate", self._after_generate, ["rewrite", "refuse", END])
        g.add_edge("refuse", END)
        return g.compile()

    def run(self, question: str, tags: list[str] | None = None) -> AnswerResult:
        state: State = self.graph.invoke(
            {
                "question": question,
                "query": question,
                "queries": [question],
                "attempt": 0,
                "timings_ms": {},
                "usage": Usage(),
                "refusal_reason": None,
                "citations": [],
            },
            config={
                "run_name": "answer",
                "tags": [self.config.name, PROMPT_VERSION, *(tags or [])],
                "metadata": {
                    "retrieval_config": self.config.name,
                    "model": self.model.name,
                    "prompt_version": PROMPT_VERSION,
                },
            },
        )
        refused = state.get("refusal_reason") is not None
        draft = state.get("draft")
        retrieved = state.get("retrieved", [])
        return AnswerResult(
            question=question,
            answer=REFUSAL_MESSAGE if refused or draft is None else draft.answer,
            refused=refused,
            refusal_reason=state.get("refusal_reason"),
            citations=[] if refused else state.get("citations", []),
            retrieved=retrieved,
            queries=state["queries"],
            timings_ms=state["timings_ms"],
            usage=state["usage"],
            model=self.model.name,
            config=self.config.name,
            flagged_sources=[
                _label(i) for i, r in enumerate(retrieved) if r.metadata.get("injection_flags")
            ],
        )

    # --- nodes ------------------------------------------------------------------------------

    def _retrieve(self, state: State) -> State:
        start = time.perf_counter()
        with self.pool.connection() as conn:
            retrieved = self.retriever.search(conn, state["query"], self.config)
        reason: RefusalReason | None = None
        if not retrieved:
            reason = "no_documents"
        elif max(r.similarity for r in retrieved) < self.min_similarity:
            reason = "low_retrieval_similarity"
        return {
            "retrieved": retrieved,
            "refusal_reason": reason,
            "timings_ms": _timed(state, f"retrieve_{state['attempt']}", start),
        }

    def _rewrite(self, state: State) -> State:
        start = time.perf_counter()
        query, usage = self.model.rewrite(state["question"])
        state["usage"].add(usage)
        return {
            "query": query,
            "queries": [*state["queries"], query],
            "attempt": state["attempt"] + 1,
            "timings_ms": _timed(state, f"rewrite_{state['attempt']}", start),
        }

    def _generate(self, state: State) -> State:
        sources = [
            Source(_label(i), r.title, r.pages, r.text, bool(r.metadata.get("injection_flags")))
            for i, r in enumerate(state["retrieved"])
        ]
        start = time.perf_counter()
        output = self.model.answer(state["question"], sources)
        state["usage"].add(output.usage)
        by_label = {_label(i): r for i, r in enumerate(state["retrieved"])}
        citations = [
            Citation(label, r.chunk_id, r.doc_id, r.title, r.filename, r.pages, r.text)
            for label in dict.fromkeys(output.draft.citations)
            if (r := by_label.get(label)) is not None
        ]
        reason: RefusalReason | None = None
        if not output.draft.answerable:
            reason = "insufficient_evidence"
        elif not citations:
            reason = "no_valid_citation"
        return {
            "draft": output.draft,
            "citations": citations,
            "refusal_reason": reason,
            "timings_ms": _timed(state, f"generate_{state['attempt']}", start),
        }

    def _refuse(self, state: State) -> State:
        return {}

    # --- routing ----------------------------------------------------------------------------

    # Routing functions only read state; LangGraph keeps only what nodes return.

    def _route(self, state: State, ok: str) -> str:
        reason = state.get("refusal_reason")
        if reason is None:
            return ok
        retryable = reason != "no_valid_citation"
        return "rewrite" if retryable and state["attempt"] < self.max_rewrites else "refuse"

    def _after_retrieve(self, state: State) -> str:
        return self._route(state, "generate")

    def _after_generate(self, state: State) -> str:
        return self._route(state, END)


def _label(index: int) -> str:
    return f"S{index + 1}"


def _timed(state: State, key: str, start: float) -> dict[str, float]:
    return {**state["timings_ms"], key: (time.perf_counter() - start) * 1000}
