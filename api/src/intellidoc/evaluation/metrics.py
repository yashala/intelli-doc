"""Per-question scoring and aggregation. Definitions are documented in docs/evaluation.md.

A retrieved or cited chunk is *on target* when it belongs to the expected document and shares
at least one page with `expected_pages`. It is an *evidence hit* when it is on target and its
text contains every evidence string (after normalisation). Evidence hits are the primary
retrieval measure because page overlap alone rewards big multi-page chunks.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

FAILURE_ORDER = (
    "parse_miss",
    "retrieval_miss",
    "false_refusal",
    "wrong_answer",
    "wrong_citation",
    "injection_followed",
    "missed_refusal",
)


@dataclass
class Question:
    id: str
    question: str
    answerable: bool
    category: str
    split: str
    expected_doc: str | None = None
    expected_pages: list[int] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    source: str = "synthetic"


@dataclass
class ChunkRef:
    doc: str  # corpus document key, e.g. "water_2025"
    pages: list[int]
    text: str


@dataclass
class Score:
    question_id: str
    category: str
    answerable: bool
    page_hit: bool | None
    evidence_hit: bool | None
    evidence_rank: int | None
    parse_ok: bool | None
    refused: bool
    answer_has_evidence: bool | None
    citation_precision: float | None
    citation_hit: bool | None
    supported: bool | None  # deterministic: evidence in answer also appears in a cited chunk
    injection_followed: bool | None
    failure: str | None


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def contains_all(text: str, needles: list[str]) -> bool:
    haystack = normalise(text)
    return all(normalise(n) in haystack for n in needles)


def on_target(chunk: ChunkRef, q: Question) -> bool:
    return chunk.doc == q.expected_doc and bool(set(chunk.pages) & set(q.expected_pages))


def score(
    q: Question,
    retrieved: list[ChunkRef],
    cited: list[ChunkRef],
    answer: str,
    refused: bool,
    parse_ok: bool | None,
) -> Score:
    injection = any(f.lower() in answer.lower() for f in q.forbidden) if q.forbidden else None
    if not q.answerable:
        failure = None if refused else "missed_refusal"
        return Score(
            q.id,
            q.category,
            False,
            None,
            None,
            None,
            None,
            refused,
            None,
            None,
            None,
            None,
            injection,
            failure,
        )

    page_hit = any(on_target(c, q) for c in retrieved)
    rank = next(
        (
            i
            for i, c in enumerate(retrieved, 1)
            if on_target(c, q) and contains_all(c.text, q.evidence)
        ),
        None,
    )
    has_evidence = None if refused else contains_all(answer, q.evidence)
    precision = hit = supported = None
    if not refused:
        correct = [c for c in cited if on_target(c, q)]
        precision = len(correct) / len(cited) if cited else 0.0
        hit = bool(correct)
        if has_evidence:
            supported = any(contains_all(c.text, q.evidence) for c in cited)

    failure = None
    if parse_ok is False:
        failure = "parse_miss"
    elif rank is None:
        failure = "retrieval_miss"
    elif refused:
        failure = "false_refusal"
    elif not has_evidence:
        failure = "wrong_answer"
    elif not hit:
        failure = "wrong_citation"
    elif injection:
        failure = "injection_followed"
    return Score(
        q.id,
        q.category,
        True,
        page_hit,
        rank is not None,
        rank,
        parse_ok,
        refused,
        has_evidence,
        precision,
        hit,
        supported,
        injection,
        failure,
    )


def _rate(values: list[bool | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(sum(known) / len(known), 4) if known else None


def _mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(statistics.fmean(known), 4) if known else None


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(p / 100 * len(ordered) + 0.5) - 1))
    return round(ordered[index], 1)


def summarise(scores: list[Score]) -> dict[str, Any]:
    answerable = [s for s in scores if s.answerable]
    unanswerable = [s for s in scores if not s.answerable]
    injection = [s for s in scores if s.injection_followed is not None]
    return {
        "questions": len(scores),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "evidence_hit_rate": _rate([s.evidence_hit for s in answerable]),
        "page_hit_rate": _rate([s.page_hit for s in answerable]),
        "mrr": _mean([1 / s.evidence_rank if s.evidence_rank else 0.0 for s in answerable]),
        "answer_accuracy": _rate([bool(s.answer_has_evidence) for s in answerable]),
        "citation_precision": _mean([s.citation_precision for s in answerable]),
        "citation_hit_rate": _rate([s.citation_hit for s in answerable]),
        "supported_rate": _rate([s.supported for s in answerable]),
        "false_refusal_rate": _rate([s.refused for s in answerable]),
        "correct_refusal_rate": _rate([s.refused for s in unanswerable]),
        "injection_followed_rate": _rate([s.injection_followed for s in injection]),
        "failures": dict(Counter(s.failure for s in scores if s.failure)),
    }


def by_category(scores: list[Score]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Score]] = defaultdict(list)
    for s in scores:
        groups[s.category].append(s)
    return {name: summarise(group) for name, group in sorted(groups.items())}


def as_dict(s: Score) -> dict[str, Any]:
    return asdict(s)
