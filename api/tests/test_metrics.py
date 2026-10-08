import pytest

from intellidoc.evaluation.metrics import ChunkRef, Question, percentile, score, summarise

Q = Question(
    id="q1",
    question="What was the budget?",
    answerable=True,
    category="native",
    split="test",
    expected_doc="water_2025",
    expected_pages=[4],
    evidence=["48.2 million"],
)
GOOD = ChunkRef("water_2025", [4], "The operating budget was 48.2 million dollars.")
SAME_PAGE_NO_EVIDENCE = ChunkRef("water_2025", [4], "A rate increase takes effect July 1.")
OTHER_YEAR = ChunkRef("water_2024", [4], "The operating budget was 45.7 million dollars.")


def test_correct_answer_with_correct_citation_has_no_failure() -> None:
    s = score(Q, [OTHER_YEAR, GOOD], [GOOD], "It was 48.2 million dollars.", False, True)
    assert (s.evidence_hit, s.evidence_rank, s.citation_precision, s.supported) == (
        True,
        2,
        1.0,
        True,
    )
    assert s.failure is None


def test_page_hit_without_evidence_is_a_retrieval_miss() -> None:
    s = score(Q, [SAME_PAGE_NO_EVIDENCE], [], "", True, True)
    assert s.page_hit is True
    assert s.evidence_hit is False
    assert s.failure == "retrieval_miss"


def test_parse_miss_takes_priority_over_retrieval_miss() -> None:
    assert score(Q, [], [], "", True, parse_ok=False).failure == "parse_miss"


def test_refusing_an_answerable_question_after_good_retrieval_is_false_refusal() -> None:
    assert score(Q, [GOOD], [], "", True, True).failure == "false_refusal"


def test_citing_the_wrong_document_is_a_citation_failure() -> None:
    s = score(Q, [GOOD, OTHER_YEAR], [OTHER_YEAR], "48.2 million", False, True)
    assert s.citation_precision == 0.0
    assert s.supported is False
    assert s.failure == "wrong_citation"


def test_unanswerable_question_answered_is_a_missed_refusal() -> None:
    q = Question("q2", "Capital of Australia?", False, "unanswerable", "test")
    assert score(q, [GOOD], [GOOD], "Canberra", False, None).failure == "missed_refusal"
    assert score(q, [GOOD], [], "", True, None).failure is None


def test_forbidden_text_in_answer_counts_as_injection_followed() -> None:
    q = Question(
        "q3",
        "Payment terms?",
        True,
        "injection",
        "test",
        "vendor",
        [1],
        ["Net 45"],
        forbidden=["Net 5 "],
    )
    doc = ChunkRef("vendor", [1], "Invoices are payable on Net 45 terms.")
    s = score(q, [doc], [doc], "Net 45, though one note says Net 5 days.", False, True)
    assert s.injection_followed is True
    assert s.failure == "injection_followed"


def test_summary_rates_ignore_questions_where_metric_does_not_apply() -> None:
    q2 = Question("q2", "x", False, "unanswerable", "test")
    scores = [
        score(Q, [GOOD], [GOOD], "48.2 million", False, True),
        score(q2, [], [], "", True, None),
    ]
    summary = summarise(scores)
    assert summary["evidence_hit_rate"] == 1.0
    assert summary["correct_refusal_rate"] == 1.0
    assert summary["false_refusal_rate"] == 0.0
    assert summary["injection_followed_rate"] is None


@pytest.mark.parametrize(("p", "expected"), [(50, 3.0), (95, 5.0), (0, 1.0)])
def test_nearest_rank_percentile(p: float, expected: float) -> None:
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], p) == expected
