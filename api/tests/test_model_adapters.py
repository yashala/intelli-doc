from typing import Any

import pytest
from langchain_core.messages import AIMessage

from intellidoc.answer.models import AnswerDraft, ChatAnswerModel, Source, make_chat


@pytest.mark.parametrize(("model", "temperature"), [("gpt-4.1-mini", 0.0), ("gpt-5-mini", None)])
def test_openai_reasoning_models_keep_default_temperature(
    model: str, temperature: float | None
) -> None:
    chat = make_chat("openai", model, "sk-test")
    assert type(chat).__name__ == "ChatOpenAI"
    assert chat.temperature == temperature  # type: ignore[attr-defined]


def test_gemini_provider_builds_gemini_client() -> None:
    assert (
        type(make_chat("gemini", "gemini-2.5-flash", "test")).__name__ == "ChatGoogleGenerativeAI"
    )


class StubChat:
    """Stands in for a LangChain chat model: returns a fixed structured-output result."""

    def __init__(self, parsed: AnswerDraft | None) -> None:
        self.parsed = parsed

    def with_structured_output(self, schema: Any, include_raw: bool) -> "StubChat":
        assert schema is AnswerDraft and include_raw
        return self

    def invoke(self, messages: Any, **_: Any) -> dict[str, Any]:
        raw = AIMessage(
            content="",
            usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
        )
        return {"raw": raw, "parsed": self.parsed, "parsing_error": None}


SOURCES = [Source("S1", "Memo", [2], "Probe L12 cost 7,450 dollars.")]


def test_structured_answer_and_usage_are_returned() -> None:
    draft = AnswerDraft(answerable=True, answer="7,450 dollars", citations=["S1"])
    output = ChatAnswerModel(StubChat(draft), "stub").answer("Cost?", SOURCES)  # type: ignore[arg-type]
    assert output.draft == draft
    assert (output.usage.input_tokens, output.usage.output_tokens, output.usage.calls) == (
        120,
        30,
        1,
    )


def test_unparseable_response_becomes_a_non_answer() -> None:
    output = ChatAnswerModel(StubChat(None), "stub").answer("Cost?", SOURCES)  # type: ignore[arg-type]
    assert output.draft.answerable is False
    assert output.draft.citations == []
