"""LLM-as-judge groundedness check, used only in online evaluation runs."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from intellidoc.answer.models import Usage, _usage

JUDGE_VERSION = "judge-v1"

JUDGE_PROMPT = """You check whether an answer is supported by the cited sources.

An answer is grounded only if every factual claim in it is stated in, or directly follows from,
the cited source text. Ignore style. Do not use outside knowledge: a true claim that is not in
the sources is unsupported. List each unsupported claim verbatim."""


class Verdict(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)


class ChatJudge:
    def __init__(self, chat: BaseChatModel, name: str) -> None:
        self.name = name
        self._structured = chat.with_structured_output(Verdict, include_raw=True)

    def judge(
        self, question: str, answer: str, cited: list[str], tags: list[str] | None = None
    ) -> tuple[Verdict | None, Usage]:
        sources = "\n\n".join(f"<source>\n{text}\n</source>" for text in cited) or "(none)"
        result = self._structured.invoke(
            [
                ("system", JUDGE_PROMPT),
                ("human", f"Question: {question}\n\nAnswer: {answer}\n\nCited sources:\n{sources}"),
            ],
            config={"run_name": "groundedness_judge", "tags": [JUDGE_VERSION, *(tags or [])]},
        )
        assert isinstance(result, dict)
        return result["parsed"], _usage(result["raw"])
