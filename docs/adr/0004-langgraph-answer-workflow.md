# ADR 0004: LangGraph for the answer workflow; LangChain only for the model adapter

Status: accepted (2026-10-08)

## Context
Answering is not a single call. It retrieves, checks retrieval confidence, may rewrite the
query and retry once, generates a structured answer, validates citations, and refuses with
a recorded reason when a gate fails.

## Decision
- LangGraph `StateGraph` for the workflow. The branches and the bounded retry are explicit
  edges, every node is traced as its own span when LangSmith is enabled, and the state carries
  timings and token usage.
- LangChain chat models (`ChatOpenAI`, `ChatGoogleGenerativeAI`) with
  `with_structured_output(AnswerDraft, include_raw=True)`, so the answer is schema-checked and token usage stays available.
- No LangChain document loaders, text splitters or vector store: they would discard the
  per-element page provenance, or conflict with ADR 0001.

## What a simpler version looks like
A plain function with two `if` statements and a loop would implement the same control flow in
about 40 lines. LangGraph adds a dependency and a state-reducer model to learn. It earns its
place here by making the control flow inspectable in traces and by keeping routing separate
from node logic; if tracing were not a requirement, the plain function would be the better choice.

## Consequences
- Conditional-edge functions must not mutate state (LangGraph discards it); decisions are
  returned from nodes as `refusal_reason`. This was a bug in the first draft.
- Citations are resolved from labels to chunk ids in code, so the model cannot cite pages
  that were not retrieved.
