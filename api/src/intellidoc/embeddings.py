from __future__ import annotations

from functools import cached_property
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]

# bge models are trained with this instruction on the query side only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class FastEmbedder:
    """Local ONNX embeddings via fastembed. No network after the first model download."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", dim: int = 384) -> None:
        self.name = model
        self.dim = dim

    @cached_property
    def _model(self):  # type: ignore[no-untyped-def]
        from fastembed import TextEmbedding

        return TextEmbedding(self.name)

    def embed_passages(self, texts: list[str]) -> list[Vector]:
        return [np.asarray(v, dtype=np.float32) for v in self._model.embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> Vector:
        prefix = BGE_QUERY_INSTRUCTION if "bge" in self.name.lower() else ""
        return np.asarray(next(iter(self._model.embed([prefix + text]))), dtype=np.float32)
