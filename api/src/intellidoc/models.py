from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ElementKind = Literal["heading", "text", "table", "image_text"]
ExtractionMethod = Literal["native", "ocr"]


@dataclass(frozen=True)
class Element:
    """A unit of page content with its provenance."""

    page: int  # 1-based
    kind: ElementKind
    text: str
    method: ExtractionMethod
    bbox: tuple[float, float, float, float] | None = None  # x0, top, x1, bottom in PDF points
    ocr_confidence: float | None = None  # mean Tesseract word confidence, 0-100


@dataclass
class PageInfo:
    page: int
    method: ExtractionMethod
    native_chars: int
    ocr_confidence: float | None = None


@dataclass
class ParsedDocument:
    title: str
    pages: list[PageInfo]
    elements: list[Element]


@dataclass
class Chunk:
    doc_id: str
    pages: list[int]
    kind: ElementKind
    text: str
    section: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
