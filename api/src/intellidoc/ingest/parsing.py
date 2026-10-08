"""PDF parsing with page-level provenance.

Native pages are read from the text layer with pdfplumber. Pages with almost no text layer
but a large raster image are treated as scans and OCR'd with Tesseract. Embedded images on
native pages are OCR'd separately so text that only exists inside figures is retrievable.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber
import pypdfium2 as pdfium
import pytesseract
from PIL import Image

from intellidoc.ingest.ocr import ocr_page
from intellidoc.models import Element, PageInfo, ParsedDocument

# A page with fewer native characters than this, and a raster image covering most of it,
# is treated as scanned.
SCANNED_MAX_CHARS = 25
SCANNED_MIN_IMAGE_COVERAGE = 0.5
# A raster image covering this much of the page means a scan even if it has a text layer.
SCAN_WITH_TEXT_LAYER_COVERAGE = 0.9
# Embedded images smaller than this fraction of the page are not OCR'd (logos, bullets).
MIN_IMAGE_AREA_FRACTION = 0.02
OCR_DPI = 300
GUTTER_MIN_WIDTH = 6.0  # points; word spaces are ~2-3pt, IRS Pub 501 gutters are ~10pt
MIN_OCR_CONFIDENCE = 30.0


@dataclass
class Line:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    bold: bool


def parse_pdf(path: Path) -> ParsedDocument:
    pages: list[PageInfo] = []
    elements: list[Element] = []
    rendered = pdfium.PdfDocument(str(path))
    try:
        with pdfplumber.open(path) as pdf:
            body_size = _body_font_size(pdf.pages)
            for index, page in enumerate(pdf.pages):
                number = index + 1
                native_chars = sum(1 for c in page.chars if not c["text"].isspace())
                if _is_scanned(page, native_chars):
                    image = _render(rendered[index])
                    page_elements, confidence = ocr_page(image, number, OCR_DPI)
                    pages.append(PageInfo(number, "ocr", native_chars, confidence))
                else:
                    page_elements = _parse_native_page(page, number, body_size, rendered[index])
                    pages.append(PageInfo(number, "native", native_chars))
                elements.extend(page_elements)
    finally:
        rendered.close()
    return ParsedDocument(title=_title(elements, path), pages=pages, elements=elements)


def _is_scanned(page: Any, native_chars: int) -> bool:
    """A page is a scan if it is (almost) entirely one raster image.

    Scans often carry an existing OCR text layer of unknown quality (NASA TN D-233 renders
    "AZIMUTH" as "AZIIvIUTH"). Such layers are discarded and the page is re-OCR'd, so every
    scanned page goes through the same OCR pipeline; `native_chars` records what was dropped.
    """
    coverage = sum(_area(img) for img in page.images) / float(page.width * page.height)
    if coverage >= SCAN_WITH_TEXT_LAYER_COVERAGE:
        return True
    return native_chars < SCANNED_MAX_CHARS and coverage >= SCANNED_MIN_IMAGE_COVERAGE


def _area(obj: dict[str, Any]) -> float:
    return float((obj["x1"] - obj["x0"]) * (obj["bottom"] - obj["top"]))


def _body_font_size(pages: list[Any]) -> float:
    sizes = Counter(round(c["size"], 1) for page in pages for c in page.chars)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _render(page: pdfium.PdfPage) -> Image.Image:
    image: Image.Image = page.render(scale=OCR_DPI / 72).to_pil()
    return image.convert("RGB")


def _parse_native_page(
    page: Any, number: int, body_size: float, rendered: pdfium.PdfPage
) -> list[Element]:
    positioned: list[tuple[int, float, Element]] = []  # (column, top, element)

    tables = page.find_tables()
    table_boxes = [t.bbox for t in tables]
    for table in tables:
        markdown = _table_to_markdown(table.extract())
        if markdown:
            x0, top, x1, bottom = table.bbox
            positioned.append(
                (0, top, Element(number, "table", markdown, "native", (x0, top, x1, bottom)))
            )

    words = [
        w
        for w in page.extract_words(extra_attrs=["size", "fontname"], keep_blank_chars=False)
        if not any(_inside(w, box) for box in table_boxes)
    ]
    gutters = _find_gutters(words, float(page.width))
    columns = _split_columns(words, gutters)
    for column_index, column_words in enumerate(columns):
        for element in _blocks(_lines(column_words), number, body_size):
            assert element.bbox is not None
            positioned.append((column_index, element.bbox[1], element))

    page_area = float(page.width * page.height)
    image_page: Image.Image | None = None
    for img in page.images:
        if _area(img) / page_area < MIN_IMAGE_AREA_FRACTION:
            continue
        if image_page is None:
            image_page = _render(rendered)
        scale = OCR_DPI / 72
        crop = image_page.crop(
            (img["x0"] * scale, img["top"] * scale, img["x1"] * scale, img["bottom"] * scale)
        )
        text, confidence = _ocr_text(crop)
        if len(text.split()) >= 2:
            bbox = (float(img["x0"]), float(img["top"]), float(img["x1"]), float(img["bottom"]))
            column = _column_of((img["x0"] + img["x1"]) / 2, gutters)
            positioned.append(
                (column, img["top"], Element(number, "image_text", text, "ocr", bbox, confidence))
            )

    positioned.sort(key=lambda item: (item[0], item[1]))
    return [element for _, _, element in positioned]


def _inside(word: dict[str, Any], box: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = box
    cx = (word["x0"] + word["x1"]) / 2
    cy = (word["top"] + word["bottom"]) / 2
    return bool(x0 <= cx <= x1 and top <= cy <= bottom)


def _find_gutters(words: list[dict[str, Any]], width: float) -> list[float]:
    """Find vertical whitespace bands that separate text columns.

    Returns gutter x positions (empty for single-column pages). A few words may cross a
    gutter (running headers, full-width titles); more than 3% means it is not a gutter.
    The band must be wider than an inter-word space, otherwise aligned word gaps in a short
    single-column paragraph look like a gutter.
    """
    if len(words) < 20:
        return []
    half = GUTTER_MIN_WIDTH / 2
    valid: list[float] = []
    for step in range(30, 171):  # 15% to 85% of the page width in 0.5% steps
        x = width * step / 200
        crossing = sum(1 for w in words if w["x0"] < x + half and w["x1"] > x - half)
        left = sum(1 for w in words if w["x1"] <= x - half)
        right = sum(1 for w in words if w["x0"] >= x + half)
        if crossing <= 0.03 * len(words) and min(left, right) >= 0.15 * len(words):
            valid.append(x)
    # Adjacent valid positions belong to the same whitespace band; use each band's centre.
    gutters: list[float] = []
    band: list[float] = []
    for x in valid:
        if band and x - band[-1] > width / 200 + 0.01:
            gutters.append((band[0] + band[-1]) / 2)
            band = []
        band.append(x)
    if band:
        gutters.append((band[0] + band[-1]) / 2)
    return gutters


def _column_of(x_center: float, gutters: list[float]) -> int:
    return sum(1 for g in gutters if x_center >= g)


def _split_columns(words: list[dict[str, Any]], gutters: list[float]) -> list[list[dict[str, Any]]]:
    columns: list[list[dict[str, Any]]] = [[] for _ in range(len(gutters) + 1)]
    for w in words:
        columns[_column_of((w["x0"] + w["x1"]) / 2, gutters)].append(w)
    return columns


def _lines(words: list[dict[str, Any]], tolerance: float = 3.0) -> list[Line]:
    lines: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1][0]["top"] - word["top"]) <= tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])
    result = []
    for group in lines:
        group.sort(key=lambda w: w["x0"])
        result.append(
            Line(
                text=" ".join(w["text"] for w in group),
                x0=min(w["x0"] for w in group),
                x1=max(w["x1"] for w in group),
                top=min(w["top"] for w in group),
                bottom=max(w["bottom"] for w in group),
                size=max(w["size"] for w in group),
                bold=all("Bold" in w["fontname"] for w in group),
            )
        )
    return result


def _is_heading_line(line: Line, body_size: float) -> bool:
    if line.text.endswith("-"):  # a hyphenated break means a wrapped run-in head, not a heading
        return False
    if line.size >= body_size * 1.2:
        return True
    return line.bold and len(line.text) < 80


def _blocks(lines: list[Line], number: int, body_size: float) -> list[Element]:
    """Group lines into headings and paragraphs using font size and vertical gaps."""
    elements: list[Element] = []
    current: list[Line] = []

    def flush() -> None:
        if not current:
            return
        heading = all(_is_heading_line(line, body_size) for line in current)
        bbox = (
            min(line.x0 for line in current),
            current[0].top,
            max(line.x1 for line in current),
            current[-1].bottom,
        )
        text = " ".join(line.text for line in current)
        elements.append(Element(number, "heading" if heading else "text", text, "native", bbox))
        current.clear()

    for line in lines:
        if current:
            previous = current[-1]
            gap = line.top - previous.bottom
            height = previous.bottom - previous.top
            style_changed = _is_heading_line(line, body_size) != _is_heading_line(
                previous, body_size
            )
            if gap > 0.6 * height or style_changed:
                flush()
        current.append(line)
    flush()
    return elements


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    header, *body = cleaned
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(out)


def _ocr_text(image: Image.Image) -> tuple[str, float | None]:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences = []
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if not word.strip() or conf < MIN_OCR_CONFIDENCE:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confidences.append(conf)
    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    return text, statistics.fmean(confidences) if confidences else None


def _title(elements: list[Element], path: Path) -> str:
    for element in elements:
        if element.page == 1 and element.kind == "heading":
            return element.text
    first = next((e for e in elements if e.page == 1), None)
    return first.text.split("\n")[0][:120] if first else path.stem
