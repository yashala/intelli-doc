"""OCR for scanned pages, with grid-table reconstruction.

Tesseract's layout analysis breaks on ruled tables: grid lines get read as characters and
cells are emitted column-by-column, separating labels from their values. When a scanned
page contains long horizontal and vertical rules, we locate the grid, OCR each cell on its
own, and rebuild the table as Markdown. The rest of the page is OCR'd with the rules erased.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
import pytesseract
from numpy.typing import NDArray
from PIL import Image

from intellidoc.models import Element, ElementKind

DARK_THRESHOLD = 150
MIN_OCR_CONFIDENCE = 30.0
MIN_PARAGRAPH_CONFIDENCE = 50.0
MIN_HEADING_CONFIDENCE = 80.0
# Grid acceptance checks for reconstructed tables.
MAX_ROW_HEIGHT_IN = 0.6
MIN_TABLE_CONFIDENCE = 60.0
MIN_FILLED_CELLS = 0.5

BoolArray = NDArray[np.bool_]


@dataclass
class Grid:
    top: int
    bottom: int
    rows: list[tuple[int, int]]
    columns: list[tuple[int, int]]


def ocr_page(image: Image.Image, page: int, dpi: int) -> tuple[list[Element], float | None]:
    image = deskew(image.convert("L"))
    gray = np.asarray(image)
    dark = gray < DARK_THRESHOLD
    min_run = dpi // 2  # rules are at least half an inch long
    bridged = _dilate(dark)  # scan noise leaves 1px gaps in rules
    horizontal = _long_runs(bridged, min_run, axis=1) & dark
    vertical = _long_runs(bridged, min_run, axis=0) & dark

    scale = 72 / dpi
    elements: list[tuple[int, Element]] = []
    confidences: list[float] = []

    # Grid candidates are checked before they are trusted: page frames, margin stamps and
    # tables without row rules also produce long lines (seen on NASA TN D-233 pages 1 and 26).
    # A rejected candidate is left for paragraph OCR with its rules erased, because rules
    # split Tesseract's layout into narrow strips (TN D-233 Table I was unreadable until then).
    text_layer = gray.copy()
    rules = horizontal | vertical
    for grid in _find_grids(horizontal, vertical):
        accepted = False
        if max(y1 - y0 for y0, y1 in grid.rows) <= MAX_ROW_HEIGHT_IN * dpi:
            markdown, cell_conf, filled = _read_grid(gray, grid)
            mean = statistics.fmean(cell_conf) if cell_conf else 0.0
            accepted = (
                bool(markdown) and mean >= MIN_TABLE_CONFIDENCE and filled >= MIN_FILLED_CELLS
            )
        if not accepted:
            region = text_layer[grid.top : grid.bottom + 1]
            region[rules[grid.top : grid.bottom + 1]] = 255
            continue
        text_layer[grid.top : grid.bottom + 1, :] = 255
        bbox = (
            grid.columns[0][0] * scale,
            grid.top * scale,
            grid.columns[-1][1] * scale,
            grid.bottom * scale,
        )
        elements.append((grid.top, Element(page, "table", markdown, "ocr", bbox, mean)))
        confidences += cell_conf
    source = Image.fromarray(text_layer)

    for top, element, conf in _paragraphs(source, page, scale):
        elements.append((top, element))
        confidences += conf

    elements.sort(key=lambda item: item[0])
    return [e for _, e in elements], statistics.fmean(confidences) if confidences else None


def estimate_skew(image: Image.Image, max_angle: float = 3.0, step: float = 0.1) -> float:
    """Return the rotation (degrees) that makes text lines most horizontal.

    Projection-profile method: at the correct angle, rows of text and blank space alternate
    sharply, so the variance of the per-row dark-pixel count is maximal.
    """
    small = image.convert("L").reduce(4)
    binary = Image.fromarray(np.where(np.asarray(small) < DARK_THRESHOLD, 255, 0).astype(np.uint8))
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step / 2, step):
        rotated = np.asarray(binary.rotate(float(angle), resample=Image.Resampling.NEAREST))
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return round(best_angle, 2)


def deskew(image: Image.Image) -> Image.Image:
    angle = estimate_skew(image)
    if abs(angle) < 0.05:
        return image
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)


def _dilate(mask: BoolArray) -> BoolArray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _long_runs(dark: BoolArray, length: int, axis: int) -> BoolArray:
    """Mark pixels that lie inside a run of at least `length` dark pixels along `axis`."""
    d = np.moveaxis(dark, axis, -1).astype(np.int32)
    pad = np.zeros((*d.shape[:-1], 1), np.int32)
    csum = np.concatenate([pad, np.cumsum(d, axis=-1)], axis=-1)
    full = (csum[..., length:] - csum[..., :-length]) == length  # window starting here is dark
    fsum = np.concatenate([pad, np.cumsum(full, axis=-1)], axis=-1)
    idx = np.arange(d.shape[-1])
    lo = np.clip(idx - length + 1, 0, full.shape[-1])
    hi = np.clip(idx + 1, 0, full.shape[-1])
    covered = (fsum[..., hi] - fsum[..., lo]) > 0
    return np.moveaxis(covered, -1, axis)


def _clusters(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Contiguous True runs in a 1-D mask as (start, end) inclusive."""
    runs = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _find_grids(horizontal: BoolArray, vertical: BoolArray) -> list[Grid]:
    rules = _clusters(horizontal.any(axis=1))
    grids: list[Grid] = []
    band_start = 0
    while band_start < len(rules) - 1:
        # Extend the table downward while consecutive rules are joined by vertical lines.
        columns: list[tuple[int, int]] = []
        end = band_start
        while end + 1 < len(rules):
            y0, y1 = rules[end][1] + 1, rules[end + 1][0] - 1
            if y1 <= y0:
                break
            band_columns = _column_rules(vertical[y0 : y1 + 1])
            if len(band_columns) < 2 or (columns and len(band_columns) != len(columns)):
                break
            columns = band_columns
            end += 1
        if end > band_start:
            rows = [(rules[i][1] + 1, rules[i + 1][0] - 1) for i in range(band_start, end)]
            cells = [(columns[i][1] + 1, columns[i + 1][0] - 1) for i in range(len(columns) - 1)]
            grids.append(Grid(rules[band_start][0], rules[end][1], rows, cells))
        band_start = max(end, band_start + 1)
    return grids


def _column_rules(band: BoolArray) -> list[tuple[int, int]]:
    """Vertical rules that span most of a row band's height."""
    return _clusters(band.mean(axis=0) > 0.5)


def _read_grid(gray: NDArray[np.uint8], grid: Grid) -> tuple[str, list[float], float]:
    """Return the table as Markdown, per-word confidences, and the fraction of filled cells."""
    table: list[list[str]] = []
    confidences: list[float] = []
    inset = 4
    for y0, y1 in grid.rows:
        row = []
        for x0, x1 in grid.columns:
            cell = gray[y0 + inset : y1 - inset, x0 + inset : x1 - inset]
            if cell.size == 0:
                row.append("")
                continue
            text, conf = _ocr_words(Image.fromarray(cell), "--psm 7")
            row.append(text)
            confidences += conf
        table.append(row)
    cells = [cell for row in table for cell in row]
    filled = sum(1 for cell in cells if cell) / len(cells) if cells else 0.0
    table = [row for row in table if any(row)]
    if not table:
        return "", confidences, filled
    header, *body = table
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines), confidences, filled


def _ocr_words(image: Image.Image, config: str) -> tuple[str, list[float]]:
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    words, conf = [], []
    for word, c in zip(data["text"], data["conf"], strict=True):
        if word.strip() and float(c) >= MIN_OCR_CONFIDENCE:
            words.append(word)
            conf.append(float(c))
    return " ".join(words), conf


def _paragraphs(
    image: Image.Image, page: int, scale: float
) -> list[tuple[int, Element, list[float]]]:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    groups: dict[tuple[int, int], list[int]] = {}
    for i, word in enumerate(data["text"]):
        if word.strip() and float(data["conf"][i]) >= MIN_OCR_CONFIDENCE:
            groups.setdefault((data["block_num"][i], data["par_num"][i]), []).append(i)

    heights = [data["height"][i] for indices in groups.values() for i in indices]
    typical = statistics.median(heights) if heights else 0
    out = []
    for indices in groups.values():
        lines: dict[int, list[str]] = {}
        for i in indices:
            lines.setdefault(data["line_num"][i], []).append(data["text"][i])
        text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
        conf = [float(data["conf"][i]) for i in indices]
        top = min(data["top"][i] for i in indices)
        bbox = (
            min(data["left"][i] for i in indices) * scale,
            top * scale,
            max(data["left"][i] + data["width"][i] for i in indices) * scale,
            max(data["top"][i] + data["height"][i] for i in indices) * scale,
        )
        mean_conf = statistics.fmean(conf)
        if sum(ch.isalnum() for ch in text) < 3 or mean_conf < MIN_PARAGRAPH_CONFIDENCE:
            continue  # specks, margin stamps, figure fragments
        larger = statistics.median(data["height"][i] for i in indices) >= 1.25 * typical
        kind: ElementKind = (
            "heading"
            if len(lines) == 1 and _looks_like_heading(text, larger, mean_conf)
            else "text"
        )
        out.append((top, Element(page, kind, text, "ocr", bbox, mean_conf), conf))
    return out


def _looks_like_heading(text: str, larger: bool, confidence: float) -> bool:
    """A heading is a clean line of real words, in larger type or short and all capitals.

    On NASA TN D-233, size alone promoted equation numbers ("(1b)") and margin stamps
    ("won th") to headings, splitting the document into tiny chunks, while the real headings
    ("CONCLUDING REMARKS") are body-height capitals.
    """
    letters = [ch for ch in text if not ch.isspace()]
    words = text.split()
    clean = (
        confidence >= MIN_HEADING_CONFIDENCE
        and sum(ch.isalpha() for ch in letters) >= 0.7 * len(letters)
        and any(len(w) >= 3 and w.isalpha() for w in words)
    )
    short_caps = text.isupper() and len(words) <= 8
    return clean and (larger or short_caps)
