from pathlib import Path

from reportlab.platypus import PageBreak

from intellidoc.ingest.parsing import parse_pdf
from tests import pdf_factory as f


def texts(doc_path: Path) -> list[tuple[int, str, str]]:
    return [(e.page, e.kind, e.text) for e in parse_pdf(doc_path).elements]


def test_elements_keep_their_page_numbers(tmp_path: Path) -> None:
    path = f.build(
        tmp_path / "two_pages.pdf",
        [f.para("Alpha section text."), PageBreak(), f.para("Beta section text.")],
    )
    elements = texts(path)
    assert (1, "text", "Alpha section text.") in elements
    assert (2, "text", "Beta section text.") in elements


def test_heading_is_distinguished_from_body(tmp_path: Path) -> None:
    path = f.build(tmp_path / "h.pdf", [f.para("Results", "Heading2"), f.para("The value was 42.")])
    assert [(k, t) for _, k, t in texts(path)] == [
        ("heading", "Results"),
        ("text", "The value was 42."),
    ]


def test_two_column_page_is_read_column_by_column(tmp_path: Path) -> None:
    left = [f.para(f"Left column sentence number {i} about ferries.") for i in range(6)]
    right = [f.para(f"Right column sentence number {i} about parking.") for i in range(6)]
    path = f.two_column(tmp_path / "cols.pdf", left, right)
    joined = " ".join(t for _, _, t in texts(path))
    assert "Left column sentence number 5" in joined
    assert joined.index("Left column sentence number 5") < joined.index("Right column sentence")
    # No element mixes text from both columns.
    assert not any("Left" in t and "Right" in t for _, _, t in texts(path))


def test_short_single_column_paragraph_is_not_split(tmp_path: Path) -> None:
    # Regression: aligned word gaps in a 3-line paragraph were mistaken for a column gutter.
    sentence = (
        "The operating budget for 2025 was 48.2 million dollars. The largest 2026 capital "
        "project is the Northshore reservoir liner replacement, estimated at 9.6 million "
        "dollars. A rate increase of 3.5 percent takes effect on July 1, 2026."
    )
    path = f.build(tmp_path / "p.pdf", [f.para("Finance", "Heading2"), f.para(sentence)])
    assert any(t == sentence for _, _, t in texts(path))


def test_native_table_is_converted_to_markdown(tmp_path: Path) -> None:
    rows = [["Route", "Boardings"], ["12", "241,905"], ["19", "54,880"]]
    path = f.build(tmp_path / "t.pdf", [f.grid_table(rows)])
    tables = [t for _, k, t in texts(path) if k == "table"]
    assert tables == ["| Route | Boardings |\n|---|---|\n| 12 | 241,905 |\n| 19 | 54,880 |"]


def test_image_only_page_is_detected_as_scanned_and_ocrd(tmp_path: Path) -> None:
    img = f.text_image(["Autoclave unit AC-2 failed", "on 2025-08-14"])
    path = f.image_only_pdf(tmp_path / "scan.pdf", [img])
    parsed = parse_pdf(path)
    assert parsed.pages[0].method == "ocr"
    joined = " ".join(e.text for e in parsed.elements)
    assert "AC-2" in joined
    assert "2025-08-14" in joined


def test_text_inside_embedded_image_is_extracted(tmp_path: Path) -> None:
    img = f.text_image(["Transformer T2 oil temperature 71 C"])
    path = f.build(tmp_path / "fig.pdf", [f.para("Figure appendix."), f.image_flowable(img, 5)])
    parsed = parse_pdf(path)
    assert parsed.pages[0].method == "native"
    image_text = [e for e in parsed.elements if e.kind == "image_text"]
    assert len(image_text) == 1
    assert "71" in image_text[0].text
    assert image_text[0].method == "ocr"
