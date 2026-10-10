import numpy as np
import pytest
from PIL import Image

from intellidoc.ingest import ocr
from tests import pdf_factory as f


def test_long_runs_marks_only_runs_of_required_length() -> None:
    row = np.array([[1, 1, 1, 0, 1, 1, 1, 1, 1, 0]], dtype=bool)
    marked = ocr._long_runs(row, 4, axis=1)
    assert marked.tolist() == [[False, False, False, False, True, True, True, True, True, False]]


@pytest.mark.parametrize("angle", [-1.5, 0.8])
def test_estimate_skew_recovers_rotation(angle: float) -> None:
    lines = [f"Line {i} of a scanned maintenance memo with text" for i in range(12)]
    page = f.text_image(lines, size=36, width=2000).convert("L")
    rotated = page.rotate(angle, fillcolor=255, expand=False)
    # Rotating back by the estimate should undo the skew.
    assert ocr.estimate_skew(rotated) == pytest.approx(-angle, abs=0.2)


def test_scanned_grid_table_is_rebuilt_row_by_row() -> None:
    rows = [["Branch", "Loans"], ["Cedar Hill", "88,210"], ["Fox Run", "41,975"]]
    image = f.scanned_grid(rows, rotate=0.7)
    elements, confidence = ocr.ocr_page(Image.merge("RGB", [image] * 3), page=1, dpi=300)
    tables = [e for e in elements if e.kind == "table"]
    assert len(tables) == 1
    assert tables[0].text == (
        "| Branch | Loans |\n|---|---|\n| Cedar Hill | 88,210 |\n| Fox Run | 41,975 |"
    )
    assert confidence is not None and confidence > 50


def test_larger_single_line_is_marked_as_heading() -> None:
    page = Image.new("RGB", (2000, 1200), "white")
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(page)
    draw.text((100, 100), "Imaging", fill="black", font=ImageFont.load_default(size=72))
    for i, line in enumerate(
        ["The linear ultrasound probe L12 was replaced", "under a service agreement."]
    ):
        draw.text((100, 300 + 60 * i), line, fill="black", font=ImageFont.load_default(size=40))
    elements, _ = ocr.ocr_page(page, page=1, dpi=300)
    assert [(e.kind, e.text.split()[0]) for e in elements] == [
        ("heading", "Imaging"),
        ("text", "The"),
    ]


def _page_with(draw_fn) -> Image.Image:  # type: ignore[no-untyped-def]
    from PIL import ImageDraw

    page = Image.new("L", (2550, 3300), 255)
    draw_fn(ImageDraw.Draw(page))
    return Image.merge("RGB", [page] * 3)


def test_page_frame_is_not_mistaken_for_a_table() -> None:
    # Regression: a scanned cover page's frame was read as a grid and its text discarded.
    from PIL import ImageFont

    def draw(d):  # type: ignore[no-untyped-def]
        d.rectangle((150, 150, 2400, 3150), outline=0, width=4)
        d.line((150, 2900, 2400, 2900), fill=0, width=4)
        d.line((1300, 2900, 1300, 3150), fill=0, width=4)
        d.text((400, 1200), "Langley Research Center", fill=0, font=ImageFont.load_default(60))

    elements, _ = ocr.ocr_page(_page_with(draw), page=1, dpi=300)
    assert not [e for e in elements if e.kind == "table"]
    assert "Langley Research Center" in " ".join(e.text for e in elements)


def test_table_without_row_rules_falls_back_to_text_ocr() -> None:
    # Regression: TN D-233 Table I has a ruled box and header but no rules between rows.
    from PIL import ImageFont

    rows = [("N.", "NE.", "0 to 90"), ("S.", "SE.", "90 to 180"), ("N.", "NW.", "270 to 360")]

    def draw(d):  # type: ignore[no-untyped-def]
        font = ImageFont.load_default(44)
        d.rectangle((300, 500, 2100, 1300), outline=0, width=4)
        d.line((300, 620, 2100, 620), fill=0, width=4)
        for x in (900, 1500):
            d.line((x, 500, x, 1300), fill=0, width=4)
        for x, title in ((340, "Latitude"), (940, "Direction"), (1540, "Azimuth")):
            d.text((x, 540), title, fill=0, font=font)
        for i, row in enumerate(rows):
            for x, value in zip((340, 940, 1540), row, strict=True):
                d.text((x, 680 + 90 * i), value, fill=0, font=font)

    elements, _ = ocr.ocr_page(_page_with(draw), page=1, dpi=300)
    text = " ".join(e.text for e in elements)
    assert "0 to 90" in text
    assert "270 to 360" in text


@pytest.mark.parametrize(
    ("text", "larger", "conf", "expected"),
    [
        ("CONCLUDING REMARKS", False, 95, True),  # body-height capitals
        ("Imaging", True, 92, True),
        ("(1b)", True, 90, False),  # equation number
        ("won th", True, 60, False),  # margin stamp read as text
        ("Oo ON", True, 85, False),
        ("The satellite period in seconds is", False, 95, False),
    ],
)
def test_heading_rules(text: str, larger: bool, conf: float, expected: bool) -> None:
    assert ocr._looks_like_heading(text, larger, conf) is expected
