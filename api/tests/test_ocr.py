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
