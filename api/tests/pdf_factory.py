"""Small PDF builders for tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)
from reportlab.platypus import Image as RLImage

STYLES = getSampleStyleSheet()


def build(path: Path, story: list[Flowable]) -> Path:
    SimpleDocTemplate(str(path), pagesize=LETTER).build(story)
    return path


def para(text: str, style: str = "BodyText") -> Paragraph:
    return Paragraph(text, STYLES[style])


def grid_table(rows: list[list[str]]) -> Table:
    t = Table(rows)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "black")]))
    return t


def two_column(path: Path, left: list[Flowable], right: list[Flowable]) -> Path:
    from reportlab.platypus import FrameBreak

    w, h = LETTER
    col = (w - 2 * inch - 0.3 * inch) / 2
    template = PageTemplate(
        frames=[
            Frame(inch, inch, col, h - 2 * inch),
            Frame(inch + col + 0.3 * inch, inch, col, h - 2 * inch),
        ]
    )
    BaseDocTemplate(str(path), pagesize=LETTER, pageTemplates=[template]).build(
        [*left, FrameBreak(), *right]
    )
    return path


def text_image(lines: list[str], size: int = 40, width: int = 1400) -> Image.Image:
    img = Image.new("RGB", (width, 120 + 80 * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((60, 60 + 80 * i), line, fill="black", font=ImageFont.load_default(size=size))
    return img


def image_flowable(img: Image.Image, width_in: float) -> RLImage:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=width_in * inch, height=width_in * inch * img.height / img.width)


def image_only_pdf(path: Path, pages: list[Image.Image]) -> Path:
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=150)
    return path


def scanned_grid(rows: list[list[str]], rotate: float = 0.0) -> Image.Image:
    """Draw a ruled table the way a 300 dpi scan of a printed table would look."""
    cell_w, cell_h = 600, 90
    img = Image.new("L", (2550, 3300), 255)
    draw = ImageDraw.Draw(img)
    x0, y0 = 300, 500
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            box = (x0 + c * cell_w, y0 + r * cell_h, x0 + (c + 1) * cell_w, y0 + (r + 1) * cell_h)
            draw.rectangle(box, outline=0, width=3)
            draw.text((box[0] + 20, box[1] + 22), value, fill=0, font=ImageFont.load_default(42))
    if rotate:
        img = img.rotate(rotate, fillcolor=255)
    return img
