# ADR 0003: pdfplumber, pypdfium2 and Tesseract with project-owned layout logic

Status: accepted (2026-10-08), chosen by the project owner

## Context
The corpus includes native PDFs, multi-column layouts, ruled tables, text inside images,
and scans with and without legacy OCR text layers. The machine has 8 GB of RAM.

## Decision
- pdfplumber for characters, words, tables and image positions on native pages.
- pypdfium2 to render pages for OCR.
- Tesseract 5 for OCR.
- Project code for: scanned-page detection, column gutter detection, heading/paragraph grouping,
  deskew, ruled-table reconstruction from scans, and grid validation.

## Alternatives considered
- Docling: stronger layout models, but 2-4 GB RAM and about 1.5 GB of models, and most layout
  decisions become opaque.
- PyMuPDF: fast and accurate, but AGPL-3.0, which would force the repository's license.
- Unstructured: similar trade-off to Docling, plus a large dependency tree.

## Consequences
- Every heuristic is visible, testable and was tuned against observed failures (see the
  engineering journal). Each one has a regression test.
- The heuristics were tuned on a small corpus. Layouts not represented here (rotated pages,
  handwritten forms, dense financial statements) are likely to need more work.
- OCR runs at 300 dpi and costs about 2.7 s per page on the development machine.
