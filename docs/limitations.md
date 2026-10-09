# Limitations

## Evaluation
- **Small benchmark.** 85 questions over 10 documents. One question is worth 1-2 percentage
  points; treat small differences between configurations as noise.
- **Synthetic documents are short.** Most are under one page of text. On them alone the
  fixed-window chunker produces one chunk per document, so chunking comparisons rely mainly on
  the two real documents.
- **Same author for system and answer key.** The real-document questions were written during
  development, with AI assistance, from page images. They have not been reviewed by anyone else.
- **String-match scoring.** Evidence is matched by normalised substring. A different table row
  containing the same value counts as a hit (seen on NASA TN D-233 Table I). Paraphrased numbers
  count as wrong.
- **Same-vendor judge.** The groundedness judge (`gpt-5-mini`) is from the same vendor as the
  answer model (`gpt-4.1-mini`); they may share blind spots.
- **Heuristics tuned on this corpus.** The gutter width (6 pt), grid acceptance thresholds and
  scanned-page rules were set while looking at these documents. Their test-split numbers are
  therefore optimistic for unseen layouts.
- **Latency depends on memory pressure.** The development machine has 8 GB RAM and was swapping
  heavily during the first benchmark run (swap use is recorded per run). Latency numbers from
  this machine are not representative of a server.

## Parsing and OCR
- OCR is English-only (Tesseract `eng`) at 300 dpi, about 2.7 s per page on an M1.
- Rotated pages (90/180 degrees), handwriting, forms with checkboxes, and nested or
  borderless tables in native PDFs are not handled specially.
- Charts are only understood through text that OCR can read inside them; bar heights, lines and
  colours carry no meaning to the system.
- Running headers and printer marks (for example IRS proof headers) are indexed as text.
- Math in scanned technical reports (TN D-233) OCRs poorly; questions about equations are not
  in the benchmark.

## Retrieval and answering
- Exact vector search; see ADR 0002 for what changes at scale.
- One query rewrite at most; multi-hop questions that need two separate retrievals are not
  supported and not in the benchmark.
- Injection handling is detection plus prompt instructions. Pattern detection is easy to evade
  (paraphrase, other languages, encoding). The model can still follow embedded instructions;
  the benchmark measures how often it does on a handful of cases.

## Service
- No authentication, authorisation or multi-tenancy. Intended for single-user local use.
- Uploaded PDFs are parsed in-process with no sandbox or resource limits beyond file size.
- No rate limiting, no request timeouts on the model call beyond the client library's retries.

## What production use would still require
Authentication and per-tenant isolation, sandboxed parsing workers with time and memory limits,
an ingestion queue, ANN indexing with measured recall, monitoring of refusal and injection rates
on real traffic, a reviewed and much larger evaluation set drawn from the target documents,
and a data-retention policy for uploaded files.
