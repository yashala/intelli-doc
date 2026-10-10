import { useEffect, useRef, useState } from "react";
import { api, type Citation } from "../api";

interface Props {
  citation: Citation;
  pageCount: number | undefined;
  onClose: () => void;
}

export default function SourceViewer({ citation, pageCount, onClose }: Props) {
  const first = citation.pages[0] ?? 1;
  const [page, setPage] = useState(first);
  const [zoomed, setZoomed] = useState(false);
  const [loading, setLoading] = useState(true);
  const closeButton = useRef<HTMLButtonElement>(null);
  const last = pageCount ?? Math.max(...citation.pages);

  useEffect(() => {
    setPage(citation.pages[0] ?? 1);
    closeButton.current?.focus();
  }, [citation]);

  useEffect(() => setLoading(true), [page, citation.doc_id]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement) return;
      if (event.key === "Escape") onClose();
      if (event.key === "ArrowLeft") setPage((p) => Math.max(1, p - 1));
      if (event.key === "ArrowRight") setPage((p) => Math.min(last, p + 1));
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [last, onClose]);

  const cited = citation.pages.includes(page);

  return (
    <section className="viewer panel" aria-label="Source viewer">
      <header className="viewer-head">
        <div>
          <h2>{citation.title}</h2>
          <p className="muted">{citation.filename}</p>
        </div>
        <button ref={closeButton} type="button" className="ghost" onClick={onClose} aria-label="Close source viewer">
          ✕
        </button>
      </header>

      <blockquote className={cited ? "quote" : "quote dim"}>
        <span className="quote-label">Cited passage · page {citation.pages.join(", ")}</span>
        {citation.text}
      </blockquote>

      <div className="viewer-controls">
        <button type="button" className="ghost" disabled={page <= 1} onClick={() => setPage(page - 1)} aria-label="Previous page">
          ←
        </button>
        <span aria-live="polite">
          Page {page}
          {pageCount ? ` of ${pageCount}` : ""}
          {cited && <span className="badge accent">cited</span>}
        </span>
        <button type="button" className="ghost" disabled={page >= last} onClick={() => setPage(page + 1)} aria-label="Next page">
          →
        </button>
        <button type="button" className="ghost" aria-pressed={zoomed} onClick={() => setZoomed(!zoomed)}>
          {zoomed ? "Fit width" : "Zoom"}
        </button>
      </div>

      <div className={zoomed ? "page-frame zoomed" : "page-frame"}>
        {loading && <div className="page-loading">Rendering page…</div>}
        <img
          src={api.pageImageUrl(citation.doc_id, page)}
          alt={`Page ${page} of ${citation.filename}`}
          onLoad={() => setLoading(false)}
          onError={() => setLoading(false)}
        />
      </div>
      <p className="hint">← → to change page · Esc to close</p>
    </section>
  );
}
