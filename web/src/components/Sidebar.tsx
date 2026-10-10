import { useMemo, useState } from "react";
import type { DocumentInfo } from "../api";
import UploadDropzone from "./UploadDropzone";

interface Props {
  documents: DocumentInfo[] | null;
  error: string | null;
  selected: string[];
  onSelect: (ids: string[]) => void;
  onUploaded: () => void;
}

export default function Sidebar({ documents, error, selected, onSelect, onUploaded }: Props) {
  const [filter, setFilter] = useState("");
  const visible = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!documents || !q) return documents ?? [];
    return documents.filter((d) => `${d.title} ${d.filename}`.toLowerCase().includes(q));
  }, [documents, filter]);

  function toggle(id: string) {
    onSelect(selected.includes(id) ? selected.filter((s) => s !== id) : [...selected, id]);
  }

  return (
    <aside className="sidebar panel" aria-labelledby="documents-heading">
      <h2 id="documents-heading">Documents</h2>
      <UploadDropzone onUploaded={onUploaded} />

      {error && <p className="error">Could not load documents: {error}</p>}
      {documents === null && !error && <p className="muted">Loading…</p>}
      {documents?.length === 0 && <p className="empty">No documents yet. Drop a PDF above to start.</p>}

      {documents && documents.length > 0 && (
        <>
          <div className="doc-tools">
            <input
              type="search"
              placeholder="Filter documents"
              aria-label="Filter documents"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
            <p className="scope" aria-live="polite">
              {selected.length === 0 ? "Asking across all documents" : `Asking within ${selected.length} selected`}
              {selected.length > 0 && (
                <button type="button" className="link" onClick={() => onSelect([])}>
                  clear
                </button>
              )}
            </p>
          </div>
          <ul className="doc-list">
            {visible.map((doc) => (
              <li key={doc.id}>
                <label className={selected.includes(doc.id) ? "doc selected" : "doc"}>
                  <input type="checkbox" checked={selected.includes(doc.id)} onChange={() => toggle(doc.id)} />
                  <span>
                    <strong>{doc.title}</strong>
                    <span className="muted">
                      {doc.filename} · {doc.page_count} p.
                      {doc.ocr_pages > 0 && <span className="badge">OCR {doc.ocr_pages}</span>}
                    </span>
                  </span>
                </label>
              </li>
            ))}
            {visible.length === 0 && <li className="muted">No documents match “{filter}”.</li>}
          </ul>
        </>
      )}
    </aside>
  );
}
