import { useState, type ChangeEvent } from "react";
import { api, type DocumentInfo, type IngestResult } from "../api";

interface Props {
  documents: DocumentInfo[] | null;
  error: string | null;
  onUploaded: () => void;
}

export default function DocumentsPanel({ documents, error, onUploaded }: Props) {
  const [uploading, setUploading] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<IngestResult | null>(null);

  async function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setUploading(file.name);
    setUploadError(null);
    setLastResult(null);
    try {
      setLastResult(await api.upload(file));
      onUploaded();
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(null);
    }
  }

  return (
    <section className="panel documents" aria-labelledby="documents-heading">
      <h2 id="documents-heading">Documents</h2>
      <label className="upload">
        <span>Add a PDF</span>
        <input type="file" accept="application/pdf,.pdf" onChange={handleFile} disabled={!!uploading} />
      </label>
      <div aria-live="polite">
        {uploading && (
          <p className="note">
            Processing {uploading}. Scanned pages are OCR'd and take a few seconds each.
          </p>
        )}
        {uploadError && <p className="error">Upload failed: {uploadError}</p>}
        {lastResult && (
          <p className="note">
            Indexed “{lastResult.title}”: {lastResult.pages} pages ({lastResult.ocr_pages} OCR),{" "}
            {lastResult.chunks.structure ?? 0} chunks.
          </p>
        )}
      </div>

      {error && <p className="error">Could not load documents: {error}</p>}
      {documents === null && !error && <p className="note">Loading…</p>}
      {documents?.length === 0 && <p className="empty">No documents yet. Upload a PDF to start.</p>}
      {documents && documents.length > 0 && (
        <ul className="doc-list">
          {documents.map((doc) => (
            <li key={doc.id}>
              <strong>{doc.title}</strong>
              <span>
                {doc.filename} · {doc.page_count} pages
                {doc.ocr_pages > 0 && ` · ${doc.ocr_pages} OCR`}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
