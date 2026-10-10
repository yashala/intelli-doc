import { useEffect, useRef, useState, type DragEvent } from "react";
import { api } from "../api";

interface Job {
  id: number;
  name: string;
  status: "queued" | "processing" | "done" | "error";
  started?: number;
  seconds?: number;
  message?: string;
}

// Uploads run one at a time: the API OCRs scanned pages inside the request, and parallel
// uploads would compete for the same CPU.
export default function UploadDropzone({ onUploaded }: { onUploaded: () => void }) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [dragging, setDragging] = useState(false);
  const [now, setNow] = useState(Date.now());
  const queue = useRef<{ id: number; file: File }[]>([]);
  const running = useRef(false);
  const nextId = useRef(1);
  const input = useRef<HTMLInputElement>(null);

  const active = jobs.some((j) => j.status === "processing");
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(timer);
  }, [active]);

  function update(id: number, patch: Partial<Job>) {
    setJobs((all) => all.map((j) => (j.id === id ? { ...j, ...patch } : j)));
  }

  async function drain() {
    if (running.current) return;
    running.current = true;
    while (queue.current.length) {
      const { id, file } = queue.current.shift()!;
      const started = Date.now();
      update(id, { status: "processing", started });
      try {
        const result = await api.upload(file);
        const ocr = result.ocr_pages ? `, ${result.ocr_pages} OCR` : "";
        update(id, {
          status: "done",
          seconds: (Date.now() - started) / 1000,
          message: `${result.pages} pages${ocr}`,
        });
        onUploaded();
      } catch (err) {
        update(id, { status: "error", message: err instanceof Error ? err.message : "Upload failed" });
      }
    }
    running.current = false;
  }

  function add(files: FileList | File[]) {
    const pdfs = Array.from(files).filter((f) => f.name.toLowerCase().endsWith(".pdf") || f.type === "application/pdf");
    const added = pdfs.map((file) => ({ id: nextId.current++, file }));
    if (!added.length) return;
    queue.current.push(...added);
    setJobs((all) => [...added.map(({ id, file }) => ({ id, name: file.name, status: "queued" as const })), ...all].slice(0, 8));
    void drain();
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    add(event.dataTransfer.files);
  }

  return (
    <div className="upload">
      <div
        className={dragging ? "dropzone dragging" : "dropzone"}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <p>
          <strong>Drop PDFs here</strong> or{" "}
          <button type="button" className="link" onClick={() => input.current?.click()}>
            browse
          </button>
        </p>
        <p className="hint">Native, scanned and mixed PDFs, up to 25 MB. Scans take a few seconds per page.</p>
        <input
          ref={input}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          aria-label="Choose PDF files"
          onChange={(e) => {
            if (e.target.files) add(e.target.files);
            e.target.value = "";
          }}
        />
      </div>
      {jobs.length > 0 && (
        <ul className="jobs" aria-live="polite">
          {jobs.map((job) => (
            <li key={job.id} className={`job ${job.status}`}>
              <span className="job-name">{job.name}</span>
              <span className="job-status">
                {job.status === "queued" && "Queued"}
                {job.status === "processing" && `Indexing… ${Math.round((now - (job.started ?? now)) / 1000)} s`}
                {job.status === "done" && `${job.message} · ${job.seconds?.toFixed(0)} s`}
                {job.status === "error" && job.message}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
