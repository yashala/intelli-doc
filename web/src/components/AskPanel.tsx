import { useState, type FormEvent } from "react";
import { api, type AskResponse, type Citation } from "../api";

const REFUSAL_REASONS: Record<string, string> = {
  insufficient_evidence: "The retrieved passages did not contain the answer.",
  low_retrieval_similarity: "No passage was similar enough to the question.",
  no_valid_citation: "The model's answer did not cite any retrieved passage, so it was discarded.",
  no_documents: "No documents are indexed.",
};

const MODES = [
  { value: "hybrid", label: "Hybrid (dense + full text)" },
  { value: "dense", label: "Dense only" },
  { value: "hybrid_rerank", label: "Hybrid + cross-encoder rerank (slower)" },
];

export default function AskPanel({ hasDocuments }: { hasDocuments: boolean }) {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [selected, setSelected] = useState<Citation | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text) return;
    setLoading(true);
    setError(null);
    setSelected(null);
    try {
      const response = await api.ask(text, mode);
      setResult(response);
      setSelected(response.citations[0] ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel ask" aria-labelledby="ask-heading">
      <h2 id="ask-heading">Ask</h2>
      <form onSubmit={submit} className="ask-form">
        <label htmlFor="question">Question</label>
        <textarea
          id="question"
          value={question}
          maxLength={1000}
          rows={3}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) e.currentTarget.form?.requestSubmit();
          }}
          placeholder={hasDocuments ? "e.g. Which autoclave failed its spore test?" : "Upload a document first"}
        />
        <div className="form-row">
          <label htmlFor="mode">Retrieval</label>
          <select id="mode" value={mode} onChange={(e) => setMode(e.target.value)}>
            {MODES.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
          <button type="submit" disabled={loading || !question.trim() || !hasDocuments}>
            {loading ? "Answering…" : "Ask"}
          </button>
        </div>
      </form>

      <div aria-live="polite" className="answer-area">
        {error && <p className="error">Request failed: {error}</p>}
        {!result && !error && !loading && (
          <p className="empty">Answers appear here with the pages they are based on.</p>
        )}
        {result && (
          <article className={result.refused ? "answer refused" : "answer"}>
            <h3>{result.refused ? "No answer" : "Answer"}</h3>
            {/* Rendered as text, never as HTML or Markdown: model output is untrusted. */}
            <p className="answer-text">{result.answer}</p>
            {result.refused && result.refusal_reason && (
              <p className="note">{REFUSAL_REASONS[result.refusal_reason] ?? result.refusal_reason}</p>
            )}
            {result.citations.length > 0 && (
              <>
                <h4>Sources</h4>
                <ul className="citations">
                  {result.citations.map((c) => (
                    <li key={`${c.label}-${c.doc_id}`}>
                      <button
                        type="button"
                        aria-pressed={selected === c}
                        onClick={() => setSelected(c)}
                      >
                        {c.title} · p. {c.pages.join(", ")}
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
            <p className="meta">
              {result.model} · {result.config} · {Math.round(totalMs(result))} ms
              {result.usage.calls > 0 &&
                ` · ${result.usage.input_tokens + result.usage.output_tokens} tokens`}
            </p>
          </article>
        )}
        {selected && <CitationView citation={selected} />}
      </div>
    </section>
  );
}

function totalMs(result: AskResponse): number {
  return Object.values(result.timings_ms).reduce((a, b) => a + b, 0);
}

function CitationView({ citation }: { citation: Citation }) {
  return (
    <figure className="citation-view">
      <figcaption>
        {citation.filename}, page {citation.pages.join(", ")}
      </figcaption>
      <blockquote>{citation.text}</blockquote>
      {citation.pages.map((page) => (
        <img
          key={page}
          src={api.pageImageUrl(citation.doc_id, page)}
          alt={`Page ${page} of ${citation.filename}`}
          loading="lazy"
        />
      ))}
    </figure>
  );
}
