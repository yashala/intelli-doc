import { useEffect, useRef, useState, type FormEvent } from "react";
import { api, MODES, type AskResponse, type Citation, type Mode } from "../api";
import AnswerCard from "./AnswerCard";

export interface Turn {
  id: number;
  question: string;
  mode: Mode;
  scope: number;
  result?: AskResponse;
  error?: string;
}

interface Props {
  turns: Turn[];
  setTurns: (update: (turns: Turn[]) => Turn[]) => void;
  selectedDocs: string[];
  hasDocuments: boolean;
  activeCitation: Citation | null;
  onOpenCitation: (citation: Citation) => void;
}

export default function Conversation(props: Props) {
  const { turns, setTurns, selectedDocs, hasDocuments, activeCitation, onOpenCitation } = props;
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("hybrid");
  const pending = turns.some((t) => !t.result && !t.error);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    end.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [turns]);

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const text = question.trim();
    if (!text || pending || !hasDocuments) return;
    const id = Date.now();
    setTurns((all) => [...all, { id, question: text, mode, scope: selectedDocs.length }]);
    setQuestion("");
    try {
      const result = await api.ask(text, mode, selectedDocs);
      setTurns((all) => all.map((t) => (t.id === id ? { ...t, result } : t)));
      if (result.citations[0]) onOpenCitation(result.citations[0]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Request failed";
      setTurns((all) => all.map((t) => (t.id === id ? { ...t, error: message } : t)));
    }
  }

  return (
    <section className="conversation panel" aria-labelledby="ask-heading">
      <div className="panel-head">
        <h2 id="ask-heading">Ask</h2>
        {turns.length > 0 && (
          <button type="button" className="link" onClick={() => setTurns(() => [])}>
            Clear conversation
          </button>
        )}
      </div>

      <div className="thread" aria-live="polite">
        {turns.length === 0 && (
          <div className="empty-state">
            <p>
              {hasDocuments
                ? "Ask a question about your documents. Answers cite the pages they come from, and the system says so when the documents do not contain an answer."
                : "Upload a PDF to start asking questions."}
            </p>
          </div>
        )}
        {turns.map((turn) => (
          <div key={turn.id} className="turn">
            <p className="question">
              {turn.question}
              <span className="muted">
                {" "}
                · {MODES.find((m) => m.value === turn.mode)?.label}
                {turn.scope > 0 && ` · ${turn.scope} document${turn.scope > 1 ? "s" : ""}`}
              </span>
            </p>
            {!turn.result && !turn.error && (
              <div className="thinking">
                <span className="dot" /> Retrieving and answering…
              </div>
            )}
            {turn.error && <p className="error">Request failed: {turn.error}</p>}
            {turn.result && (
              <AnswerCard result={turn.result} activeCitation={activeCitation} onOpenCitation={onOpenCitation} />
            )}
          </div>
        ))}
        <div ref={end} />
      </div>

      <form className="composer" onSubmit={submit}>
        <label htmlFor="question" className="visually-hidden">
          Question
        </label>
        <textarea
          id="question"
          value={question}
          maxLength={1000}
          rows={2}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void submit();
            }
          }}
          placeholder={hasDocuments ? "Ask a question… (Enter to send, Shift+Enter for a new line)" : "Upload a document first"}
          disabled={!hasDocuments}
        />
        <div className="composer-row">
          <div className="segmented" role="radiogroup" aria-label="Retrieval mode">
            {MODES.map((m) => (
              <button
                key={m.value}
                type="button"
                role="radio"
                aria-checked={mode === m.value}
                title={m.hint}
                onClick={() => setMode(m.value)}
              >
                {m.label}
              </button>
            ))}
          </div>
          <button type="submit" className="primary" disabled={pending || !question.trim() || !hasDocuments}>
            {pending ? "Answering…" : "Ask"}
          </button>
        </div>
      </form>
    </section>
  );
}
