import { useState, type FormEvent } from "react";
import { api, MODES, type AskResponse, type Citation, type Mode } from "../api";
import AnswerCard from "./AnswerCard";

type Outcome = { status: "loading" } | { status: "done"; result: AskResponse } | { status: "error"; message: string };

interface Props {
  selectedDocs: string[];
  hasDocuments: boolean;
  activeCitation: Citation | null;
  onOpenCitation: (citation: Citation) => void;
}

// Runs one question through every retrieval mode, so the benchmark's comparison can be seen
// on a single question of your own.
export default function ComparePanel({ selectedDocs, hasDocuments, activeCitation, onOpenCitation }: Props) {
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  const [outcomes, setOutcomes] = useState<Partial<Record<Mode, Outcome>>>({});
  const busy = Object.values(outcomes).some((o) => o?.status === "loading");

  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || busy) return;
    setAsked(text);
    setOutcomes(Object.fromEntries(MODES.map((m) => [m.value, { status: "loading" }])));
    // Sequential: the reranker is CPU-heavy and parallel requests distort each other's latency.
    for (const { value } of MODES) {
      try {
        const result = await api.ask(text, value, selectedDocs);
        setOutcomes((all) => ({ ...all, [value]: { status: "done", result } }));
      } catch (err) {
        const message = err instanceof Error ? err.message : "Request failed";
        setOutcomes((all) => ({ ...all, [value]: { status: "error", message } }));
      }
    }
  }

  return (
    <section className="panel compare" aria-labelledby="compare-heading">
      <h2 id="compare-heading">Compare retrieval modes</h2>
      <p className="muted">
        One question, three retrieval strategies, run one after another. Differences in sources and
        latency show what each strategy trades.
      </p>
      <form className="compare-form" onSubmit={submit}>
        <label htmlFor="compare-question" className="visually-hidden">
          Question to compare
        </label>
        <input
          id="compare-question"
          value={question}
          maxLength={1000}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={hasDocuments ? "e.g. What is the standard deduction for head of household?" : "Upload a document first"}
          disabled={!hasDocuments}
        />
        <button type="submit" className="primary" disabled={busy || !question.trim() || !hasDocuments}>
          {busy ? "Running…" : "Compare"}
        </button>
      </form>
      {asked && <p className="question">{asked}</p>}
      <div className="compare-grid">
        {MODES.map((m) => {
          const outcome = outcomes[m.value];
          if (!outcome) return null;
          return (
            <div key={m.value} className="compare-col">
              <h3>{m.label}</h3>
              <p className="hint">{m.hint}</p>
              {outcome.status === "loading" && (
                <div className="thinking">
                  <span className="dot" /> Waiting…
                </div>
              )}
              {outcome.status === "error" && <p className="error">{outcome.message}</p>}
              {outcome.status === "done" && (
                <AnswerCard result={outcome.result} activeCitation={activeCitation} onOpenCitation={onOpenCitation} compact />
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
