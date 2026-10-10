import { REFUSAL_REASONS, totalMs, type AskResponse, type Citation } from "../api";

const STAGE_COLORS: Record<string, string> = {
  retrieve: "var(--c1)",
  generate: "var(--c2)",
  rewrite: "var(--c3)",
};

export function LatencyBar({ timings }: { timings: Record<string, number> }) {
  const total = totalMs(timings);
  if (total <= 0) return null;
  const entries = Object.entries(timings);
  return (
    <div className="latency" aria-label={`Total ${Math.round(total)} ms`}>
      <div className="latency-bar">
        {entries.map(([stage, ms]) => (
          <span
            key={stage}
            title={`${stage}: ${Math.round(ms)} ms`}
            style={{ width: `${(ms / total) * 100}%`, background: STAGE_COLORS[stage.split("_")[0]!] }}
          />
        ))}
      </div>
      <span className="latency-legend">
        {entries.map(([stage, ms]) => `${stage.replace("_0", "").replace("_1", " (retry)")} ${Math.round(ms)} ms`).join(" · ")}
      </span>
    </div>
  );
}

interface Props {
  result: AskResponse;
  activeCitation: Citation | null;
  onOpenCitation: (citation: Citation) => void;
  compact?: boolean;
}

export default function AnswerCard({ result, activeCitation, onOpenCitation, compact }: Props) {
  const tokens = result.usage.input_tokens + result.usage.output_tokens;
  return (
    <article className={result.refused ? "answer refused" : "answer"}>
      <h3 className="answer-label">{result.refused ? "No answer" : "Answer"}</h3>
      {/* Rendered as text, never as HTML or Markdown: model output is untrusted. */}
      <p className="answer-text">{result.answer}</p>
      {result.refused && result.refusal_reason && (
        <p className="muted">{REFUSAL_REASONS[result.refusal_reason] ?? result.refusal_reason}</p>
      )}
      {result.citations.length > 0 && (
        <ul className="chips" aria-label="Sources">
          {result.citations.map((c) => (
            <li key={`${c.label}-${c.doc_id}-${c.pages.join()}`}>
              <button
                type="button"
                className="chip"
                aria-pressed={activeCitation === c}
                onClick={() => onOpenCitation(c)}
              >
                <span className="chip-title">{c.title}</span>
                <span className="chip-page">p. {c.pages.join(", ")}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {!compact && <LatencyBar timings={result.timings_ms} />}
      <p className="meta">
        {result.model} · {result.config}
        {compact && ` · ${Math.round(totalMs(result.timings_ms))} ms`}
        {tokens > 0 && ` · ${tokens.toLocaleString()} tokens`}
      </p>
    </article>
  );
}
