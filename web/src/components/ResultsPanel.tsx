import { useEffect, useState } from "react";
import { api, type BenchmarkRun } from "../api";

const pct = (v: number | null) => (v === null ? "n/a" : `${(v * 100).toFixed(1)}%`);
const ms = (v: number | null | undefined) => (v == null ? "n/a" : `${Math.round(v)}`);

const COLUMNS: [string, keyof BenchmarkRun["summary"][string]][] = [
  ["Evidence hit@5", "evidence_hit_rate"],
  ["MRR", "mrr"],
  ["Answer accuracy", "answer_accuracy"],
  ["Citation precision", "citation_precision"],
  ["False refusal", "false_refusal_rate"],
  ["Correct refusal", "correct_refusal_rate"],
  ["Injection followed", "injection_followed_rate"],
];

export default function ResultsPanel() {
  const [runs, setRuns] = useState<BenchmarkRun[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .results()
      .then(setRuns)
      .catch((err: Error) => setError(err.message));
  }, []);

  if (error) return <p className="error">Could not load results: {error}</p>;
  if (runs === null) return <p className="note">Loading…</p>;
  if (runs.length === 0)
    return <p className="empty">No benchmark runs found. Run `make eval-offline` to create one.</p>;

  return (
    <div className="results">
      {runs.map((run) => (
        <section key={run.run_id} className="panel" aria-labelledby={`run-${run.run_id}`}>
          <h2 id={`run-${run.run_id}`}>
            {run.mode === "offline" ? "Offline (extractive reader)" : `Online (${run.answer_model})`}
          </h2>
          <p className="meta">
            Run {run.run_id} on {run.cpu}, {run.memory_gb} GB RAM. Numbers come from the committed
            run artifacts; see docs/evaluation.md for definitions.
          </p>
          <div className="table-wrap">
            <table>
              <caption>Retrieval, answer and refusal metrics by configuration</caption>
              <thead>
                <tr>
                  <th scope="col">Config</th>
                  {COLUMNS.map(([label]) => (
                    <th scope="col" key={label}>
                      {label}
                    </th>
                  ))}
                  <th scope="col">Latency p50 / p95 ms</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(run.summary).map(([name, s]) => (
                  <tr key={name}>
                    <th scope="row" title={s.config.description}>
                      {name}
                    </th>
                    {COLUMNS.map(([label, key]) => {
                      const value = s[key] as number | null;
                      return <td key={label}>{key === "mrr" ? (value?.toFixed(3) ?? "n/a") : pct(value)}</td>;
                    })}
                    <td>
                      {ms(s.latency_ms.total?.p50)} / {ms(s.latency_ms.total?.p95)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </div>
  );
}
