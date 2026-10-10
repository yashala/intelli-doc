import { useEffect, useMemo, useState } from "react";
import { api, type BenchmarkRun, type MetricSummary } from "../api";

type MetricKey = keyof Pick<
  MetricSummary,
  | "evidence_hit_rate"
  | "mrr"
  | "answer_accuracy"
  | "citation_precision"
  | "judge_grounded_rate"
  | "correct_refusal_rate"
  | "false_refusal_rate"
>;

const METRICS: { key: MetricKey; label: string; help: string; lowerIsBetter?: boolean }[] = [
  { key: "evidence_hit_rate", label: "Evidence hit@5", help: "A top-5 chunk is on the expected page and contains the evidence." },
  { key: "answer_accuracy", label: "Answer accuracy", help: "The answer contains every evidence string; refusals count as wrong." },
  { key: "citation_precision", label: "Citation precision", help: "Share of cited chunks that are on an expected page." },
  { key: "judge_grounded_rate", label: "Judge grounded", help: "LLM judge: every claim is supported by the cited text (online runs only)." },
  { key: "correct_refusal_rate", label: "Correct refusal", help: "Unanswerable questions that were refused." },
  { key: "false_refusal_rate", label: "False refusal", help: "Answerable questions that were refused.", lowerIsBetter: true },
  { key: "mrr", label: "MRR", help: "Mean reciprocal rank of the first evidence hit." },
];

const FAILURES = ["parse_miss", "retrieval_miss", "false_refusal", "wrong_answer", "wrong_citation", "injection_followed", "missed_refusal"];
const FAILURE_COLORS = ["#9a9a92", "#e45756", "#f58518", "#eeca3b", "#54a24b", "#b279a2", "#4c78a8"];

const fmt = (key: MetricKey, v: number | null | undefined) =>
  v == null ? "n/a" : key === "mrr" ? v.toFixed(3) : `${(v * 100).toFixed(1)}%`;

export default function BenchmarkPanel() {
  const [runs, setRuns] = useState<BenchmarkRun[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [metric, setMetric] = useState<MetricKey>("evidence_hit_rate");

  useEffect(() => {
    api
      .results()
      .then((r) => {
        setRuns(r);
        const preferred = r.find((x) => x.mode !== "offline") ?? r[0];
        setRunId(preferred?.run_id ?? null);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  const run = runs?.find((r) => r.run_id === runId) ?? null;
  const configs = useMemo(() => (run ? Object.entries(run.summary) : []), [run]);
  const categories = useMemo(
    () => (configs[0] ? Object.keys(configs[0][1].by_category).filter((c) => c !== "unanswerable") : []),
    [configs],
  );

  if (error) return <p className="error">Could not load results: {error}</p>;
  if (runs === null) return <p className="muted">Loading…</p>;
  if (!run) return <p className="empty">No benchmark runs found. Run `make eval-offline` to create one.</p>;

  const current = METRICS.find((m) => m.key === metric)!;
  const values = configs.map(([, s]) => s[metric] ?? null);
  const best = current.lowerIsBetter
    ? Math.min(...values.filter((v): v is number => v !== null))
    : Math.max(...values.filter((v): v is number => v !== null));

  return (
    <div className="benchmark">
      <section className="panel">
        <div className="panel-head">
          <h2>Benchmark results</h2>
          <div className="segmented" role="radiogroup" aria-label="Benchmark run">
            {runs.map((r) => (
              <button key={r.run_id} type="button" role="radio" aria-checked={r.run_id === runId} onClick={() => setRunId(r.run_id)}>
                {r.mode === "offline" ? "Offline reader" : r.answer_model}
              </button>
            ))}
          </div>
        </div>
        <p className="muted">
          Run <code>{run.run_id}</code> on {run.cpu}, {run.memory_gb} GB RAM. 85 questions per configuration; numbers are read
          from the committed run artifacts.
          {run.mode === "offline" && " The offline reader is a deterministic sentence matcher, so answer accuracy is a floor, not an LLM measurement."}
        </p>

        <div className="metric-picker" role="radiogroup" aria-label="Metric">
          {METRICS.map((m) => (
            <button key={m.key} type="button" role="radio" aria-checked={metric === m.key} onClick={() => setMetric(m.key)} title={m.help}>
              {m.label}
            </button>
          ))}
        </div>
        <p className="hint">{current.help}</p>

        <div className="bars" role="list" aria-label={`${current.label} by configuration`}>
          {configs.map(([name, s], i) => {
            const v = values[i];
            const width = v == null ? 0 : metric === "mrr" ? v * 100 : v * 100;
            return (
              <div key={name} className="bar-row" role="listitem">
                <span className="bar-label" title={s.config.description}>
                  {name}
                </span>
                <span className="bar-track">
                  <span className={v === best ? "bar-fill best" : "bar-fill"} style={{ width: `${width}%` }} />
                </span>
                <span className="bar-value">{fmt(metric, v)}</span>
              </div>
            );
          })}
        </div>
      </section>

      <section className="panel">
        <h2>{current.label} by document type</h2>
        <div className="table-wrap">
          <table className="heatmap">
            <thead>
              <tr>
                <th scope="col">Config</th>
                {categories.map((c) => (
                  <th scope="col" key={c}>
                    {c}
                    <span className="muted"> n={configs[0]![1].by_category[c]!.answerable}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {configs.map(([name, s]) => (
                <tr key={name}>
                  <th scope="row">{name}</th>
                  {categories.map((c) => {
                    const v = s.by_category[c]?.[metric] ?? null;
                    const shade = v == null ? 0 : current.lowerIsBetter ? 1 - v : metric === "mrr" ? v : v;
                    return (
                      <td key={c} style={{ background: v == null ? undefined : `color-mix(in srgb, var(--good) ${Math.round(shade * 70)}%, transparent)` }}>
                        {fmt(metric, v)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="two-col">
        <section className="panel">
          <h2>Where questions fail</h2>
          <p className="hint">Each question gets at most one failure category, in pipeline order.</p>
          {configs.map(([name, s]) => {
            const total = s.questions;
            return (
              <div key={name} className="stack-row">
                <span className="bar-label">{name}</span>
                <span className="stack" aria-label={`${name} failures`}>
                  {FAILURES.map((f, i) =>
                    s.failures[f] ? (
                      <span key={f} title={`${f}: ${s.failures[f]}`} style={{ width: `${(s.failures[f]! / total) * 100}%`, background: FAILURE_COLORS[i] }} />
                    ) : null,
                  )}
                </span>
                <span className="bar-value">{Object.values(s.failures).reduce((a, b) => a + b, 0)} / {total}</span>
              </div>
            );
          })}
          <ul className="legend">
            {FAILURES.map((f, i) => (
              <li key={f}>
                <span className="swatch" style={{ background: FAILURE_COLORS[i] }} />
                {f.replace("_", " ")}
              </li>
            ))}
          </ul>
        </section>

        <section className="panel">
          <h2>Latency and cost</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Config</th>
                  <th scope="col">p50 ms</th>
                  <th scope="col">p95 ms</th>
                  <th scope="col">Tokens</th>
                  <th scope="col">Est. cost</th>
                </tr>
              </thead>
              <tbody>
                {configs.map(([name, s]) => (
                  <tr key={name}>
                    <th scope="row">{name}</th>
                    <td>{Math.round(s.latency_ms.total?.p50 ?? 0).toLocaleString()}</td>
                    <td>{Math.round(s.latency_ms.total?.p95 ?? 0).toLocaleString()}</td>
                    <td>{(s.usage.input_tokens + s.usage.output_tokens).toLocaleString()}</td>
                    <td>{s.cost_usd != null ? `$${s.cost_usd.toFixed(3)}` : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="hint">Measured on an 8 GB laptop under memory pressure; latency is not representative of a server.</p>
        </section>
      </div>
    </div>
  );
}
