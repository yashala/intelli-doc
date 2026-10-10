// Typed client for the IntelliDoc API. All requests go through the /api proxy; the browser
// never talks to a model provider and never holds a key.

export type Mode = "dense" | "hybrid" | "hybrid_rerank";

export interface DocumentInfo {
  id: string;
  filename: string;
  title: string;
  page_count: number;
  ocr_pages: number;
  created_at: string;
}

export interface IngestResult {
  doc_id: string;
  title: string;
  pages: number;
  ocr_pages: number;
  chunks: Record<string, number>;
  timings_ms: Record<string, number>;
}

export interface Citation {
  label: string;
  doc_id: string;
  title: string;
  filename: string;
  pages: number[];
  text: string;
}

export interface AskResponse {
  answer: string;
  refused: boolean;
  refusal_reason: string | null;
  citations: Citation[];
  config: string;
  model: string;
  prompt_version: string;
  timings_ms: Record<string, number>;
  usage: { input_tokens: number; output_tokens: number; calls: number };
}

export interface Health {
  status: "ok" | "degraded";
  database: boolean;
  documents: number | null;
  llm_mode: string;
  answer_model: string;
}

export interface LatencyStat {
  p50: number | null;
  p95: number | null;
  n: number;
}

export interface MetricSummary {
  questions: number;
  answerable: number;
  evidence_hit_rate: number | null;
  page_hit_rate: number | null;
  mrr: number | null;
  answer_accuracy: number | null;
  citation_precision: number | null;
  supported_rate: number | null;
  judge_grounded_rate?: number | null;
  false_refusal_rate: number | null;
  correct_refusal_rate: number | null;
  injection_followed_rate: number | null;
  failures: Record<string, number>;
}

export interface ConfigSummary extends MetricSummary {
  latency_ms: Record<string, LatencyStat>;
  by_category: Record<string, MetricSummary>;
  usage: { input_tokens: number; output_tokens: number; calls: number };
  cost_usd?: number;
  config: { name: string; description: string };
}

export interface BenchmarkRun {
  run_id: string;
  mode: string;
  answer_model: string;
  cpu: string;
  memory_gb: number;
  summary: Record<string, ConfigSummary>;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // keep statusText
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),
  documents: () => request<DocumentInfo[]>("/documents"),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<IngestResult>("/documents", { method: "POST", body: form });
  },
  ask: (question: string, mode: Mode, docIds: string[] = []) =>
    request<AskResponse>("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(docIds.length ? { question, mode, doc_ids: docIds } : { question, mode }),
    }),
  results: () => request<BenchmarkRun[]>("/results"),
  pageImageUrl: (docId: string, page: number) => `/api/documents/${docId}/pages/${page}.png`,
};

export const MODES: { value: Mode; label: string; hint: string }[] = [
  { value: "hybrid", label: "Hybrid", hint: "Dense vectors + Postgres full text, fused" },
  { value: "dense", label: "Dense", hint: "Embedding similarity only" },
  { value: "hybrid_rerank", label: "Hybrid + rerank", hint: "Cross-encoder reorders the top 20; slower" },
];

export const REFUSAL_REASONS: Record<string, string> = {
  insufficient_evidence: "The retrieved passages did not contain the answer, even after rewriting the query.",
  low_retrieval_similarity: "No passage was similar enough to the question.",
  no_valid_citation: "The model's answer did not cite any retrieved passage, so it was discarded.",
  no_documents: "No matching documents are indexed.",
};

export function totalMs(timings: Record<string, number>): number {
  return Object.values(timings).reduce((a, b) => a + b, 0);
}
