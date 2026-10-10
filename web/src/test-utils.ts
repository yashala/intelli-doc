import { vi } from "vitest";

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

/** Stub fetch with a function of (path, init) so tests can route by endpoint. */
export function stubFetch(handler: (path: string, init?: RequestInit) => Response | Promise<Response>) {
  const mock = vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
    Promise.resolve(handler(String(input).replace(/^\/api/, ""), init)),
  );
  vi.stubGlobal("fetch", mock);
  return mock;
}

export const answer = (overrides: Record<string, unknown> = {}) => ({
  answer: "Medira Systems replaced it.",
  refused: false,
  refusal_reason: null,
  citations: [
    { label: "S1", doc_id: "abc", title: "Clinic memo", filename: "memo.pdf", pages: [2], text: "Probe L12 was replaced by Medira Systems." },
  ],
  config: "structure-hybrid",
  model: "gpt-4.1-mini",
  prompt_version: "answer-v1",
  timings_ms: { retrieve_0: 40, generate_0: 900 },
  usage: { input_tokens: 600, output_tokens: 30, calls: 1 },
  ...overrides,
});
