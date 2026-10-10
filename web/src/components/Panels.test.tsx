import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { DocumentInfo } from "../api";
import { answer, jsonResponse, stubFetch } from "../test-utils";
import BenchmarkPanel from "./BenchmarkPanel";
import ComparePanel from "./ComparePanel";
import Sidebar from "./Sidebar";

afterEach(() => vi.unstubAllGlobals());

const docs: DocumentInfo[] = [
  { id: "a", filename: "irs.pdf", title: "Standard Deduction", page_count: 31, ocr_pages: 0, created_at: "" },
  { id: "b", filename: "nasa.pdf", title: "NASA TN D-233", page_count: 36, ocr_pages: 36, created_at: "" },
];

describe("Sidebar", () => {
  it("filters documents and reports the selection scope", async () => {
    const onSelect = vi.fn();
    render(<Sidebar documents={docs} error={null} selected={[]} onSelect={onSelect} onUploaded={() => {}} />);
    await userEvent.type(screen.getByLabelText("Filter documents"), "nasa");
    expect(screen.queryByText("Standard Deduction")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("checkbox"));
    expect(onSelect).toHaveBeenCalledWith(["b"]);
    expect(screen.getByText("Asking across all documents")).toBeInTheDocument();
  });
});

describe("ComparePanel", () => {
  it("asks once per retrieval mode and shows each answer", async () => {
    const fetchMock = stubFetch((_, init) => {
      const { mode } = JSON.parse(init!.body as string) as { mode: string };
      return jsonResponse(answer({ answer: `answer from ${mode}`, config: `structure-${mode}` }));
    });
    render(<ComparePanel selectedDocs={[]} hasDocuments activeCitation={null} onOpenCitation={() => {}} />);
    await userEvent.type(screen.getByLabelText("Question to compare"), "Who?");
    await userEvent.click(screen.getByRole("button", { name: "Compare" }));

    expect(await screen.findByText("answer from hybrid_rerank")).toBeInTheDocument();
    expect(screen.getByText("answer from dense")).toBeInTheDocument();
    expect(fetchMock.mock.calls.map((c) => JSON.parse(c[1]!.body as string).mode)).toEqual(["hybrid", "dense", "hybrid_rerank"]);
  });
});

const summary = (hit: number, refusal: number) => ({
  questions: 85, answerable: 69, evidence_hit_rate: hit, page_hit_rate: hit, mrr: 0.8, answer_accuracy: 0.9,
  citation_precision: 0.9, supported_rate: 1, judge_grounded_rate: 0.97, false_refusal_rate: 0.03,
  correct_refusal_rate: refusal, injection_followed_rate: 0, failures: { retrieval_miss: 4 },
  latency_ms: { total: { p50: 900, p95: 2000, n: 85 } }, usage: { input_tokens: 1000, output_tokens: 100, calls: 85 },
  cost_usd: 0.1, config: { name: "x", description: "" },
  by_category: { table: { questions: 13, answerable: 13, evidence_hit_rate: hit, correct_refusal_rate: null, failures: {} } },
});

describe("BenchmarkPanel", () => {
  it("switches the charted metric", async () => {
    stubFetch(() =>
      jsonResponse([{ run_id: "r1", mode: "openai", answer_model: "gpt-4.1-mini", cpu: "Apple M1", memory_gb: 8, summary: { c1: summary(0.913, 0.688), c3: summary(0.942, 1) } }]),
    );
    render(<BenchmarkPanel />);
    const bars = await screen.findByRole("list", { name: "Evidence hit@5 by configuration" });
    expect(within(bars).getByText("94.2%")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("radio", { name: "Correct refusal" }));
    const refusal = screen.getByRole("list", { name: "Correct refusal by configuration" });
    expect(within(refusal).getByText("68.8%")).toBeInTheDocument();
    expect(within(refusal).getByText("100.0%")).toBeInTheDocument();
  });
});
