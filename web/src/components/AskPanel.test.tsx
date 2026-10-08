import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import AskPanel from "./AskPanel";

function mockFetch(status: number, body: unknown) {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const base = {
  config: "structure-hybrid",
  model: "offline-extractive",
  prompt_version: "answer-v1",
  timings_ms: { retrieve_0: 20, generate_0: 1 },
  usage: { input_tokens: 0, output_tokens: 0, calls: 0 },
};

async function ask(question: string) {
  await userEvent.type(screen.getByLabelText("Question"), question);
  await userEvent.click(screen.getByRole("button", { name: "Ask" }));
}

afterEach(() => vi.unstubAllGlobals());

describe("AskPanel", () => {
  it("shows the answer and opens the cited page", async () => {
    const fetchMock = mockFetch(200, {
      ...base,
      answer: "Medira Systems replaced it.",
      refused: false,
      refusal_reason: null,
      citations: [
        { label: "S1", doc_id: "abc", title: "Clinic memo", filename: "memo.pdf", pages: [2], text: "Probe L12 was replaced by Medira Systems." },
      ],
    });
    render(<AskPanel hasDocuments />);
    await ask("Who replaced the probe?");

    expect(await screen.findByText("Medira Systems replaced it.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Clinic memo · p. 2" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("img", { name: "Page 2 of memo.pdf" })).toHaveAttribute("src", "/api/documents/abc/pages/2.png");
    expect(JSON.parse(fetchMock.mock.calls[0]![1].body)).toEqual({ question: "Who replaced the probe?", mode: "hybrid" });
  });

  it("explains why it refused", async () => {
    mockFetch(200, {
      ...base,
      answer: "The indexed documents do not contain enough evidence to answer this.",
      refused: true,
      refusal_reason: "insufficient_evidence",
      citations: [],
    });
    render(<AskPanel hasDocuments />);
    await ask("What is the capital of Australia?");

    expect(await screen.findByRole("heading", { name: "No answer" })).toBeInTheDocument();
    expect(screen.getByText("The retrieved passages did not contain the answer.")).toBeInTheDocument();
  });

  it("renders model output as text, not HTML", async () => {
    mockFetch(200, { ...base, answer: "<img src=x onerror=alert(1)>", refused: false, refusal_reason: null, citations: [] });
    render(<AskPanel hasDocuments />);
    await ask("anything");

    expect(await screen.findByText("<img src=x onerror=alert(1)>")).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("shows API errors", async () => {
    mockFetch(422, { detail: "The PDF could not be parsed" });
    render(<AskPanel hasDocuments />);
    await ask("anything");

    expect(await screen.findByText("Request failed: The PDF could not be parsed")).toBeInTheDocument();
  });

  it("disables asking until a document exists", () => {
    render(<AskPanel hasDocuments={false} />);
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
  });
});
