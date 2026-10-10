import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Citation } from "../api";
import { answer, jsonResponse, stubFetch } from "../test-utils";
import Conversation, { type Turn } from "./Conversation";

function Harness({ selected = [], onOpen = () => {} }: { selected?: string[]; onOpen?: (c: Citation) => void }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  return (
    <Conversation turns={turns} setTurns={setTurns} selectedDocs={selected} hasDocuments activeCitation={null} onOpenCitation={onOpen} />
  );
}

async function ask(text: string) {
  await userEvent.type(screen.getByLabelText("Question"), text);
  await userEvent.click(screen.getByRole("button", { name: "Ask" }));
}

afterEach(() => vi.unstubAllGlobals());

describe("Conversation", () => {
  it("keeps earlier turns and opens the first source of each answer", async () => {
    stubFetch(() => jsonResponse(answer()));
    const onOpen = vi.fn();
    render(<Harness onOpen={onOpen} />);
    await ask("Who replaced the probe?");
    await ask("And when?");

    expect(await screen.findAllByText("Medira Systems replaced it.")).toHaveLength(2);
    expect(screen.getByText("Who replaced the probe?", { exact: false })).toBeInTheDocument();
    expect(onOpen).toHaveBeenCalledWith(expect.objectContaining({ doc_id: "abc", pages: [2] }));
    expect(screen.getAllByRole("button", { name: /Clinic memo/ })).toHaveLength(2);
  });

  it("sends the selected documents and chosen mode", async () => {
    const fetchMock = stubFetch(() => jsonResponse(answer()));
    render(<Harness selected={["abc"]} />);
    await userEvent.click(screen.getByRole("radio", { name: "Dense" }));
    await ask("Who replaced the probe?");

    await screen.findByText("Medira Systems replaced it.");
    expect(JSON.parse(fetchMock.mock.calls[0]![1]!.body as string)).toEqual({
      question: "Who replaced the probe?",
      mode: "dense",
      doc_ids: ["abc"],
    });
  });

  it("explains refusals", async () => {
    stubFetch(() =>
      jsonResponse(answer({ answer: "The indexed documents do not contain enough evidence to answer this.", refused: true, refusal_reason: "insufficient_evidence", citations: [] })),
    );
    render(<Harness />);
    await ask("What is the capital of Australia?");
    expect(await screen.findByRole("heading", { name: "No answer" })).toBeInTheDocument();
    expect(screen.getByText(/did not contain the answer/)).toBeInTheDocument();
  });

  it("renders model output as text, not HTML", async () => {
    stubFetch(() => jsonResponse(answer({ answer: "<img src=x onerror=alert(1)>", citations: [] })));
    render(<Harness />);
    await ask("anything");
    expect(await screen.findByText("<img src=x onerror=alert(1)>")).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("shows API errors in the turn", async () => {
    stubFetch(() => jsonResponse({ detail: "database unavailable" }, 503));
    render(<Harness />);
    await ask("anything");
    expect(await screen.findByText("Request failed: database unavailable")).toBeInTheDocument();
  });
});
