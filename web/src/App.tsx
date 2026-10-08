import { useCallback, useEffect, useState } from "react";
import { api, type DocumentInfo, type Health } from "./api";
import AskPanel from "./components/AskPanel";
import DocumentsPanel from "./components/DocumentsPanel";
import ResultsPanel from "./components/ResultsPanel";

type Tab = "ask" | "benchmark";

export default function App() {
  const [tab, setTab] = useState<Tab>("ask");
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [documents, setDocuments] = useState<DocumentInfo[] | null>(null);
  const [documentsError, setDocumentsError] = useState<string | null>(null);

  const refreshDocuments = useCallback(() => {
    api
      .documents()
      .then((docs) => {
        setDocuments(docs);
        setDocumentsError(null);
      })
      .catch((err: Error) => setDocumentsError(err.message));
  }, []);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch(() => setHealthError(true));
    refreshDocuments();
  }, [refreshDocuments]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>IntelliDoc</h1>
        <p className="status" role="status">
          {healthError
            ? "API unreachable"
            : health
              ? `${health.status === "ok" ? "Connected" : "Degraded"} · model ${health.answer_model}`
              : "Connecting…"}
        </p>
        <nav aria-label="Views">
          <div role="tablist">
            {(["ask", "benchmark"] as const).map((name) => (
              <button
                key={name}
                role="tab"
                id={`tab-${name}`}
                aria-selected={tab === name}
                aria-controls={`panel-${name}`}
                onClick={() => setTab(name)}
              >
                {name === "ask" ? "Ask" : "Benchmark"}
              </button>
            ))}
          </div>
        </nav>
      </header>

      {tab === "ask" ? (
        <main id="panel-ask" role="tabpanel" aria-labelledby="tab-ask" className="layout">
          <DocumentsPanel
            documents={documents}
            error={documentsError}
            onUploaded={refreshDocuments}
          />
          <AskPanel hasDocuments={(documents?.length ?? 0) > 0} />
        </main>
      ) : (
        <main id="panel-benchmark" role="tabpanel" aria-labelledby="tab-benchmark">
          <ResultsPanel />
        </main>
      )}
    </div>
  );
}
