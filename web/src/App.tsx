import { useCallback, useEffect, useState } from "react";
import { api, type Citation, type DocumentInfo, type Health } from "./api";
import BenchmarkPanel from "./components/BenchmarkPanel";
import ComparePanel from "./components/ComparePanel";
import Conversation, { type Turn } from "./components/Conversation";
import Sidebar from "./components/Sidebar";
import SourceViewer from "./components/SourceViewer";

type Tab = "ask" | "compare" | "benchmark";
const TABS: { id: Tab; label: string }[] = [
  { id: "ask", label: "Ask" },
  { id: "compare", label: "Compare" },
  { id: "benchmark", label: "Benchmark" },
];

function readTheme(): "light" | "dark" | null {
  try {
    const t = localStorage.getItem("theme");
    return t === "light" || t === "dark" ? t : null;
  } catch {
    return null;
  }
}

export default function App() {
  const [tab, setTab] = useState<Tab>("ask");
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [documents, setDocuments] = useState<DocumentInfo[] | null>(null);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [citation, setCitation] = useState<Citation | null>(null);
  const [theme, setTheme] = useState<"light" | "dark" | null>(readTheme);

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
    api.health().then(setHealth).catch(() => setHealthError(true));
    refreshDocuments();
  }, [refreshDocuments]);

  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
    try {
      if (theme) localStorage.setItem("theme", theme);
    } catch {
      // storage unavailable; theme still applies for this session
    }
  }, [theme]);

  const hasDocuments = (documents?.length ?? 0) > 0;
  const pageCount = documents?.find((d) => d.id === citation?.doc_id)?.page_count;
  const closeViewer = useCallback(() => setCitation(null), []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <h1>IntelliDoc</h1>
          <p className="status" role="status">
            <span className={healthError ? "status-dot bad" : health?.status === "ok" ? "status-dot ok" : "status-dot"} />
            {healthError ? "API unreachable" : health ? `${health.answer_model} · ${health.documents ?? 0} documents` : "Connecting…"}
          </p>
        </div>
        <nav aria-label="Views" role="tablist" className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              id={`tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls={`panel-${t.id}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <button
          type="button"
          className="ghost theme-toggle"
          onClick={() => setTheme(document.documentElement.dataset.theme === "dark" || (!theme && matchMedia("(prefers-color-scheme: dark)").matches) ? "light" : "dark")}
          aria-label="Toggle colour theme"
        >
          ◐
        </button>
      </header>

      {tab === "benchmark" ? (
        <main id="panel-benchmark" role="tabpanel" aria-labelledby="tab-benchmark">
          <BenchmarkPanel />
        </main>
      ) : (
        <main
          id={`panel-${tab}`}
          role="tabpanel"
          aria-labelledby={`tab-${tab}`}
          className={citation ? "workspace with-viewer" : "workspace"}
        >
          <Sidebar
            documents={documents}
            error={documentsError}
            selected={selectedDocs}
            onSelect={setSelectedDocs}
            onUploaded={() => {
              refreshDocuments();
              api.health().then(setHealth).catch(() => setHealthError(true));
            }}
          />
          {tab === "ask" ? (
            <Conversation
              turns={turns}
              setTurns={setTurns}
              selectedDocs={selectedDocs}
              hasDocuments={hasDocuments}
              activeCitation={citation}
              onOpenCitation={setCitation}
            />
          ) : (
            <ComparePanel selectedDocs={selectedDocs} hasDocuments={hasDocuments} activeCitation={citation} onOpenCitation={setCitation} />
          )}
          {citation && <SourceViewer citation={citation} pageCount={pageCount} onClose={closeViewer} />}
        </main>
      )}
    </div>
  );
}
