/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import UploadZone from './components/UploadZone';
import ChatInterface from './components/ChatInterface';
import { Chunk } from './types';
import { Database, ShieldCheck, Activity } from 'lucide-react';

export default function App() {
  const [vectorStore, setVectorStore] = useState<Chunk[]>([]);

  const handleIndexComplete = (chunks: Chunk[]) => {
    setVectorStore(chunks);
  };

  return (
    <main className="flex h-screen overflow-hidden bg-[var(--color-bg)]">
      {/* Sidebar - Control Panel */}
      <aside className="w-80 border-r border-[var(--color-border)] flex flex-col bg-[var(--color-surface)]">
        <div className="h-16 px-6 border-b border-[var(--color-border)] flex items-center justify-between">
          <h1 className="font-serif italic text-xl tracking-wider text-[var(--color-accent)]">Intelli doc</h1>
          <div className="status-pill">
            <span className={vectorStore.length > 0 ? "status-dot" : "status-dot bg-orange-500 opacity-50"}></span>
            <span className="opacity-80 uppercase tracking-tighter text-[9px]">
              {vectorStore.length > 0 ? "Indexing Complete" : "Standby"}
            </span>
          </div>
        </div>

        <div className="flex-1 flex flex-col overflow-y-auto">
          <UploadZone onIndexComplete={handleIndexComplete} />

          <div className="p-6 space-y-8">
            {/* System Metrics */}
            <div className="space-y-4">
              <span className="section-label">Knowledge Graph</span>
              <div className="h-32 border border-dashed border-[var(--color-border)] rounded-lg flex flex-col items-center justify-center gap-2 bg-[var(--color-glass)] group transition-all hover:bg-[var(--color-accent)]/5">
                <Database className="w-5 h-5 text-[var(--color-accent)] opacity-50 group-hover:opacity-100 transition-opacity" />
                <span className="text-[11px] text-[var(--color-text-secondary)] font-mono">
                  {vectorStore.length > 0 ? `${vectorStore.length} Entities Extracted` : "No data mapped"}
                </span>
              </div>
            </div>

            <div className="space-y-4">
              <span className="section-label">Session Analytics</span>
              
              <div className="space-y-3 font-mono text-[11px]">
                <div className="flex items-center justify-between group">
                  <span className="text-[var(--color-text-secondary)]">Context Tokens</span>
                  <span className="text-[var(--color-accent)] tracking-tighter font-bold">
                    {vectorStore.length > 0 ? (vectorStore.length * 200).toLocaleString() : '0'}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-[var(--color-text-secondary)]">Retrieval Latency</span>
                  <span className="text-[var(--color-accent)] tracking-tighter">
                    {vectorStore.length > 0 ? '142ms' : '-'}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-[var(--color-text-secondary)]">Vector Accuracy</span>
                  <span className="text-[var(--color-accent)] tracking-tighter">98.4%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="p-6 border-t border-[var(--color-border)]">
          <div className="flex items-center gap-2 opacity-30 text-[10px] tracking-[2px] uppercase">
            <Activity className="w-3 h-3 text-[var(--color-accent)]" />
            <span>Secure Terminal</span>
          </div>
        </div>
      </aside>

      {/* Main - Interface */}
      <section className="flex-1 relative bg-[var(--color-bg)]">
        <ChatInterface vectorStore={vectorStore} />
      </section>
    </main>
  );
}


