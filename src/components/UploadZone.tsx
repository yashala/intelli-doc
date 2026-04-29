/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useCallback, useState } from 'react';
import { Upload, FileText, Loader2, CheckCircle2, AlertCircle, HardDrive } from 'lucide-react';
import * as pdfjsLib from 'pdfjs-dist';
import { chunkText, generateEmbeddings } from '../lib/rag';
import { Chunk, DocumentState } from '../types';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// More robust worker setup for Vite/CDN
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`;

interface UploadZoneProps {
  onIndexComplete: (chunks: Chunk[]) => void;
}

interface DetailedState extends DocumentState {
  fileSize: string | null;
  pages?: number;
}

export default function UploadZone({ onIndexComplete }: UploadZoneProps) {
  const [state, setState] = useState<DetailedState>({
    isProcessing: false,
    fileName: null,
    chunkCount: 0,
    error: null,
    fileSize: null,
  });

  const formatSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const extractTextFromPdf = async (file: File): Promise<{ text: string, pages: number }> => {
    const arrayBuffer = await file.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;
    let fullText = '';
    
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items.map((item: any) => item.str).join(' ');
      fullText += pageText + '\n';
    }
    
    return { text: fullText, pages: pdf.numPages };
  };

  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setState({ 
      isProcessing: true, 
      fileName: file.name, 
      chunkCount: 0, 
      error: null, 
      fileSize: formatSize(file.size),
      pages: undefined
    });

    try {
      let text = '';
      let pages = 0;

      if (file.type === 'application/pdf') {
        const result = await extractTextFromPdf(file);
        text = result.text;
        pages = result.pages;
      } else {
        text = await file.text();
        pages = 1;
      }

      if (!text.trim()) {
        throw new Error('Document is empty or has no extractable text layer.');
      }

      const chunks = chunkText(text);
      const embeddedChunks = await generateEmbeddings(chunks, file.name);
      
      setState(prev => ({ 
        ...prev, 
        isProcessing: false, 
        chunkCount: embeddedChunks.length,
        pages 
      }));
      onIndexComplete(embeddedChunks);
    } catch (err) {
      console.error(err);
      setState(prev => ({ 
        ...prev, 
        isProcessing: false, 
        error: err instanceof Error ? err.message : 'Failed to process document' 
      }));
    }
  }, [onIndexComplete]);

  return (
    <div className="p-6 border-b border-[var(--color-border)]">
      <div className="flex items-center justify-between mb-4">
        <span className="section-label">Active Document</span>
        {state.isProcessing && (
          <div className="flex items-center gap-2 text-[var(--color-accent)] animate-pulse">
            <Loader2 className="w-3 h-3 animate-spin" />
            <span className="text-[10px] font-mono uppercase tracking-tighter">Analysing...</span>
          </div>
        )}
      </div>

      <div 
        className={cn(
          "relative bg-[var(--color-glass)] border border-[var(--color-border)] rounded-lg p-6 transition-all group",
          state.isProcessing ? "opacity-50 pointer-events-none" : "hover:border-[var(--color-accent)]/50 cursor-pointer"
        )}
      >
        <input
          type="file"
          accept=".pdf,.txt,.md"
          onChange={handleFileUpload}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
          disabled={state.isProcessing}
        />
        
        <div className="flex items-center gap-4">
          <div className="w-10 h-12 bg-[#222] border-t-4 border-[var(--color-accent)] rounded flex items-center justify-center shrink-0">
            <FileText className="w-5 h-5 text-[var(--color-accent)] opacity-40 group-hover:opacity-100 transition-opacity" />
          </div>
          <div className="overflow-hidden">
            <p className="text-sm font-medium text-[var(--color-text-primary)] truncate">
              {state.fileName ? state.fileName : "Select Knowledge Source"}
            </p>
            <p className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wider flex items-center gap-2">
              {state.chunkCount > 0 ? (
                <>
                  <CheckCircle2 className="w-3 h-3 text-green-500" />
                  {state.chunkCount} Fragments indexed
                </>
              ) : "PDF, TXT, MD"}
            </p>
          </div>
        </div>
      </div>

      {state.error && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 flex items-center gap-3 text-red-500 rounded-md">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <p className="text-[10px] font-mono uppercase tracking-tight">{state.error}</p>
        </div>
      )}

      {state.fileName && !state.isProcessing && (
        <div className="mt-6 pt-6 border-t border-[var(--color-border)]/50 space-y-4">
          <span className="section-label">Metadata Analysis</span>
          <div className="space-y-3 font-mono text-[11px]">
            <div className="flex justify-between">
              <span className="text-[var(--color-text-secondary)]">Resolution</span>
              <span className="text-[var(--color-text-primary)] uppercase tracking-tighter">Vectorized</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--color-text-secondary)]">File Weight</span>
              <span className="text-[var(--color-text-primary)] uppercase tracking-tighter">{state.fileSize}</span>
            </div>
            {state.pages && (
              <div className="flex justify-between">
                <span className="text-[var(--color-text-secondary)]">Scope</span>
                <span className="text-[var(--color-text-primary)] uppercase tracking-tighter">{state.pages} Pages</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}


