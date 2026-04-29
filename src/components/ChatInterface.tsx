/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Loader2, Quote } from 'lucide-react';
import { Message, Chunk } from '../types';
import { semanticSearch, generateAnswer } from '../lib/rag';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface ChatInterfaceProps {
  vectorStore: Chunk[];
}

export default function ChatInterface({ vectorStore }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || vectorStore.length === 0) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const relevantChunks = await semanticSearch(input, vectorStore);
      const answer = await generateAnswer(input, relevantChunks);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: answer || 'I encountered an error generating an answer.',
        sources: relevantChunks,
        timestamp: Date.now(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error(err);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'I failed to process your request. Please try again.',
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden bg-[var(--color-bg)]">
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-12 space-y-16 scroll-smooth"
      >
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-6">
            <div className="w-16 h-16 rounded-full border border-[var(--color-border)] flex items-center justify-center bg-[var(--color-surface)]">
              <Bot className="w-8 h-8 text-[var(--color-accent)] opacity-50" />
            </div>
            <div className="space-y-2">
              <h2 className="font-serif italic text-2xl text-[var(--color-text-primary)]">Intelli doc</h2>
              <p className="text-[10px] uppercase tracking-[3px] text-[var(--color-text-secondary)]">
                {vectorStore.length > 0 ? "Knowledge base online / Awaiting query" : "Awaiting document ingestion"}
              </p>
            </div>
          </div>
        )}

        <div className="max-w-3xl mx-auto w-full space-y-12">
          <AnimatePresence mode="popLayout">
            {messages.map((m) => (
              <motion.div
                key={m.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                layout
                className="flex flex-col gap-4"
              >
                <div className={cn(
                  "font-serif transition-opacity",
                  m.role === 'user' ? "text-2xl text-[var(--color-text-primary)] leading-relaxed" : "text-[15px] text-[var(--color-text-secondary)] leading-loose border-l-2 border-[var(--color-accent)] pl-8"
                )}>
                  {m.content}
                </div>

                {m.sources && m.sources.length > 0 && (
                  <div className="flex flex-wrap gap-2 pt-2 ml-8 opacity-60 hover:opacity-100 transition-opacity">
                    {m.sources.map((source, idx) => (
                      <div 
                        key={source.id} 
                        className="text-[10px] px-3 py-1 bg-[var(--color-surface)] border border-[var(--color-border)] rounded text-[var(--color-accent)] hover:border-[var(--color-accent)] transition-all cursor-help"
                        title={source.text}
                      >
                        § REF-{idx} / {source.metadata.fileName}
                      </div>
                    ))}
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="border-l-2 border-[var(--color-accent)] pl-8 flex items-center gap-4 text-[var(--color-text-secondary)] animate-pulse"
            >
              <Loader2 className="w-4 h-4 animate-spin text-[var(--color-accent)]" />
              <span className="text-xs uppercase tracking-widest">Reasoning...</span>
            </motion.div>
          )}
        </div>
      </div>

      <div className="p-8">
        <div className="max-w-3xl mx-auto w-full">
          <form 
            onSubmit={handleSubmit} 
            className="flex items-center gap-4 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-xl p-4 shadow-2xl focus-within:border-[var(--color-accent)]/50 transition-all"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading || vectorStore.length === 0}
              placeholder={vectorStore.length > 0 ? "Ask a question about the document..." : "Index a document first..."}
              className="flex-1 bg-transparent border-none focus:outline-none text-[15px] placeholder:text-[var(--color-text-secondary)]/50 text-[var(--color-text-primary)]"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim() || vectorStore.length === 0}
              className="px-6 py-2 bg-[var(--color-accent)] text-black rounded-lg text-[11px] font-bold uppercase tracking-widest hover:scale-105 transition-all disabled:opacity-30 disabled:hover:scale-100"
            >
              Analyze
            </button>
          </form>
          <p className="mt-4 text-[9px] text-center uppercase tracking-widest text-[var(--color-text-secondary)] opacity-30">
            Aether RAG Protocol v2.5 / Secure Neural Link
          </p>
        </div>
      </div>
    </div>
  );
}

