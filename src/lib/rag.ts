/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI } from "@google/genai";
import { Chunk } from "../types";

const ai = new GoogleGenAI({ 
  apiKey: process.env.GEMINI_API_KEY || '' 
});

/**
 * Splits text into overlapping chunks.
 */
export function chunkText(text: string, chunkSize: number = 1000, overlap: number = 200): string[] {
  const chunks: string[] = [];
  let startIndex = 0;

  while (startIndex < text.length) {
    const endIndex = Math.min(startIndex + chunkSize, text.length);
    chunks.push(text.slice(startIndex, endIndex));
    startIndex += chunkSize - overlap;
  }

  return chunks;
}

/**
 * Generates embeddings for a list of text chunks.
 */
export async function generateEmbeddings(chunks: string[], fileName: string): Promise<Chunk[]> {
  const results: Chunk[] = [];
  const model = "gemini-embedding-2-preview";

  // Process chunks one by one for maximum compatibility with SDK versions
  // and simpler error handling, although batching is faster.
  // Given the potentially small number of chunks in a typical user doc, this is safe.
  for (let i = 0; i < chunks.length; i++) {
    try {
      const response = await (ai as any).models.embedContent({
        model: model,
        contents: [{ parts: [{ text: chunks[i] }] }]
      });

      const embedding = response.embedding?.values || [];
      
      results.push({
        id: `chunk-${i}`,
        text: chunks[i],
        embedding,
        metadata: { fileName }
      });
    } catch (err) {
      console.error(`Error embedding chunk ${i}:`, err);
    }
  }

  return results;
}


/**
 * Calculates cosine similarity between two vectors.
 */
export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let mA = 0;
  let mB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    mA += vecA[i] * vecA[i];
    mB += vecB[i] * vecB[i];
  }

  return dotProduct / (Math.sqrt(mA) * Math.sqrt(mB));
}

/**
 * Finds top K similar chunks for a query.
 */
export async function semanticSearch(query: string, vectorStore: Chunk[], topK: number = 3): Promise<Chunk[]> {
  const model = "gemini-embedding-2-preview";

  const response = await (ai as any).models.embedContent({
    model: model,
    contents: [{ parts: [{ text: query }] }]
  });

  const queryEmbedding = response.embedding.values;

  const results = vectorStore.map(chunk => ({
    chunk,
    score: cosineSimilarity(queryEmbedding, chunk.embedding)
  }));

  return results
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(r => r.chunk);
}

/**
 * Generates an answer using retrieved context.
 */
export async function generateAnswer(query: string, contextChunks: Chunk[]) {
  const contextText = contextChunks.map(c => `[Source: ${c.metadata.fileName}]: ${c.text}`).join("\n\n---\n\n");
  
  const systemPrompt = `You are a helpful AI assistant. Answer the user's question using ONLY the provided context. 
If the answer is not in the context, say "I don't have enough information in the provided document to answer that."
Always cite the source filename in your answer.

CONTEXT:
${contextText}`;

  // Use 'gemini-3-flash-preview' as per skill instructions
  const response = await ai.models.generateContent({
    model: "gemini-3-flash-preview",
    contents: query,
    config: {
      systemInstruction: systemPrompt
    }
  });

  return response.text;
}
