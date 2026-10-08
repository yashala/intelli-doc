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
  const modelName = "gemini-embedding-2-preview";

  // Process chunks one by one for maximum reliability
  for (let i = 0; i < chunks.length; i++) {
    try {
      const result = await ai.models.embedContent({
        model: modelName,
        contents: [{ parts: [{ text: chunks[i] }] }]
      });

      // Handle response structure which might vary by SDK version or plural vs singular contents
      const embeddingValues = (result as any).embeddings?.[0]?.values || (result as any).embedding?.values;
      
      if (!embeddingValues) {
        throw new Error("No embedding values returned from API");
      }

      results.push({
        id: `chunk-${i}-${Date.now()}`,
        text: chunks[i],
        embedding: embeddingValues,
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
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
  
  let dotProduct = 0;
  let mA = 0;
  let mB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    mA += vecA[i] * vecA[i];
    mB += vecB[i] * vecB[i];
  }

  const denominator = Math.sqrt(mA) * Math.sqrt(mB);
  if (denominator === 0) return 0;
  return dotProduct / denominator;
}

/**
 * Finds top K similar chunks for a query.
 */
export async function semanticSearch(query: string, vectorStore: Chunk[], topK: number = 3): Promise<Chunk[]> {
  if (vectorStore.length === 0) return [];
  
  const modelName = "gemini-embedding-2-preview";

  const result = await ai.models.embedContent({
    model: modelName,
    contents: [{ parts: [{ text: query }] }]
  });

  const queryEmbedding = (result as any).embeddings?.[0]?.values || (result as any).embedding?.values;

  if (!queryEmbedding) {
    console.error("Failed to generate query embedding");
    return [];
  }

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
  
  const systemPrompt = `You are a helpful AI assistant named "Intelli doc". Answer the user's question using ONLY the provided context. 
If the answer is not in the context, say "I don't have enough information in the provided document to answer that."
Always cite the source filename in your answer.

Formatting Instructions:
1. Use professional Markdown.
2. Use clear headings (h2, h3) to structure different parts of your answer.
3. Organize information into distinct, readable paragraphs.
4. Use bullet points for lists.
5. Avoid excessive bolding of entire paragraphs; use bolding only for key terms.
6. Ensure each distinct concept or section of your answer occurs in its own paragraph with a descriptive title if appropriate.

CONTEXT:
${contextText}`;

  const response = await ai.models.generateContent({
    model: "gemini-3-flash-preview",
    contents: [{ parts: [{ text: query }] }],
    config: {
      systemInstruction: systemPrompt
    }
  });

  return response.text || "I'm sorry, I couldn't generate a response.";
}
