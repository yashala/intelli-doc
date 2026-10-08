/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface Chunk {
  id: string;
  text: string;
  embedding: number[];
  metadata: {
    fileName: string;
    pageNumber?: number;
  };
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Chunk[];
  timestamp: number;
}

export interface DocumentState {
  isProcessing: boolean;
  fileName: string | null;
  chunkCount: number;
  error: string | null;
}
