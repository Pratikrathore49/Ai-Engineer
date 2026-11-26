// src/shared/types/index.ts
// Cross-module shared TypeScript types.

export interface SourceCitation {
  filename: string
  chunk_index: number
  similarity_score: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceCitation[]
  timestamp: Date
}
