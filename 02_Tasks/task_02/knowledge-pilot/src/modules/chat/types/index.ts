// src/modules/chat/types/index.ts

export interface RAGRequest {
  question: string
  document_id?: string | null
  conversation_id?: string | null
  top_k?: number
}

export interface RAGResponse {
  answer: string
  conversation_id: string
  sources: SourceCitation[]
  chunks_used: number
}

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
